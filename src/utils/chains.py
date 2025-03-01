# src/utils/chains.py
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from src.config import MODEL_NAME, TEMPERATURE, OPENAI_API_KEY, RETRIEVER_K
from src.prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT
from src.utils.filters import create_filter_dict

def create_filtered_retriever(vectorstore, selected_sources, question):
    """Crea un retriever filtrado basado en la selección y la pregunta"""
    filter_dict = create_filter_dict(selected_sources, question)
    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVER_K,
            # Si deseas aplicar el filtro, descomenta la siguiente línea:
            # "filter": filter_dict if filter_dict else None
        }
    )

def create_conversation_chain(vectorstore, selected_sources):
    """Crea la cadena de conversación RAG con debug para imprimir información extra"""
    # Modelo de lenguaje
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Cadena para condensar la pregunta basada en el historial
    condense_question_chain = (
        {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )
    
    # Cadena para responder
    answer_chain = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    
    def debug_retrieve(condensed_question):
        """Recupera documentos, imprime en consola el contexto y la pregunta reformulada."""
        retriever = create_filtered_retriever(vectorstore, selected_sources, condensed_question)
        docs = retriever.get_relevant_documents(condensed_question)
        print("----- DEBUG RETRIEVE -----")
        print("Pregunta reformulada:", condensed_question)
        print("Documentos recuperados:")
        for doc in docs:
            filename = doc.metadata.get("filename", "N/A")
            print(f"  - Fuente: {filename}")
            snippet = doc.page_content[:2000] if len(doc.page_content) > 2000 else doc.page_content
            print(f"    Contenido (snippet): {snippet}")
            print("  ----------------------")
        return {"context": docs, "question": condensed_question}
    
    def debug_answer(inputs):
        answer = answer_chain.invoke(inputs)
        print("----- DEBUG ANSWER -----")
        print("Respuesta generada:", answer)
        return {"response": answer, "context": inputs.get("context"), "question": inputs.get("question")}

    
    chain = (
        {"question": itemgetter("question"), "chat_history": itemgetter("chat_history"), "selected_sources": itemgetter("selected_sources")}
        | condense_question_chain
        | debug_retrieve
        | debug_answer
    )
    
    return chain
