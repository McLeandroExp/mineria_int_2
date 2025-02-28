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
            # "filter": filter_dict if filter_dict else None
        }
    )

def create_conversation_chain(vectorstore, selected_sources):
    """Crea la cadena de conversación RAG"""
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
    
    # Cadena completa de RAG conversacional
    chain = (
        {"question": itemgetter("question"), "chat_history": itemgetter("chat_history"), "selected_sources": itemgetter("selected_sources")}
        | condense_question_chain
        | (lambda condensed_question: {
            "context": create_filtered_retriever(vectorstore, selected_sources, condensed_question)
                       .get_relevant_documents(condensed_question),
            "question": condensed_question
        })
        | answer_chain
    )
    
    return chain