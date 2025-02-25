import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from get_embedding_function import get_embedding_function
from text_preprocessing import preprocess_text
from htmlTemplates import css, bot_template, user_template
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from operator import itemgetter

# Cargar variables de entorno
load_dotenv()

# Configuración inicial
def setup():
    st.set_page_config(page_title="LegisChat", page_icon="⚖️")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat con Documentos Legales ⚖️")
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5                    
        )

# Carga la base de datos vectorial Pinecone
def load_pinecone():
    embedding_function = get_embedding_function()
    # Asegúrate de tener estas variables en tu archivo .env
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Inicializar Pinecone con la nueva API
    Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_function,
        text_key="text"  # El campo que contiene el texto en Pinecone
    )

# Definir el prompt con "context"
ANSWER_PROMPT = PromptTemplate.from_template("""
Eres un asistente legal altamente calificado.
Responde de manera clara y precisa con base en los documentos legales provistos.
Si no puedes responder con certeza, indica que no tienes suficiente información.

Contexto:
{context}

Pregunta: {question}

Respuesta:
""")

# Prompt para generar preguntas más precisas basado en el historial
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Dado el historial de conversación: {chat_history} y la nueva pregunta: {question}, 
reestructura la pregunta si es necesario para hacerla más precisa.

Pregunta reestructurada:
""")

# Creación de la cadena conversacional moderna
def create_conversation_chain(vectorstore):
    # Modelo de lenguaje
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
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
        {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
        | condense_question_chain
        | (lambda condensed_question: {
            "context": retriever.get_relevant_documents(condensed_question),
            "question": condensed_question
        })
        | answer_chain
    )
    
    return chain

def handle_question(question):
    processed_question = preprocess_text(question)
    if not processed_question.strip():
        st.warning("Por favor, ingresa una pregunta válida.")
        return

    with st.spinner("Cargando..."):
        # Obtener historial de chat del estado de la sesión para la memoria
        chat_history = []
        for role, msg in st.session_state.chat_history:
            if role == "user":
                chat_history.append({"role": "user", "content": msg})
            else:
                chat_history.append({"role": "assistant", "content": msg})
                
        # Invocar la cadena con la nueva pregunta y el historial
        response = st.session_state.chain.invoke({
            "question": processed_question, 
            "chat_history": chat_history
        })
    
    # Actualizar historial de chat
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", response))

    # Renderizar historial de chat
    for role, msg in st.session_state.chat_history:
        template = user_template if role == "user" else bot_template
        st.write(template.replace("{{MSG}}", msg), unsafe_allow_html=True)

# Función principal
def main():
    setup()
    vectorstore = load_pinecone()
    
    if st.session_state.chain is None:
        st.session_state.chain = create_conversation_chain(vectorstore)
    
    user_question = st.text_input("Haz tu pregunta legal:")
    if user_question:
        handle_question(user_question)

if __name__ == "__main__":
    main()