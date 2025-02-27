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
import re

# Cargar variables de entorno
load_dotenv()

# Definición de tipos de documentos
DOCUMENT_TYPES = {
    "todos": "Todos los documentos",
    "constitucion": "Constitución",
    "convenio_internacional": "Convenios Internacionales",
    "ley": "Leyes",
    "codigo": "Códigos"
}

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
    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = ["todos"]

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
Eres un asistente legal altamente calificado especializado en leyes ecuatorianas.
Responde de manera clara y precisa con base en los documentos legales provistos.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
1. Si encontraste información relevante en los documentos legales proporcionados, responde utilizando esa información y cita la fuente específica (nombre del documento).
2. Si la pregunta hace referencia a artículos o secciones específicas de la ley y no has encontrado el contenido exacto, indícalo claramente.
3. Si no hay información en el contexto que responda directamente a la pregunta, pero puedes proporcionar información general sobre el tema legal, hazlo aclarando que es información general.
4. Si la pregunta no está relacionada con temas legales o está fuera del ámbito de la legislación ecuatoriana, proporciona una respuesta general basada en tu conocimiento.

Respuesta:
""")

# Prompt para generar preguntas más precisas basado en el historial
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Dado el historial de conversación: {chat_history} y la nueva pregunta: {question}, 
reestructura la pregunta si es necesario para hacerla más precisa.
Intenta identificar si la pregunta se refiere a artículos específicos de algún documento legal ecuatoriano.

Pregunta reestructurada:
""")

# Función para filtrar documentos por tipo
def create_filtered_retriever(vectorstore, selected_sources, question):
    # Detectar referencias a documentos específicos en la pregunta
    specific_source = None
    doc_type_filter = None
    
    # Patrones para detectar referencias a tipos de documentos
    constitution_pattern = r"(constituci[oó]n|carta magna)"
    code_pattern = r"(c[oó]digo\s+\w+)"
    law_pattern = r"(ley\s+\w+|ley\s+org[aá]nica\s+\w+|ley\s+de\s+\w+)"
    international_pattern = r"(convenio|tratado|acuerdo)\s+internacional"
    
    # Convertir a minúsculas para facilitar la detección
    question_lower = question.lower()
    
    # Filtrar por tipo de documento basado en patrones en la pregunta
    if re.search(constitution_pattern, question_lower):
        doc_type_filter = "constitucion"
    elif re.search(code_pattern, question_lower):
        doc_type_filter = "codigo"
        # Intentar extraer el nombre específico del código
        match = re.search(code_pattern, question_lower)
        if match:
            specific_source = match.group(0)
    elif re.search(law_pattern, question_lower):
        doc_type_filter = "ley"
        # Intentar extraer el nombre específico de la ley
        match = re.search(law_pattern, question_lower)
        if match:
            specific_source = match.group(0)
    elif re.search(international_pattern, question_lower):
        doc_type_filter = "convenio_internacional"
    
    # Definir el filtro basado en la selección del usuario y la detección en la pregunta
    filter_dict = {}
    
    # Si se seleccionó "todos" y no se detectó un tipo específico, no aplicar filtro
    if "todos" in selected_sources and not doc_type_filter:
        # No aplicar ningún filtro
        pass
    # Si se detectó un tipo específico en la pregunta, usar ese filtro
    elif doc_type_filter:
        filter_dict["doc_type"] = doc_type_filter
    # De lo contrario, usar los tipos seleccionados en la UI
    elif "todos" not in selected_sources:
        filter_dict["doc_type"] = {"$in": selected_sources}
    
    # Si se detectó una referencia a un documento específico, añadir al filtro
    if specific_source:
        # Crear un filtro más flexible para el nombre del archivo
        filter_dict["filename"] = {"$contains": specific_source}
    
    print(f"Filtro aplicado: {filter_dict}")
    
    # Crear retriever con filtro
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": filter_dict if filter_dict else None
        }
    )

# Creación de la cadena conversacional moderna
def create_conversation_chain(vectorstore, selected_sources):
    # Modelo de lenguaje
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
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
            "context": create_filtered_retriever(vectorstore, st.session_state.selected_sources, condensed_question)
                       .get_relevant_documents(condensed_question),
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
            "chat_history": chat_history,
            "selected_sources": st.session_state.selected_sources
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
    
    # Panel lateral para filtros
    with st.sidebar:
        st.title("Filtros de búsqueda")
        st.caption("Selecciona las fuentes legales para tu consulta:")
        
        # Lista de opciones de documentos
        source_options = list(DOCUMENT_TYPES.items())
        
        # Checkbox para cada tipo de documento
        selected = st.multiselect(
            "Buscar en:",
            options=[key for key, _ in source_options],
            default=["todos"],
            format_func=lambda x: DOCUMENT_TYPES.get(x, x)
        )
        
        # Actualizar estado de sesión solo si hay cambios
        if selected:
            st.session_state.selected_sources = selected
        else:
            st.session_state.selected_sources = ["todos"]
            st.warning("Se debe seleccionar al menos una fuente. Se ha seleccionado 'Todos los documentos' por defecto.")
    
    # Reconstruir la cadena de conversación con los nuevos filtros
    st.session_state.chain = create_conversation_chain(vectorstore, st.session_state.selected_sources)
    
    # Panel principal para chat
    user_question = st.text_input("Haz tu pregunta legal:")
    if user_question:
        handle_question(user_question)

if __name__ == "__main__":
    main()