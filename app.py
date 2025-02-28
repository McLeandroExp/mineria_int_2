import streamlit as st
from langchain.memory import ConversationBufferMemory
from src.populate_database import load_pinecone
from src.utils.chains import create_conversation_chain
from src.htmlTemplates import css, bot_template, user_template
from src.config import DOCUMENT_TYPES, MEMORY_K

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
            k=MEMORY_K                    
        )
    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = ["todos"]

def handle_question(question):
    if not question.strip():
        st.warning("Por favor, ingresa una pregunta válida.")
        return

    with st.spinner("Cargando..."):
        # Obtener historial de chat del estado de la sesión para la memoria
        chat_history = []
        for role, msg in st.session_state.chat_history:
            chat_history.append({"role": role, "content": msg})
                
        # Invocar la cadena con la nueva pregunta y el historial
        response_data = st.session_state.chain.invoke({
            "question": question, 
            "chat_history": chat_history,
            "selected_sources": st.session_state.selected_sources
        })
        
        respuesta = response_data["response"]
        contexto = response_data["context"]
        pregunta_reformulada = response_data["question"]

        # Imprimir en consola para depuración
        print("----- DEBUG INFO -----")
        print("Pregunta original:", question)
        print("Pregunta reformulada:", pregunta_reformulada)
        print("Contexto enviado al modelo:", contexto)
        print("Fuentes utilizadas:")
        for idx, fuente in enumerate(contexto):
            print(f"Fuente {idx + 1}: {fuente.metadata['source']}")
            print(f"Contenido: {fuente.page_content[:500]}...")  # Mostrar solo una parte para evitar saturación

    # Actualizar historial de chat
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", respuesta))

    # Renderizar historial de chat
    for role, msg in st.session_state.chat_history:
        template = user_template if role == "user" else bot_template
        st.write(template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    # Mostrar fuentes como desplegables con contenido completo
    st.subheader("📚 Fuentes utilizadas:")
    for idx, fuente in enumerate(contexto):
        with st.expander(f"Fuente {idx + 1}: {fuente.metadata['source']}"):
            st.write(fuente.page_content)

def build_sidebar():
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

# Función principal
def main():
    setup()
    vectorstore = load_pinecone()
    
    # Panel lateral para filtros
    build_sidebar()
    
    # Reconstruir la cadena de conversación con los nuevos filtros
    st.session_state.chain = create_conversation_chain(vectorstore, st.session_state.selected_sources)
    
    # Panel principal para chat
    user_question = st.text_input("Haz tu pregunta legal:")
    if user_question:
        handle_question(user_question)

if __name__ == "__main__":
    main()
