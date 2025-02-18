# import streamlit as st
# from langchain.vectorstores.chroma import Chroma
# from langchain_community.llms.ollama import Ollama
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from get_embedding_function import get_embedding_function
# from text_preprocessing import preprocess_text
# from htmlTemplates import css, bot_template, user_template

# # Configuración inicial
# def setup():
#     st.set_page_config(page_title="LegisChat", page_icon="⚖️")
#     st.write(css, unsafe_allow_html=True)
#     st.header("Chat con Documentos Legales ⚖️")
    
#     if "conversation_chain" not in st.session_state:
#         st.session_state.conversation_chain = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

# # Cargar Chroma
# def load_chroma():
#     embedding_function = get_embedding_function()
#     return Chroma(persist_directory="chroma", embedding_function=embedding_function)

# # Definir el prompt personalizado
# PROMPT_TEMPLATE = """
# Eres un asistente especializado en leyes y propuestas legislativas. Tu tarea es responder preguntas relacionadas con el contenido de propuestas de ley y artículos legales, utilizando únicamente la información proporcionada en el contexto. Sigue estas pautas:

# 1. **Contexto legal**: Responde como si fueras un experto en leyes, utilizando un lenguaje formal y técnico adecuado para el ámbito jurídico.
# 2. **Precisión**: Basa tu respuesta estrictamente en el contexto proporcionado. Si no hay información suficiente, indica que no puedes responder con certeza.
# 3. **Claridad**: Explica los conceptos de manera clara y estructurada, utilizando términos jurídicos correctos.
# 4. **Formato**: Si es necesario, organiza la respuesta en puntos o párrafos para facilitar la lectura.

# Contexto proporcionado:
# {context}

# ---

# Pregunta: {question}

# Respuesta (en español):
# """

# # Crear cadena de conversación
# def create_conversation_chain(vectorstore):
#     llm = Ollama(model="llama3.2")
    
#     # Memoria para el historial
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer"
#     )
    
#     # Definir el prompt personalizado
#     prompt = PromptTemplate(
#         template=PROMPT_TEMPLATE,
#         input_variables=["context", "question"]
#     )
    
#     # Cadena de conversación con el prompt personalizado
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),  # Usamos el retriever estándar
#         memory=memory,
#         return_source_documents=True,
#         combine_docs_chain_kwargs={"prompt": prompt}  # Pasamos el prompt personalizado
#     )

# # Manejar preguntas
# def handle_question(question):
#     # Preprocesar la pregunta antes de pasarla a la cadena
#     processed_question = preprocess_text(question)
    
#     # Obtener la respuesta del modelo
#     response = st.session_state.conversation_chain({"question": processed_question})
    
#     # Mostrar historial
#     for msg in response["chat_history"]:
#         if msg.type == "human":
#             st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# # Función principal
# def main():
#     setup()
#     vectorstore = load_chroma()
    
#     if st.session_state.conversation_chain is None:
#         st.session_state.conversation_chain = create_conversation_chain(vectorstore)
    
#     user_question = st.text_input("Haz tu pregunta legal:")
#     if user_question:
#         handle_question(user_question)

# if __name__ == "__main__":
#     main()

import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from get_embedding_function import get_embedding_function
from text_preprocessing import preprocess_text
from htmlTemplates import css, bot_template, user_template

# Configuración inicial
def setup():
    st.set_page_config(page_title="LegisChat", page_icon="⚖️")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat con Documentos Legales ⚖️")
    
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5                    
        )

# Carga la base de datos vectorial ChromaDB
def load_chroma():
    embedding_function = get_embedding_function()
    return Chroma(persist_directory="chroma/", embedding_function=embedding_function)

# Definir el prompt con "context"
prompt = PromptTemplate(
    template="""
    Eres un asistente legal altamente calificado.
    Responde de manera clara y precisa con base en los documentos legales provistos.
    Si no puedes responder con certeza, indica que no tienes suficiente información.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta:
    """,
    input_variables=["context", "question"]
)

# Creación de la cadena conversacional
def create_conversation_chain(vectorstore):
    llm = Ollama(model="llama3.2")
    memory = st.session_state.memory
    retriever = vectorstore.as_retriever()
    
    # Generador de preguntas basado en el historial
    question_prompt = PromptTemplate(
        template="Dado el historial de conversación: {chat_history} y la nueva pregunta: {question}, reestructura la pregunta si es necesario.",
        input_variables=["chat_history", "question"]
    )
    question_generator = LLMChain(llm=llm, prompt=question_prompt)

    # Cadena para responder preguntas basadas en documentos
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        question_generator=question_generator,  # ✅ Ahora se incluye correctamente
        combine_docs_chain=qa_chain  # ✅ Se pasa en el formato correcto
    )

def handle_question(question):
    processed_question = preprocess_text(question)
    if not processed_question.strip():
        st.warning("Por favor, ingresa una pregunta válida.")
        return

    with st.spinner("Cargando..."):  # 🔹 Agrega un spinner normal
        response = st.session_state.conversation_chain({"question": processed_question})
    
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", response["answer"]))

    # Renderizar historial de chat
    for role, msg in st.session_state.chat_history:
        template = user_template if role == "user" else bot_template
        st.write(template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    # Mostrar documentos fuente si existen
    for doc in response.get("source_documents", []):
        st.markdown(f"**Fuente:** {doc.metadata.get('source', 'Desconocida')}")



# Función principal
def main():
    setup()
    vectorstore = load_chroma()
    
    if st.session_state.conversation_chain is None:
        st.session_state.conversation_chain = create_conversation_chain(vectorstore)
    
    user_question = st.text_input("Haz tu pregunta legal:")
    if user_question:
        handle_question(user_question)

if __name__ == "__main__":
    main()
