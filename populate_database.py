import argparse
import os
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_pinecone import PineconeVectorStore
from text_preprocessing import preprocess_text
from dotenv import load_dotenv
from datetime import datetime
import uuid
from pinecone import Pinecone, ServerlessSpec
import time

# Cargar variables de entorno
load_dotenv()

DATA_PATH = "data"

def main():
    # Verificar si se debe limpiar la base de datos (usando el flag --reset).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    # Inicializar Pinecone con la nueva API
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Si se solicita reiniciar la base de datos
    if args.reset:
        print("✨ Clearing Database")
        clear_database(pc, index_name)

    # Crear (o actualizar) el almacén de datos.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_pinecone(chunks, index_name, pc)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    
    # Aplicar preprocesamiento a cada documento
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
    
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    
    # Imprimir los chunks generados
    print("📄 Chunks generados:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk.page_content)
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)
    
    return chunks

def add_to_pinecone(chunks: list[Document], index_name: str, pc: Pinecone):
    embedding_function = get_embedding_function()
    
    # Asegurarse de que el índice existe
    ensure_index_exists(pc, index_name)
    
    # Añadir fecha de creación a los metadatos
    timestamp = datetime.now().isoformat()
    for chunk in chunks:
        if "id" not in chunk.metadata:
            # Crear ID único basado en la fuente, página y UUID
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "0")
            unique_id = str(uuid.uuid4())[:8]  # Usar los primeros 8 caracteres del UUID
            chunk.metadata["id"] = f"{source}:{page}:{unique_id}"
        
        # Añadir fecha de creación
        chunk.metadata["created_at"] = timestamp
        # Asegurarse de que el texto está accesible en el campo text_key para Pinecone
        chunk.metadata["text"] = chunk.page_content
    
    # Crear o actualizar el almacén vectorial
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_function,
        index_name=index_name,
        text_key="text"
    )
    
    print(f"✅ Documentos añadidos a Pinecone exitosamente")

def ensure_index_exists(pc: Pinecone, index_name: str):
    """Asegura que el índice existe, si no, lo crea"""
    # Obtener lista de índices disponibles
    indexes = pc.list_indexes()
    
    # Comprobar si el índice existe
    if index_name not in indexes.names():
        print(f"🔍 Creando índice {index_name}")
        
        # Crear índice con dimensiones para el modelo de embeddings
        # Usar ServerlessSpec si utilizas Pinecone serverless
        region = os.getenv("PINECONE_REGION", "us-east-1")  # Valor predeterminado us-west-2
        cloud = os.getenv("PINECONE_CLOUD", "aws")         # Valor predeterminado aws
        
        pc.create_index(
            name=index_name,
            dimension=3072,  # dimensión para text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        # Esperar a que el índice esté listo
        print("⏳ Esperando a que el índice esté listo...")
        time.sleep(60)  # Dar tiempo adicional para índices serverless
    else:
        print(f"✅ Índice {index_name} ya existe")

def clear_database(pc: Pinecone, index_name: str):
    """Elimina y recrea el índice en Pinecone"""
    try:
        indexes = pc.list_indexes()
        if index_name in indexes.names():
            print(f"🗑️ Eliminando índice {index_name}")
            pc.delete_index(index_name)
            print(f"🗑️ Índice {index_name} eliminado")
            time.sleep(20)
            
            # Recrear el índice después de eliminarlo
            ensure_index_exists(pc, index_name)
    except Exception as e:
        print(f"Error al reiniciar la base de datos: {e}")

if __name__ == "__main__":
    main()