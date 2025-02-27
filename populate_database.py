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

# Definición de directorios
ROOT_DATA_PATH = "data"
DOCUMENT_TYPES = {
    "01_constitucion": "constitucion",
    "02_convenios_internacionales": "convenio_internacional",
    "03_leyes": "ley",
    "04_codigos": "codigo"
}

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

    # Cargar documentos de cada directorio
    all_documents = []
    for subdir, doc_type in DOCUMENT_TYPES.items():
        dir_path = os.path.join(ROOT_DATA_PATH, subdir)
        if os.path.exists(dir_path):
            print(f"📚 Cargando documentos de {subdir} (tipo: {doc_type})")
            documents = load_documents_from_directory(dir_path, doc_type)
            all_documents.extend(documents)
        else:
            print(f"⚠️ El directorio {dir_path} no existe.")

    # Procesar y almacenar documentos
    if all_documents:
        chunks = split_documents(all_documents)
        add_to_pinecone(chunks, index_name, pc)
    else:
        print("❌ No se encontraron documentos para procesar.")

def load_documents_from_directory(directory_path, doc_type):
    document_loader = PyPDFDirectoryLoader(directory_path)
    documents = document_loader.load()
    
    # Aplicar preprocesamiento y agregar metadatos de tipo de documento
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
        doc.metadata["doc_type"] = doc_type
        
        # Extraer el nombre base del archivo para facilitar búsquedas específicas
        source_path = doc.metadata.get("source", "")
        doc.metadata["filename"] = os.path.basename(source_path)
    
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
    for i, chunk in enumerate(chunks[:3]):  # Mostrar solo los primeros 3 para no saturar la consola
        print(f"\nChunk {i + 1}:")
        print(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)
    
    print(f"Total de chunks generados: {len(chunks)}")
    return chunks

def add_to_pinecone(chunks: list[Document], index_name: str, pc: Pinecone):
    embedding_function = get_embedding_function()
    
    # Asegurarse de que el índice existe
    ensure_index_exists(pc, index_name)
    
    # Añadir fecha de creación a los metadatos
    timestamp = datetime.now().isoformat()
    for chunk in chunks:
        if "id" not in chunk.metadata:
            # Crear ID único basado en la fuente, página, tipo y UUID
            source = chunk.metadata.get("filename", "unknown")
            page = chunk.metadata.get("page", "0")
            doc_type = chunk.metadata.get("doc_type", "unknown")
            unique_id = str(uuid.uuid4())[:8]
            chunk.metadata["id"] = f"{doc_type}:{source}:{page}:{unique_id}"
        
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
        region = os.getenv("PINECONE_REGION", "us-east-1")  # Valor predeterminado us-east-1
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