import argparse
import os
import re
import unicodedata
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from .get_embedding_function import get_embedding_function
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from datetime import datetime
import uuid
from pinecone import Pinecone, ServerlessSpec
import time
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from .utils.normalize_filename import normalize_filename
from .multi_representation import generate_summary

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
        clear_database(pc, index_name)

    # Verificar la existencia del índice
    ensure_index_exists(pc, index_name)
    
    # Rastrear archivos ya procesados
    existing_files = get_existing_files(pc, index_name) if not args.reset else set()
    
    # Cargar documentos de cada directorio
    new_documents = []
    for subdir, doc_type in DOCUMENT_TYPES.items():
        dir_path = os.path.join(ROOT_DATA_PATH, subdir)
        if os.path.exists(dir_path):
            documents = load_new_documents(dir_path, doc_type, existing_files)
            new_documents.extend(documents)

    # Procesar y almacenar documentos nuevos
    if new_documents:
        chunks = split_documents(new_documents)
        add_to_pinecone(chunks, index_name, pc)
    else:
        print("No se encontraron nuevos documentos para procesar.")

def to_ascii_id(text):
    """
    Convierte un texto a un ID compatible con ASCII.
    Elimina acentos, espacios, caracteres especiales y paréntesis.
    """
    # Normalizar para separar acentos de las letras
    text = unicodedata.normalize('NFKD', text)
    # Filtrar caracteres no ASCII y reemplazar caracteres problemáticos
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Eliminar caracteres no ASCII
    text = re.sub(r'[^\w\s-]', '', text)       # Eliminar caracteres especiales
    text = re.sub(r'[\(\)\[\]\{\}]', '', text) # Eliminar paréntesis y corchetes
    text = re.sub(r'\s+', '_', text)           # Reemplazar espacios con guiones bajos
    return text

def load_pinecone():
    """Carga la base de datos vectorial Pinecone"""
    embedding_function = get_embedding_function()
    
    # Inicializar Pinecone con la nueva API
    Pinecone(api_key=PINECONE_API_KEY)
    
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_function,
        text_key="text"  # El campo que contiene el texto en Pinecone
    )

def get_existing_files(pc, index_name):
    """Obtiene la lista de archivos ya indexados en Pinecone"""
    try:
        index = pc.Index(index_name)
        # Obtener una muestra de vectores para extraer los metadatos
        query_response = index.query(
            vector=[0.0] * 3072,  # Vector de consulta dummy
            top_k=10000,          # Número máximo de registros a recuperar
            include_metadata=True
        )
        
        # Extraer los nombres de archivo únicos
        existing_files = set()
        for match in query_response.matches:
            if 'filename' in match.metadata and 'source' in match.metadata:
                existing_files.add(match.metadata['source'])
                
        return existing_files
    except Exception as e:
        print(f"Error al obtener archivos existentes: {e}")
        return set()

def load_new_documents(directory_path, doc_type, existing_files):
    """Carga solo documentos nuevos que no están en existing_files"""
    document_loader = PyPDFDirectoryLoader(directory_path)
    documents = document_loader.load()
    
    new_documents = []
    for doc in documents:
        source_path = doc.metadata.get("source", "")
        
        # Verificar si el documento ya existe en la base de datos
        if source_path not in existing_files:
            # Extraer y normalizar el nombre base del archivo
            original_filename = os.path.basename(source_path)
            doc.metadata["filename"] = normalize_filename(original_filename)
            doc.metadata["doc_type"] = doc_type
            new_documents.append(doc)
    
    if new_documents:
        print(f"Encontrados {len(new_documents)} documentos nuevos en {directory_path}")
    
    return new_documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks generados: {len(chunks)}")
    return chunks

def add_to_pinecone(chunks: list[Document], index_name: str, pc: Pinecone):
    embedding_function = get_embedding_function()
    
    # Añadir fecha de creación y generar ID apropiado para cada chunk
    timestamp = datetime.now().isoformat()
    
    for chunk in chunks:
        # Obtener metadatos necesarios
        source = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page", "0")
        page_label = chunk.metadata.get("page_label", page)
        doc_type = chunk.metadata.get("doc_type", "unknown")
        unique_id = str(uuid.uuid4())[:8]
        
        # Asegurar que los componentes del ID sean ASCII
        ascii_doc_type = to_ascii_id(doc_type)
        ascii_source = to_ascii_id(source)
        
        # Generar ID que será usado como ID primario en Pinecone (solo ASCII)
        chunk_id = f"{ascii_doc_type}:{ascii_source}:{page}:{unique_id}"
        
        # Establecer el ID tanto en el metadato como para el vector
        chunk.metadata["id"] = chunk_id
        
        # Añadir fecha de creación
        chunk.metadata["created_at"] = timestamp
        
        # Guardar los valores originales en los metadatos para búsquedas
        chunk.metadata["original_doc_type"] = doc_type
        chunk.metadata["original_filename"] = source
        
        # Generar el contexto completo (que incluye metadatos y contenido original)
        full_text = f"Tipo: {doc_type}. Archivo: {source}. Página: {page_label}. {chunk.page_content}"
        
        # Generar resumen optimizado para la búsqueda usando multi representation
        try:
            summary_text = generate_summary(full_text)
        except Exception as e:
            print(f"Error al generar resumen para el chunk: {e}")
            summary_text = full_text  # fallback a contenido completo si falla el resumen
        
        # Asignar el resumen para la búsqueda y conservar el contexto completo para la respuesta
        chunk.metadata["full_text"] = full_text
        chunk.metadata["text"] = summary_text
        # Actualizar el contenido de la página para que, al mostrar el contexto, se vea el texto completo
        chunk.page_content = full_text
    
    # Crear vectores con IDs personalizados
    vectors_with_ids = []
    
    # Obtener embeddings para todos los chunks usando la representación resumen
    texts = [chunk.metadata["text"] for chunk in chunks]
    embeddings = embedding_function.embed_documents(texts)
    
    # Crear vectores con IDs personalizados
    for i, chunk in enumerate(chunks):
        vectors_with_ids.append({
            "id": chunk.metadata["id"],
            "values": embeddings[i],
            "metadata": chunk.metadata
        })
    
    # Insertar directamente en el índice de Pinecone
    index = pc.Index(index_name)
    
    # Insertar en lotes para manejar grandes cantidades de datos
    batch_size = 100
    for i in range(0, len(vectors_with_ids), batch_size):
        batch = vectors_with_ids[i:i+batch_size]
        index.upsert(vectors=batch)
    
    print(f"Documentos añadidos a Pinecone exitosamente")

def ensure_index_exists(pc: Pinecone, index_name: str):
    """Asegura que el índice existe, si no, lo crea"""
    indexes = pc.list_indexes()
    
    if index_name not in indexes.names():
        region = os.getenv("PINECONE_REGION", "us-east-1")
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        
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
        time.sleep(60)  # Dar tiempo adicional para índices serverless

def clear_database(pc: Pinecone, index_name: str):
    """Vacía el índice sin eliminarlo por completo"""
    try:
        indexes = pc.list_indexes()
        if index_name in indexes.names():
            index = pc.Index(index_name)
            
            # Vaciar el índice sin eliminarlo (más eficiente)
            index.delete(delete_all=True)
            print(f"Índice {index_name} vaciado correctamente")
            
            # Esperar brevemente para asegurar que la operación se complete
            time.sleep(5)
    except Exception as e:
        print(f"Error al vaciar la base de datos: {e}")

if __name__ == "__main__":
    main()