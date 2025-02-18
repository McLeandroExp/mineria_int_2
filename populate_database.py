import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from text_preprocessing import preprocess_text  # Importar la función de preprocesamiento

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    # Verificar si se debe limpiar la base de datos (usando el flag --reset).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Crear (o actualizar) el almacén de datos.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

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

def add_to_chroma(chunks: list[Document]):
    # Cargar la base de datos existente.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calcular IDs de los chunks.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Agregar o actualizar los documentos.
    existing_items = db.get(include=[])  # Los IDs siempre están incluidos por defecto
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Solo agregar documentos que no existan en la base de datos.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # Crear IDs como "data/monopoly.pdf:6:2"
    # Fuente : Número de página : Índice del chunk
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Si el ID de la página es el mismo que el anterior, incrementar el índice.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calcular el ID del chunk.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Agregarlo a la metadata del chunk.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()