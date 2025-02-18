import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Cargar variables de entorno desde un archivo .env
load_dotenv()

def get_embedding_function():
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-large" 
    )
    return embeddings

