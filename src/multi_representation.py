import os
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from .prompts import MULTI_REPRESENTATION_PROMPT
def generate_summary(text: str) -> str:
    """
    Genera un resumen del texto dado para optimizar la búsqueda semantica 

    """

    # Instanciar el LLM (se puede ajustar el modelo y temperatura según necesidad)
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     temperature=0
    # )
    llm = Ollama(model="llama3.2", temperature=0)
        # Crear cadena de procesamiento
    chain = MULTI_REPRESENTATION_PROMPT | llm | StrOutputParser()

    # Generar el resumen
    summary = chain.invoke({"text": text})
    return summary
