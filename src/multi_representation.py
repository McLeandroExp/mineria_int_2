import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_summary(text: str) -> str:
    """
    Genera un resumen del texto dado para optimizar la búsqueda semantica 

    """
    # Definir un prompt simple para resumir
    prompt_template = ChatPromptTemplate.from_template(
        """
        Eres un experto en legislación ecuatoriana y en procesamiento de documentos legales para optimización de búsqueda semántica.
        Analiza el siguiente fragmento de un documento legal y genera una representación optimizada para indexación.
        Instrucciones:
        1. Extrae y enfatiza los términos jurídicos y técnicos más relevantes.
        2. Identifica artículos, capítulos o normativas clave que puedan mejorar la búsqueda.
        3. No resumas el contenido, sino reescríbelo destacando los conceptos más importantes.
        4. Mantén la coherencia del texto para que conserve su significado legal preciso.
        5. Si hay referencias a leyes específicas, inclúyelas explícitamente en la reformulación.
        6. Si hay prefijos de palabras legales o tecnicas reescribelas para que sean mas explícitas.
        7. Siempre mantén las referencias numéricas como el numero de ley o artículo.
        Documento original:
        \n\n{text}"""
    )
    # Instanciar el LLM (se puede ajustar el modelo y temperatura según necesidad)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
    # Construir el prompt reemplazando la variable
    prompt = prompt_template.format(text=text)
    # Llamar al LLM para obtener el resumen
    summary = llm(prompt)
    return summary.content
