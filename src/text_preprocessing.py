# text_preprocessing.py
import spacy
from unidecode import unidecode

# Cargar el modelo de idioma español de spaCy
nlp = spacy.load("es_core_news_sm")

def preprocess_text(text: str) -> str:
    """
    Preprocesa el texto: lematiza, elimina acentos, convierte a minúsculas y normaliza.
    """
    # Eliminar acentos y caracteres especiales
    text = unidecode(text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Lematización y eliminación de stopwords
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Unir las lemas en un solo texto
    return " ".join(lemmas)