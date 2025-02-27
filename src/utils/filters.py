import re

def detect_document_type(question):
    """Detecta el tipo de documento basado en patrones en la pregunta"""
    # Convertir a minúsculas para facilitar la detección
    question_lower = question.lower()
    
    # Patrones para detectar referencias a tipos de documentos
    constitution_pattern = r"(constituci[oó]n|carta magna)"
    code_pattern = r"(c[oó]digo\s+\w+)"
    law_pattern = r"(ley\s+\w+|ley\s+org[aá]nica\s+\w+|ley\s+de\s+\w+)"
    international_pattern = r"(convenio|tratado|acuerdo)\s+internacional"
    
    # Filtrar por tipo de documento basado en patrones en la pregunta
    doc_type_filter = None
    specific_source = None
    
    if re.search(constitution_pattern, question_lower):
        doc_type_filter = "constitucion"
    elif re.search(code_pattern, question_lower):
        doc_type_filter = "codigo"
        # Intentar extraer el nombre específico del código
        match = re.search(code_pattern, question_lower)
        if match:
            specific_source = match.group(0)
    elif re.search(law_pattern, question_lower):
        doc_type_filter = "ley"
        # Intentar extraer el nombre específico de la ley
        match = re.search(law_pattern, question_lower)
        if match:
            specific_source = match.group(0)
    elif re.search(international_pattern, question_lower):
        doc_type_filter = "convenio_internacional"
    
    return doc_type_filter, specific_source

def create_filter_dict(selected_sources, question):
    """Crea un diccionario de filtro para el retriever de Pinecone"""
    doc_type_filter, specific_source = detect_document_type(question)
    filter_dict = {}
    
    # Si se seleccionó "todos" y no se detectó un tipo específico, no aplicar filtro
    if "todos" in selected_sources and not doc_type_filter:
        # No aplicar ningún filtro
        pass
    # Si se detectó un tipo específico en la pregunta, usar ese filtro
    elif doc_type_filter:
        filter_dict["doc_type"] = doc_type_filter
    # De lo contrario, usar los tipos seleccionados en la UI
    elif "todos" not in selected_sources:
        filter_dict["doc_type"] = {"$in": selected_sources}
    
    # Si se detectó una referencia a un documento específico, añadir al filtro
    if specific_source:
        # Crear un filtro más flexible para el nombre del archivo
        filter_dict["filename"] = {"$contains": specific_source}
    
    print(f"Filtro aplicado: {filter_dict}")
    return filter_dict