# src/utils/normalize_filename.py
import os
import re

def normalize_filename(filename: str) -> str:
    """
    Normaliza el nombre de un archivo:
      - Convierte a minúsculas.
      - Reemplaza guiones bajos (_) y guiones (-) por espacios.
      - Elimina espacios extra.
      - Conserva la extensión original.
    """
    base, ext = os.path.splitext(filename)
    # Reemplazar "_" y "-" por espacios
    normalized_base = base.replace("_", " ").replace("-", " ")
    # Convertir a minúsculas y eliminar espacios redundantes
    normalized_base = " ".join(normalized_base.lower().split())
    # Reconstruir el nombre con la extensión original en minúsculas
    return f"{normalized_base}{ext.lower()}"

def rename_files_in_directory(directory_path: str):
    """
    Recorre el directorio dado y renombra todos los archivos (PDF u otros)
    aplicando la normalización al nombre.
    """
    for filename in os.listdir(directory_path):
        current_path = os.path.join(directory_path, filename)
        if os.path.isfile(current_path):
            new_filename = normalize_filename(filename)
            new_path = os.path.join(directory_path, new_filename)
            if current_path != new_path:
                print(f"Renombrando: {filename}  ->  {new_filename}")
                os.rename(current_path, new_path)

def main():
    # Determinar la ruta raíz de los datos relativa a este script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_data_path = os.path.join(script_dir, "..", "..", "data")

    # Directorios a procesar (según tu DOCUMENT_TYPES)
    directories = [
        "01_constitucion",
        "02_convenios_internacionales",
        "03_leyes",
        "04_codigos"
    ]

    for subdir in directories:
        dir_path = os.path.join(root_data_path, subdir)
        if os.path.exists(dir_path):
            print(f"\nProcesando directorio: {dir_path}")
            rename_files_in_directory(dir_path)
        else:
            print(f"El directorio {dir_path} no existe.")

if __name__ == "__main__":
    main()
