# Usa una imagen base de Python
FROM python:3.10

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8501 para Streamlit
EXPOSE 8501

# Especifica el comando de inicio
CMD ["streamlit", "run", "app.py"]
