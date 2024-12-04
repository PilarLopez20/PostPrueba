# Base de Ubuntu
FROM ubuntu:22.04


# Instalar dependencias del sistema necesarias para OpenCV y MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instalar pip manualmente si no está incluido
RUN apt-get update && apt-get install -y python3-pip

# Establecer el directorio de trabajo

WORKDIR /main

# Copiar los archivos de requirements primero
# Copiar todos los archivos del proyecto al contenedor
COPY . /main

# Actualizar pip e instalar dependencias críticas
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy scipy
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto (Render lo asignará dinámicamente)
EXPOSE 5000

CMD ["gunicorn", "--workers=1", "--timeout=120", "-b", "0.0.0.0:5000", "main:app"]

