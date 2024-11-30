# Base de Ubuntu
FROM ubuntu:22.04

# Instalar Python y pip
RUN apt-get update && apt-get install -y \
    python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /main

# Copiar archivos del proyecto
COPY . .

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto 5000
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
