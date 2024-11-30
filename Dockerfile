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
COPY model /main/model



# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto (Render lo asignará dinámicamente)
EXPOSE 5000

CMD ["gunicorn", "--workers=1", "--timeout=120", "-b", "0.0.0.0:5000", "main:app"]

