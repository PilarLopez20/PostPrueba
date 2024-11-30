import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import os

app = Flask(__name__)

# Define las clases generales (posturas)
CLASSES_GENERALES = ["frontal", "posterior", "latDerecho", "latIzquierdo"]

# Ruta del modelo TensorFlow Lite
TFLITE_MODEL_PATH = "model/modelo_posturas_general.tflite"

# Cargar el intérprete de TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def procesar_imagen(imagen_bytes):
    """
    Procesa una imagen y la convierte en un vector plano con las dimensiones esperadas por el modelo.
    """
    try:
        # Cargar la imagen
        imagen = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')

        # Redimensionar la imagen (ajusta al tamaño esperado por tu pipeline)
        imagen = imagen.resize((256, 256))

        # Convertir la imagen a un numpy array y normalizar los valores
        imagen_array = np.array(imagen, dtype=np.float32) / 255.0

        # Aplanar la imagen
        imagen_flatten = imagen_array.flatten()

        # Ajustar las dimensiones a [1, 120] (si es lo que tu modelo espera)
        if len(imagen_flatten) < 120:
            imagen_flatten = np.pad(imagen_flatten, (0, 120 - len(imagen_flatten)), mode='constant')
        elif len(imagen_flatten) > 120:
            imagen_flatten = imagen_flatten[:120]

        # Regresar el vector con la forma esperada
        return np.expand_dims(imagen_flatten, axis=0)  # [1, 120]
    except Exception as e:
        raise ValueError(f"Error procesando la imagen: {str(e)}")


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Servidor Flask funcionando correctamente',
        'classes': CLASSES_GENERALES
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen.'}), 400

        imagen = request.files['file'].read()
        input_data = procesar_imagen(imagen)

        # Configurar los datos de entrada del intérprete
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Realizar la inferencia
        interpreter.invoke()

        # Obtener los resultados de la predicción
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])

        # Traduce el índice a la clase general
        predicted_posture = CLASSES_GENERALES[predicted_class]

        return jsonify({'prediction': predicted_posture})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
