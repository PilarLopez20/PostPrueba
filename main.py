import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from pose_analysis import (POSE_LANDMARKS, analyze_pose, analyze_lateral,analyze_frontal, analyze_posterior, detect_face
)
from PIL import Image
import io

# Inicializar la aplicación Flask
app = Flask(__name__)

# Inicializar MediaPipe Pose y Face Detection
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Servidor Flask con MediaPipe funcionando correctamente",
        "info": "Envía una imagen al endpoint /predict para obtener análisis de postura."
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen."}), 400

        # Leer la imagen enviada
        image_file = request.files["file"].read()
        
        # Convertir la imagen en un formato que OpenCV pueda procesar
        pil_image = Image.open(io.BytesIO(image_file)).convert('RGB')

        # Redimensionar la imagen a 256x256
        pil_image = pil_image.resize((256, 256))
        
        # Convertir la imagen a un array de NumPy
        np_image = np.array(pil_image)

        # Convertir de RGB a BGR para usarlo con OpenCV y MediaPipe
        image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # Procesar la imagen con MediaPipe Pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                return jsonify({"error": "No se detectaron puntos clave en la imagen."}), 400

            # Obtener dimensiones de la imagen
            image_height, image_width = 256, 256  # Ya sabemos que es 256x256

            # Análisis de la pose
            pose_type, validations = analyze_pose(
                results.pose_landmarks, image, image_height, image_width  # Se pasan los argumentos correctos
            )

            # Resultados adicionales según el tipo de pose
            if "Lateral" in pose_type:
                lateral_results = analyze_lateral(
                    results.pose_landmarks, image_width, image_height
                )
                validations.update(lateral_results)
            elif "Frontal" in pose_type:
                frontal_results = analyze_frontal(
                    results.pose_landmarks, image_height
                )
                validations.update(frontal_results)
            elif "Posterior" in pose_type:
                posterior_results = analyze_posterior(
                    results.pose_landmarks, image_height, image_width
                )
                validations.update(posterior_results)

            # Respuesta en formato JSON
            response = {
                "pose_type": pose_type,
                "validations": validations,
            }
            return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
