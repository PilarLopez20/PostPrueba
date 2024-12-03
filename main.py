import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from pose_analysis import POSE_LANDMARKS, analyze_pose, analyze_lateral, analyze_frontal, analyze_posterior

# Inicializar la aplicación Flask
app = Flask(__name__)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose

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
        np_image = np.array(bytearray(image_file), dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "No se pudo procesar la imagen."}), 400

        # Convertir a RGB para MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe Pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results = pose.process(rgb_image)

            if not results.pose_landmarks:
                return jsonify({"error": "No se detectaron puntos clave en la imagen."}), 400

            # Obtener dimensiones de la imagen
            image_height, image_width = rgb_image.shape[:2]

            # Análisis de la pose
            pose_type, validations = analyze_pose(
                results.pose_landmarks, image_height, image_width
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
