import math

# Índices de puntos clave en MediaPipe
POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_eye": 1,  # Ojo izquierdo
    "right_eye": 2,  # Ojo derecho
    "mouth": 9,       # Boca
    "left_knee": 25,  # Rodilla izquierda
    "right_knee": 26, # Rodilla derecha
}

def calculate_angle(p1, p2, p3):
    """Calcula el ángulo entre tres puntos clave."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    )
    return abs(angle)

def calculate_angle_horizontal(p1, p2):
    """
    Calcula la desviación del ángulo respecto a 180° (horizontal).
    Retorna la diferencia relativa, positiva si hacia arriba, negativa si hacia abajo.
    """
    x1, y1 = p1
    x2, y2 = p2
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  # Ángulo respecto al eje horizontal
    reference_angle = 180.0  # El ángulo base
    deviation = angle - reference_angle if angle >= 0 else angle + reference_angle
    return deviation

def classify_lumbar_angle(left_hip, right_hip, reference_point):
    """Clasifica la región lumbar según el ángulo."""
    angle = calculate_angle(left_hip, right_hip, reference_point)
    if angle < 10:
        return "Curvatura lumbar normal"
    elif angle < 30:
        return "Zona lumbar ligeramente hundida"
    else:
        return "Zona lumbar muy hundida"


def classify_dorsal_angle(left_shoulder, right_shoulder, reference_point):
    """Clasifica la región dorsal según el ángulo."""
    angle = calculate_angle(left_shoulder, right_shoulder, reference_point)
    if angle < 10:
        return "Curvatura dorsal normal"
    elif angle < 30:
        return "Parte superior de la espalda algo más curvada"
    else:
        return "Parte superior de la espalda muy curvada"


def analyze_lateral(landmarks, image_width, image_height):
    """Analiza poses laterales: dorsal y lumbar."""
    # Coordenadas de puntos clave
    left_shoulder = [
        landmarks.landmark[POSE_LANDMARKS["left_shoulder"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["left_shoulder"]].y * image_height
    ]
    right_shoulder = [
        landmarks.landmark[POSE_LANDMARKS["right_shoulder"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["right_shoulder"]].y * image_height
    ]
    left_hip = [
        landmarks.landmark[POSE_LANDMARKS["left_hip"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["left_hip"]].y * image_height
    ]
    right_hip = [
        landmarks.landmark[POSE_LANDMARKS["right_hip"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["right_hip"]].y * image_height
    ]

    # Clasificación lumbar y dorsal
    lumbar_label = classify_lumbar_angle(left_hip, right_hip, left_shoulder)
    dorsal_label = classify_dorsal_angle(left_shoulder, right_shoulder, right_hip)

    # Retornar resultados
    return {"lumbar": lumbar_label, "dorsal": dorsal_label}


def classify_pose(landmarks):
    """Clasifica la pose como frontal, posterior o lateral."""
    left_shoulder = landmarks.landmark[POSE_LANDMARKS["left_shoulder"]]
    right_shoulder = landmarks.landmark[POSE_LANDMARKS["right_shoulder"]]
    
    # Verificar si es lateral basándose en la profundidad (z)
    if abs(left_shoulder.z - right_shoulder.z) >= 0.1:
        return "Lateral Izquierdo" if left_shoulder.z > right_shoulder.z else "Lateral Derecho"

    # Verificar si es frontal o posterior basándose en ojos y boca
    left_eye = landmarks.landmark[POSE_LANDMARKS["left_eye"]]
    right_eye = landmarks.landmark[POSE_LANDMARKS["right_eye"]]
    mouth = landmarks.landmark[POSE_LANDMARKS["mouth"]]

    # Si la visibilidad de ojos y boca es baja, es posterior
    if (
        left_eye.visibility < 1 or
        right_eye.visibility < 1 or
        mouth.visibility < 1
    ):
        return "Posterior"

    # Si los puntos faciales son visibles, es frontal
    return "Frontal"


def analyze_column(landmarks, image_width):
    """Analiza la alineación de la columna usando hombros y caderas."""
    # Obtener coordenadas
    left_shoulder = landmarks.landmark[POSE_LANDMARKS["left_shoulder"]]
    right_shoulder = landmarks.landmark[POSE_LANDMARKS["right_shoulder"]]
    left_hip = landmarks.landmark[POSE_LANDMARKS["left_hip"]]
    right_hip = landmarks.landmark[POSE_LANDMARKS["right_hip"]]

    # Calcular los puntos medios de caderas y hombros
    midpoint_shoulders = (left_shoulder.x + right_shoulder.x) / 2 * image_width
    midpoint_hips = (left_hip.x + right_hip.x) / 2 * image_width

    # Calcular desviación entre caderas y hombros
    deviation = abs(midpoint_shoulders - midpoint_hips)

    # Definir un margen de tolerancia para la alineación
    tolerance = 10  # Ajustar según tus necesidades

    # Clasificar como correcto o incorrecto
    if deviation <= tolerance:
        alignment_label = f"Columna alineada (Desviación: {deviation:.1f}px)"
    else:
        alignment_label = f"Columna desviada (Desviación: {deviation:.1f}px)"

    return alignment_label



def analyze_frontal(landmarks, image_height):
    """Analiza poses frontales: hombros y rodillas, calculando la desviación y el lado."""
    left_shoulder = [landmarks.landmark[POSE_LANDMARKS["left_shoulder"]].x,
                     landmarks.landmark[POSE_LANDMARKS["left_shoulder"]].y * image_height]
    right_shoulder = [landmarks.landmark[POSE_LANDMARKS["right_shoulder"]].x,
                      landmarks.landmark[POSE_LANDMARKS["right_shoulder"]].y * image_height]
    left_knee = [landmarks.landmark[POSE_LANDMARKS["left_knee"]].x,
                 landmarks.landmark[POSE_LANDMARKS["left_knee"]].y * image_height]
    right_knee = [landmarks.landmark[POSE_LANDMARKS["right_knee"]].x,
                  landmarks.landmark[POSE_LANDMARKS["right_knee"]].y * image_height]

    shoulder_deviation = calculate_angle_horizontal(left_shoulder, right_shoulder)
    if shoulder_deviation > 0:
        shoulder_label = f"Hombro derecho más alto: {abs(shoulder_deviation):.1f}°"
    elif shoulder_deviation < 0:
        shoulder_label = f"Hombro izquierdo más alto: {abs(shoulder_deviation):.1f}°"
    else:
        shoulder_label = "Hombros nivelados: 0.0°"

    knee_deviation = calculate_angle_horizontal(left_knee, right_knee)
    if knee_deviation > 0:
        knee_label = f"Rodilla derecha más alta: {abs(knee_deviation):.1f}°"
    elif knee_deviation < 0:
        knee_label = f"Rodilla izquierda más alta: {abs(knee_deviation):.1f}°"
    else:
        knee_label = "Rodillas niveladas: 0.0°"

    return {"hombros": shoulder_label, "rodillas": knee_label}

def analyze_posterior(landmarks, image_height, image_width):
    """Analiza poses posteriores: caderas, tobillos y columna."""
    left_hip = [
        landmarks.landmark[POSE_LANDMARKS["left_hip"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["left_hip"]].y * image_height
    ]
    right_hip = [
        landmarks.landmark[POSE_LANDMARKS["right_hip"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["right_hip"]].y * image_height
    ]
    left_ankle = [
        landmarks.landmark[POSE_LANDMARKS["left_ankle"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["left_ankle"]].y * image_height
    ]
    right_ankle = [
        landmarks.landmark[POSE_LANDMARKS["right_ankle"]].x * image_width,
        landmarks.landmark[POSE_LANDMARKS["right_ankle"]].y * image_height
    ]

    # Continuar con los cálculos necesarios...

    # Calcular desviación de caderas
    hip_deviation = calculate_angle_horizontal(left_hip, right_hip)
    if hip_deviation > 0:
        hip_label = f"Cadera derecha más alta: {abs(hip_deviation):.1f}°"
    elif hip_deviation < 0:
        hip_label = f"Cadera izquierda más alta: {abs(hip_deviation):.1f}°"
    else:
        hip_label = "Caderas niveladas: 0.0°"

    # Calcular desviación de tobillos
    ankle_deviation = calculate_angle_horizontal(left_ankle, right_ankle)
    if ankle_deviation > 0:
        ankle_label = f"Tobillo derecho más alto: {abs(ankle_deviation):.1f}°"
    elif ankle_deviation < 0:
        ankle_label = f"Tobillo izquierdo más alto: {abs(ankle_deviation):.1f}°"
    else:
        ankle_label = "Tobillos nivelados: 0.0°"

    # Calcular alineación de la columna usando los hombros
    left_shoulder = [landmarks.landmark[POSE_LANDMARKS["left_shoulder"]].x,
                     landmarks.landmark[POSE_LANDMARKS["left_shoulder"]].y * image_height]
    right_shoulder = [landmarks.landmark[POSE_LANDMARKS["right_shoulder"]].x,
                      landmarks.landmark[POSE_LANDMARKS["right_shoulder"]].y * image_height]

    column_deviation = calculate_angle_horizontal(left_shoulder, right_shoulder)
    if abs(column_deviation) <= 5:
        column_label = "Columna recta (Correcto)"
    else:
        column_label = f"Columna desviada: {abs(column_deviation):.1f}° (Incorrecto)"

    # Clasificar como "Correcto" o "Incorrecto" según el margen de ±5°
    hip_label += " (Correcto)" if abs(hip_deviation) <= 5 else " (Incorrecto)"
    ankle_label += " (Correcto)" if abs(ankle_deviation) <= 5 else " (Incorrecto)"

    # Imprimir para depuración
    print(f"Caderas: {hip_label}")
    print(f"Tobillos: {ankle_label}")
    print(f"Columna: {column_label}")

    return {"caderas": hip_label, "tobillos": ankle_label, "columna": column_label}


def analyze_pose(landmarks, image_height, image_width):
    """Analiza la pose según el tipo detectado."""
    pose_type = classify_pose(landmarks)

    if "Lateral" in pose_type:
        results = analyze_lateral(landmarks, image_width, image_height)  # Pasa los argumentos aquí
    elif pose_type == "Frontal":
        results = analyze_frontal(landmarks, image_height)
    elif pose_type == "Posterior":
        results = analyze_posterior(landmarks, image_height, image_width)

    return pose_type, results

