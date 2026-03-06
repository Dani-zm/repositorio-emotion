"""
Script para la preparación de los datos de entrenamiento.
Lee las imágenes guardadas en las carpetas de las distintas emociones,
extrae los puntos de referencia faciales (landmarks) usando OpenCV, 
los normaliza y los guarda en un archivo de texto (data.txt) que 
luego se usará para entrenar el modelo.
"""
import os

import cv2
import numpy as np

from utils import get_face_landmarks


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

if not os.path.isdir(data_dir):
    raise FileNotFoundError(
        f"No se encontró el directorio de datos '{data_dir}'. "
        f"Crea la carpeta y subcarpetas por emoción con imágenes antes de ejecutar este script."
    )

# Solo procesar estas emociones (nombres de carpetas, case-insensitive)
ALLOWED_EMOTIONS = {"happy", "sad"}

output = []
# Procesar solo carpetas de emociones permitidas, en orden fijo: happy=0, sad=1
emotion_folders = [
    e for e in sorted(os.listdir(data_dir))
    if os.path.isdir(os.path.join(data_dir, e)) and e.lower() in ALLOWED_EMOTIONS
]
# Ordenar para índices consistentes: happy=0, sad=1
emotion_folders = sorted(emotion_folders, key=str.lower)

if not emotion_folders:
    raise RuntimeError(
        f"No se encontraron carpetas 'happy' o 'sad' en '{data_dir}'. "
        f"Solo se procesan estas dos emociones."
    )

for emotion_indx, emotion in enumerate(emotion_folders):
    emotion_path = os.path.join(data_dir, emotion)

    for image_path_ in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_path_)

        image = cv2.imread(image_path)
        if image is None:
            continue

        face_landmarks = get_face_landmarks(image)

        # Con la implementación actual, el número de características viene
        # determinado por el modelo de landmarks (por ejemplo, 68 puntos * 2).
        if len(face_landmarks) > 0:
            sample = face_landmarks + [int(emotion_indx)]
            output.append(sample)

if not output:
    raise RuntimeError(
        "No se generaron muestras. "
        "Revisa que las imágenes contengan caras detectables por OpenCV."
    )

np.savetxt("data.txt", np.asarray(output))
