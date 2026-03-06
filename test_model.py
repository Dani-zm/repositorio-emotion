"""
Script para probar el modelo entrenado en tiempo real.
Activa la cámara de la computadora, extrae las características faciales 
fotograma por fotograma, y utiliza el modelo guardado para predecir 
y mostrar la emoción en la pantalla. Útil para verificar que el modelo funciona 
antes de probar la aplicación web.
"""
import os
import pickle

import cv2

from utils import get_face_landmarks


emotions = ["HAPPY", "SAD"]

model_path = "./model"
if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado en '{model_path}'. "
        f"Ejecuta antes 'train_model.py'."
    )

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Cámara por defecto de la laptop (índice 0)
cam_index = 0
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara índice {cam_index}.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    if len(face_landmarks) > 0:
        output = model.predict([face_landmarks])
        emotion_text = emotions[int(output[0])] if 0 <= int(output[0]) < len(emotions) else "UNKNOWN"
        cv2.putText(
            frame,
            emotion_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Emotion recognition", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()