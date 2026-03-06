"""
Archivo de utilidades y funciones compartidas.
Contiene la función principal (get_face_landmarks) para procesar una imagen 
y extraer las coordenadas de la cara utilizando el modelo Haar Cascade 
y LBF de OpenCV. También incluye funciones para descargar los modelos 
automáticamente si no existen.
"""
import os
import urllib.request
from typing import List

import cv2


_HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)
_LBF_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
)

_HAAR_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
_LBF_PATH = os.path.join(os.path.dirname(__file__), "lbfmodel.yaml")


def _download_file(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)


def _ensure_models():
    if not os.path.isfile(_HAAR_PATH):
        _download_file(_HAAR_URL, _HAAR_PATH)
    if not os.path.isfile(_LBF_PATH):
        _download_file(_LBF_URL, _LBF_PATH)


_face_detector = None
_facemark = None

def _get_models():
    global _face_detector, _facemark
    _ensure_models()
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(_HAAR_PATH)
    if _facemark is None:
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "Tu instalación de OpenCV no incluye el módulo `cv2.face`. "
                "Instala `opencv-contrib-python` (no solo `opencv-python`):\n"
                "    pip install opencv-contrib-python\n"
            )
        _facemark = cv2.face.createFacemarkLBF()
        _facemark.loadModel(_LBF_PATH)
    return _face_detector, _facemark


def get_face_landmarks(image, draw: bool = False, static_image_mode: bool = True) -> List[float]:
    """
    Extrae landmarks faciales 2D usando únicamente OpenCV (sin Mediapipe).
    Utiliza un detector Haar + FacemarkLBF (68 puntos): devuelve una lista
    plana [x1_norm, y1_norm, x2_norm, y2_norm, ...].
    """

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return []

    face_detector, facemark = _get_models()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return []

    ok, landmarks = facemark.fit(gray, faces)
    if not ok or len(landmarks) == 0:
        return []

    # Solo usamos la primera cara
    points = landmarks[0][0]  # shape: (68, 2)

    xs = points[:, 0]
    ys = points[:, 1]

    min_x, min_y = xs.min(), ys.min()
    max_x, max_y = xs.max(), ys.max()
    
    width = float(max_x - min_x)
    height = float(max_y - min_y)
    if width <= 0: width = 1.0
    if height <= 0: height = 1.0

    features: List[float] = []
    for (x, y) in zip(xs, ys):
        features.append(float((x - min_x) / width))
        features.append(float((y - min_y) / height))

    if draw:
        for (x, y) in points:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

    return features