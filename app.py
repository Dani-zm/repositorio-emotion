"""
Archivo principal de la aplicación web Flask.
Se encarga de levantar el servidor web, servir la interfaz (index.html),
recibir las imágenes de la cámara web, calcular las
emociones detectadas usando el modelo de Machine Learning, y comunicarse
con la API de Gemini para dar retroalimentación empática.
"""
import os
import pickle
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from utils import get_face_landmarks

app = Flask(__name__, static_folder='.', static_url_path='')

emotions = ["HAPPY", "SAD"]

model_path = "./model"
if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado en '{model_path}'. "
        f"Ejecuta antes 'train_model.py'."
    )

with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_landmarks = get_face_landmarks(frame, draw=False, static_image_mode=True)

        if len(face_landmarks) > 0:
            output = model.predict([face_landmarks])
            emotion = emotions[int(output[0])] if 0 <= int(output[0]) < len(emotions) else "UNKNOWN"
            return jsonify({'emotion': emotion})
        else:
            return jsonify({'emotion': 'No face detected'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_session', methods=['POST'])
def analyze_session():
    try:
        data = request.json
        context = data.get('context', '')
        emotions_array = data.get('emotions', [])
        
        if not context:
            return jsonify({'error': 'No context provided'}), 400
            
        import sys
        try:
            from google import genai
            # Initialize client. The user needs to have GEMINI_API_KEY environment variable set.
            # We catch the error if it isn't set.
            client = genai.Client(api_key="")
            prompt = f"El usuario acaba de tener una sesión de 30 segundos hablando a una cámara sobre el siguiente tema/contexto:\n\n'{context}'\n\nDurante este tiempo, la IA detectó segundo a segundo la siguiente secuencia de emociones en su rostro:\n{emotions_array}\n\nActúa como un asistente muy empático. Analiza brevemente qué significa esta fluctuación de emociones dado el contexto y, lo más importante, brinda un mensaje de mucho apoyo o recuérdale algo positivo y bonito al usuario para subirle el ánimo. Da una respuesta cálida y concisa (máximo 2 párrafos)."
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            return jsonify({'analysis': response.text})
            
        except ImportError:
            return jsonify({'error': 'La librería google-genai no está instalada o configurada. Por favor instala con pip install google-genai'}), 500
        except Exception as e:
            if "API key" in str(e).lower() or "credentials" in str(e).lower():
                return jsonify({'error': 'Falta configurar tu GEMINI_API_KEY. Configúrala como variable de entorno o en el archivo app.py.'}), 500
            return jsonify({'error': f'Error usando Gemini: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Iniciando servidor Flask de EmoChat en http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
