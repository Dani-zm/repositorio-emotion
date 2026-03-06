"""
Script para entrenar el modelo de Machine Learning.
Carga los datos preprocesados de data.txt (características faciales y sus etiquetas), 
entrena un modelo de Random Forest Classifier y luego lo guarda en el disco 
(archivo 'model') para que pueda ser utilizado por la aplicación principal.
"""
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


data_file = "data.txt"

if not os.path.isfile(data_file):
    raise FileNotFoundError(
        f"No se encontró '{data_file}'. Ejecuta primero 'prepare_data.py' para generarlo."
    )

# Cargar datos desde el archivo de texto
data = np.loadtxt(data_file)

if data.ndim == 1:
    # Solo una muestra -> remodelar a (1, n_features)
    data = data.reshape(1, -1)

if data.shape[1] < 2:
    raise ValueError(
        f"El archivo '{data_file}' no tiene suficiente número de columnas. "
        f"Esperaba características + etiqueta."
    )

# Separar en características (X) y etiquetas (y)
X = data[:, :-1]
y = data[:, -1].astype(int)

if len(np.unique(y)) < 2:
    raise ValueError(
        "Se necesita al menos dos clases diferentes en los datos para entrenar el modelo."
    )

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y,
)

# Clasificador Random Forest (parámetros explícitos y reproducibles)
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

# Entrenar el modelo
rf_classifier.fit(X_train, y_train)

# Evaluar en el conjunto de prueba
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

# Guardar modelo entrenado
with open("./model", "wb") as f:
    pickle.dump(rf_classifier, f)
