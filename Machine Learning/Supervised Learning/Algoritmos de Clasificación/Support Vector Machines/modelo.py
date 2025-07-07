# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC  # SVM Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas (especies)

# 2. Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crear el modelo SVM con kernel lineal
svm_model = SVC(kernel='linear', random_state=42)

# 4. Entrenar el modelo SVM
svm_model.fit(X_train, y_train)

# 5. Realizar predicciones
y_pred = svm_model.predict(X_test)

# 6. Evaluar el rendimiento del modelo

# Precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# 7. Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# 8. Reporte de clasificación
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Reporte de Clasificación:")
print(class_report)

# 9. Visualización del hiperplano de decisión (solo para 2 características)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette='Set1', style=y_test, s=100)
plt.title('Visualización de la Clasificación SVM')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(title="Especies")
plt.show()
