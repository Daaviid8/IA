# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas (especies)

# 2. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crear el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Entrenar el modelo
rf_model.fit(X_train, y_train)

# 5. Realizar predicciones
y_pred = rf_model.predict(X_test)

# 6. Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# 7. Matriz de confusión
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

# 9. Importancia de las características
feature_importances = rf_model.feature_importances_
print("Importancia de las características:")
for feature, importance in zip(iris.feature_names, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Visualizar la importancia de las características
plt.figure(figsize=(8, 6))
plt.barh(iris.feature_names, feature_importances, color='skyblue')
plt.xlabel('Importancia')
plt.title('Importancia de las características en el modelo Random Forest')
plt.show()
