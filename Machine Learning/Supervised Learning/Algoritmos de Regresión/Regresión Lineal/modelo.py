import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data[:, [3]]  # Usamos el largo del pétalo (columna 3) como predictor
y = iris.data[:, 0]    # Variable objetivo: largo del sépalo (columna 0)

# 2. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Realizar predicciones
y_pred = model.predict(X_test)

# 5. Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Coeficiente (pendiente):", model.coef_[0])
print("Intercepto:", model.intercept_)
print("MSE:", mse)
print("R²:", r2)

# 6. Graficar resultados
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresión lineal')
plt.xlabel('Largo del pétalo (cm)')
plt.ylabel('Largo del sépalo (cm)')
plt.title('Regresión Lineal - Dataset Iris')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
