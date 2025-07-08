import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Cargar datos
iris = load_iris()
X = iris.data[:, [3]]  # Largo del pétalo
y = iris.data[:, 0]    # Largo del sépalo

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo polinómico (grado 2)
grado = 2
modelo = make_pipeline(PolynomialFeatures(grado), LinearRegression())
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

# Evaluación
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Gráfico
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
x_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_range_pred = modelo.predict(x_range)
plt.plot(x_range, y_range_pred, color='green', label='Regresión polinómica (grado 2)')
plt.xlabel('Largo del pétalo (cm)')
plt.ylabel('Largo del sépalo (cm)')
plt.title('Regresión Polinómica - Dataset Iris')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
