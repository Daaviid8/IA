import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset California Housing
data = fetch_california_housing()
X, y = data.data, data.target

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características (muy importante para Elastic Net)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo Elastic Net
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42)

# Entrenar
elastic_net.fit(X_train_scaled, y_train)

# Predecir en test
y_pred = elastic_net.predict(X_test_scaled)

# Métricas de desempeño
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Elastic Net MSE: {mse:.3f}")
print(f"Elastic Net R²: {r2:.3f}")

# Mostrar coeficientes
print("Coeficientes del modelo:")
for feature, coef in zip(data.feature_names, elastic_net.coef_):
    print(f"{feature}: {coef:.4f}")

# Gráfica: Valores reales vs Predicciones
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='teal', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valor Real")
plt.ylabel("Predicción Elastic Net")
plt.title("Elastic Net: Predicción vs Valor Real (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

