import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos sintéticos
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear los modelos
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=1.0)

# Entrenar los modelos
linear_reg.fit(X_train_scaled, y_train)
ridge_reg.fit(X_train_scaled, y_train)
lasso_reg.fit(X_train_scaled, y_train)

# Hacer predicciones
y_pred_linear = linear_reg.predict(X_test_scaled)
y_pred_ridge = ridge_reg.predict(X_test_scaled)
y_pred_lasso = lasso_reg.predict(X_test_scaled)

# Calcular métricas
print("Métricas de evaluación:")
print(f"Regresión Lineal - MSE: {mean_squared_error(y_test, y_pred_linear):.2f}, R²: {r2_score(y_test, y_pred_linear):.3f}")
print(f"Ridge - MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}, R²: {r2_score(y_test, y_pred_ridge):.3f}")
print(f"Lasso - MSE: {mean_squared_error(y_test, y_pred_lasso):.2f}, R²: {r2_score(y_test, y_pred_lasso):.3f}")

# Crear gráfica
plt.figure(figsize=(12, 8))

# Subplot 1: Comparación de predicciones
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.7, label='Lineal', color='blue')
plt.scatter(y_test, y_pred_ridge, alpha=0.7, label='Ridge', color='red')
plt.scatter(y_test, y_pred_lasso, alpha=0.7, label='Lasso', color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación de Predicciones')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Datos originales con líneas de regresión
plt.subplot(2, 2, 2)
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)

y_plot_linear = linear_reg.predict(X_plot_scaled)
y_plot_ridge = ridge_reg.predict(X_plot_scaled)
y_plot_lasso = lasso_reg.predict(X_plot_scaled)

plt.scatter(X, y, alpha=0.5, color='gray', label='Datos originales')
plt.plot(X_plot, y_plot_linear, color='blue', label='Lineal', linewidth=2)
plt.plot(X_plot, y_plot_ridge, color='red', label='Ridge', linewidth=2)
plt.plot(X_plot, y_plot_lasso, color='green', label='Lasso', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Líneas de Regresión')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Comparación de coeficientes
plt.subplot(2, 2, 3)
models = ['Lineal', 'Ridge', 'Lasso']
coefficients = [linear_reg.coef_[0], ridge_reg.coef_[0], lasso_reg.coef_[0]]
colors = ['blue', 'red', 'green']

bars = plt.bar(models, coefficients, color=colors, alpha=0.7)
plt.ylabel('Coeficientes')
plt.title('Comparación de Coeficientes')
plt.grid(True, alpha=0.3)

# Agregar valores en las barras
for bar, coef in zip(bars, coefficients):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{coef:.2f}', ha='center', va='bottom' if height > 0 else 'top')

# Subplot 4: Comparación de MSE
plt.subplot(2, 2, 4)
mse_values = [mean_squared_error(y_test, y_pred_linear),
              mean_squared_error(y_test, y_pred_ridge),
              mean_squared_error(y_test, y_pred_lasso)]

bars = plt.bar(models, mse_values, color=colors, alpha=0.7)
plt.ylabel('MSE')
plt.title('Error Cuadrático Medio')
plt.grid(True, alpha=0.3)

# Agregar valores en las barras
for bar, mse in zip(bars, mse_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{mse:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Mostrar información adicional sobre regularización
print("\nInformación sobre regularización:")
print(f"Coeficiente Regresión Lineal: {linear_reg.coef_[0]:.4f}")
print(f"Coeficiente Ridge: {ridge_reg.coef_[0]:.4f}")
print(f"Coeficiente Lasso: {lasso_reg.coef_[0]:.4f}")
print(f"\nReducción del coeficiente:")
print(f"Ridge vs Lineal: {abs(ridge_reg.coef_[0] - linear_reg.coef_[0]):.4f}")
print(f"Lasso vs Lineal: {abs(lasso_reg.coef_[0] - linear_reg.coef_[0]):.4f}")
