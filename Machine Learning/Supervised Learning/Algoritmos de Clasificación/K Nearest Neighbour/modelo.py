import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración para mejores gráficos
plt.style.use('default')
sns.set_palette("husl")

print("=" * 50)
print("ANÁLISIS KNN CON DATASET IRIS")
print("=" * 50)

# 1. CARGAR Y PREPARAR LOS DATOS
print("\n1. Cargando y preparando datos...")
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

# División del dataset con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# 2. ENCONTRAR EL MEJOR VALOR DE K
print("\n2. Optimizando valor de K...")
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Mejor K
best_k = k_range[np.argmax(cv_scores)]
print(f"Mejor K encontrado: {best_k}")
print(f"Precisión con validación cruzada: {max(cv_scores)*100:.2f}%")

# 3. ENTRENAR EL MODELO OPTIMIZADO
print("\n3. Entrenando modelo final...")
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión en conjunto de prueba: {accuracy*100:.2f}%")

# 4. EVALUACIÓN DETALLADA
print("\n4. Evaluación detallada del modelo...")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Mostrar errores específicos
errors = np.where(y_pred != y_test)[0]
if len(errors) > 0:
    print(f"\nErrores de clasificación ({len(errors)} de {len(y_test)}):")
    for idx in errors:
        print(f"  Muestra {idx}: Predicho={target_names[y_pred[idx]]}, "
              f"Real={target_names[y_test[idx]]}")
else:
    print("\n¡Sin errores de clasificación!")

# 5. VISUALIZACIONES
print("\n5. Generando visualizaciones...")

# Preparar PCA para visualización 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Crear figura con múltiples subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Optimización de K
axes[0, 0].plot(k_range, cv_scores, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', 
                   label=f'Mejor K = {best_k}')
axes[0, 0].set_xlabel('Valor de K')
axes[0, 0].set_ylabel('Precisión (Validación Cruzada)')
axes[0, 0].set_title('Optimización de K para KNN')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=target_names, yticklabels=target_names)
axes[0, 1].set_title('Matriz de Confusión')
axes[0, 1].set_xlabel('Predicción')
axes[0, 1].set_ylabel('Valor Real')

# Subplot 3: Visualización PCA con frontera de decisión
h = 0.02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Entrenar KNN en espacio PCA para visualización
knn_pca = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_pca.fit(X_train_pca, y_train)
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar frontera de decisión
axes[1, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
axes[1, 0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                   cmap='viridis', marker='o', alpha=0.7, s=60, label='Entrenamiento')
axes[1, 0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                   cmap='viridis', marker='^', alpha=0.9, s=80, 
                   edgecolors='black', linewidth=1, label='Prueba')
axes[1, 0].set_xlabel('Componente Principal 1')
axes[1, 0].set_ylabel('Componente Principal 2')
axes[1, 0].set_title(f'Frontera de Decisión KNN (K={best_k})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Análisis de componentes principales
explained_variance = pca.explained_variance_ratio_
axes[1, 1].bar(['PC1', 'PC2'], explained_variance, 
               color=['orange', 'lightcoral'], alpha=0.7)
axes[1, 1].set_ylabel('Varianza Explicada')
axes[1, 1].set_title(f'Componentes Principales\nVarianza Total: {explained_variance.sum():.2%}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. ANÁLISIS DE DISTANCIAS
print("\n6. Análisis de distancias...")
distances, indices = knn.kneighbors(X_test_scaled)
mean_distances = distances.mean(axis=1)

plt.figure(figsize=(12, 5))

# Histograma de distancias
plt.subplot(1, 2, 1)
plt.hist(mean_distances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Distancia media a vecinos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Distancias a Vecinos')
plt.grid(True, alpha=0.3)

# Gráfico de dispersión: distancia vs precisión por muestra
correct_predictions = (y_pred == y_test).astype(int)
plt.subplot(1, 2, 2)
colors = ['red' if pred == 0 else 'green' for pred in correct_predictions]
plt.scatter(mean_distances, correct_predictions, c=colors, alpha=0.7)
plt.xlabel('Distancia media a vecinos')
plt.ylabel('Predicción correcta (1) / Incorrecta (0)')
plt.title('Relación Distancia-Precisión')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. ANÁLISIS DE CARACTERÍSTICAS MÁS IMPORTANTES
print("\n7. Análisis de características...")
from sklearn.inspection import permutation_importance

# Calcular importancia por permutación
perm_importance = permutation_importance(knn, X_test_scaled, y_test, 
                                       n_repeats=10, random_state=42)

# Visualizar importancia
plt.figure(figsize=(10, 6))
indices = np.argsort(perm_importance.importances_mean)[::-1]

plt.bar(range(len(indices)), 
        perm_importance.importances_mean[indices],
        yerr=perm_importance.importances_std[indices],
        capsize=5, alpha=0.7, color='lightgreen')

plt.xlabel('Características')
plt.ylabel('Importancia')
plt.title('Importancia de Características (Permutación)')
plt.xticks(range(len(indices)), 
           [feature_names[i] for i in indices], 
           rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. RESUMEN FINAL
print("\n" + "=" * 50)
print("RESUMEN FINAL")
print("=" * 50)
print(f"Mejor K: {best_k}")
print(f"Precisión final: {accuracy*100:.2f}%")
print(f"Varianza explicada por PCA: {explained_variance.sum():.2%}")
print(f"Característica más importante: {feature_names[indices[0]]}")
print(f"Errores de clasificación: {len(errors)} de {len(y_test)}")

# Mostrar primeras predicciones
print(f"\nPrimeras 10 predicciones:")
for i in range(min(10, len(y_pred))):
    status = "✓" if y_pred[i] == y_test[i] else "✗"
    print(f"  {status} Predicción: {target_names[y_pred[i]]:<12} Real: {target_names[y_test[i]]}")

print(f"\n{'='*50}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*50}")
