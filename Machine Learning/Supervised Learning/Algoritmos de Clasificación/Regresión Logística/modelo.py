# ===========================
# IMPORTACIÓN DE LIBRERÍAS
# ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ===========================
# CARGA Y EXPLORACIÓN DE DATOS
# ===========================

# Cargar el dataset - ejemplo con el famoso dataset del Titanic
# Puedes descargar este dataset desde: https://www.kaggle.com/c/titanic/data
print("=== CARGA DEL DATASET ===")
try:
    # Intentar cargar el dataset
    df = pd.read_csv('titanic.csv')
    print(f"Dataset cargado exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
except FileNotFoundError:
    print("⚠️  Archivo no encontrado. Creando un dataset de ejemplo...")
    # Crear un dataset de ejemplo si no se encuentra el archivo
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'Age': np.random.normal(30, 10, n_samples),
        'Fare': np.random.exponential(30, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.3, n_samples),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.2, 0.1, 0.7]),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    # Introducir algunos valores faltantes para simular datos reales
    df.loc[np.random.choice(df.index, size=50), 'Age'] = np.nan
    df.loc[np.random.choice(df.index, size=20), 'Embarked'] = np.nan
    print(f"Dataset de ejemplo creado: {df.shape[0]} filas y {df.shape[1]} columnas")

# Mostrar información básica del dataset
print("\n=== INFORMACIÓN BÁSICA DEL DATASET ===")
print(df.info())
print("\n=== PRIMERAS 5 FILAS ===")
print(df.head())

# ===========================
# ANÁLISIS EXPLORATORIO DE DATOS
# ===========================

print("\n=== ANÁLISIS EXPLORATORIO ===")

# Verificar valores faltantes
print("Valores faltantes por columna:")
print(df.isnull().sum())

# Estadísticas descriptivas
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe())

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(12, 8))

# Subplot 1: Distribución de supervivencia
plt.subplot(2, 2, 1)
survived_counts = df['Survived'].value_counts()
plt.pie(survived_counts.values, labels=['No Sobrevivió', 'Sobrevivió'], 
        autopct='%1.1f%%', startangle=90)
plt.title('Distribución de Supervivencia')

# Subplot 2: Supervivencia por género
plt.subplot(2, 2, 2)
survival_by_sex = pd.crosstab(df['Sex'], df['Survived'])
survival_by_sex.plot(kind='bar', ax=plt.gca())
plt.title('Supervivencia por Género')
plt.xlabel('Género')
plt.ylabel('Cantidad')
plt.legend(['No Sobrevivió', 'Sobrevivió'])
plt.xticks(rotation=45)

# Subplot 3: Supervivencia por clase
plt.subplot(2, 2, 3)
survival_by_class = pd.crosstab(df['Pclass'], df['Survived'])
survival_by_class.plot(kind='bar', ax=plt.gca())
plt.title('Supervivencia por Clase')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.legend(['No Sobrevivió', 'Sobrevivió'])

# Subplot 4: Distribución de edad
plt.subplot(2, 2, 4)
plt.hist(df['Age'].dropna(), bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# ===========================
# PREPROCESAMIENTO DE DATOS
# ===========================

print("\n=== PREPROCESAMIENTO DE DATOS ===")

# Crear una copia del dataset para el procesamiento
df_processed = df.copy()

# 1. Manejar valores faltantes
print("1. Tratando valores faltantes...")

# Imputar valores faltantes en Age con la mediana
age_imputer = SimpleImputer(strategy='median')
df_processed['Age'] = age_imputer.fit_transform(df_processed[['Age']])

# Imputar valores faltantes en Embarked con la moda
embarked_imputer = SimpleImputer(strategy='most_frequent')
df_processed['Embarked'] = embarked_imputer.fit_transform(df_processed[['Embarked']]).ravel()

# 2. Crear nuevas características (Feature Engineering)
print("2. Creando nuevas características...")

# Crear característica de tamaño de familia
df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1

# Crear característica de si está solo
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

# Crear grupos de edad
df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                 bins=[0, 18, 35, 50, 100], 
                                 labels=['Menor', 'Adulto_Joven', 'Adulto', 'Mayor'])

# 3. Codificar variables categóricas
print("3. Codificando variables categóricas...")

# Codificar género (0 = female, 1 = male)
le_sex = LabelEncoder()
df_processed['Sex_encoded'] = le_sex.fit_transform(df_processed['Sex'])

# Codificar puerto de embarque usando one-hot encoding
embarked_dummies = pd.get_dummies(df_processed['Embarked'], prefix='Embarked')
df_processed = pd.concat([df_processed, embarked_dummies], axis=1)

# Codificar grupos de edad
age_group_dummies = pd.get_dummies(df_processed['AgeGroup'], prefix='AgeGroup')
df_processed = pd.concat([df_processed, age_group_dummies], axis=1)

# 4. Seleccionar características para el modelo
print("4. Seleccionando características...")

# Definir las características que usaremos en el modelo
features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
           'FamilySize', 'IsAlone', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Verificar que todas las características estén disponibles
available_features = [f for f in features if f in df_processed.columns]
print(f"Características disponibles: {available_features}")

# Preparar matriz de características (X) y variable objetivo (y)
X = df_processed[available_features]
y = df_processed['Survived']

print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# ===========================
# DIVISIÓN DE DATOS
# ===========================

print("\n=== DIVISIÓN DE DATOS ===")

# Dividir los datos en conjuntos de entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Verificar la distribución de clases
print(f"\nDistribución en entrenamiento:")
print(y_train.value_counts(normalize=True))
print(f"\nDistribución en prueba:")
print(y_test.value_counts(normalize=True))

# ===========================
# ESCALADO DE CARACTERÍSTICAS
# ===========================

print("\n=== ESCALADO DE CARACTERÍSTICAS ===")

# Crear el escalador y ajustarlo solo con datos de entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transformar, no ajustar

print("Escalado completado. Estadísticas del conjunto de entrenamiento escalado:")
print(f"Media: {np.mean(X_train_scaled, axis=0)[:5]}...")  # Mostrar solo primeras 5
print(f"Desviación estándar: {np.std(X_train_scaled, axis=0)[:5]}...")

# ===========================
# ENTRENAMIENTO DEL MODELO
# ===========================

print("\n=== ENTRENAMIENTO DEL MODELO ===")

# Crear el modelo de regresión logística
# C: parámetro de regularización (inverso de la fuerza de regularización)
# random_state: para reproducibilidad
logistic_model = LogisticRegression(
    C=1.0,                    # Fuerza de regularización
    random_state=42,          # Para reproducibilidad
    max_iter=1000,           # Máximo número de iteraciones
    solver='lbfgs'           # Algoritmo de optimización
)

# Entrenar el modelo
logistic_model.fit(X_train_scaled, y_train)

print("Modelo entrenado exitosamente!")
print(f"Número de iteraciones: {logistic_model.n_iter_[0]}")

# ===========================
# PREDICCIONES Y EVALUACIÓN
# ===========================

print("\n=== PREDICCIONES Y EVALUACIÓN ===")

# Hacer predicciones
y_pred = logistic_model.predict(X_test_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades de clase 1

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Reporte de clasificación detallado
print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, y_pred, target_names=['No Sobrevivió', 'Sobrevivió']))

# ===========================
# VISUALIZACIONES DE RESULTADOS
# ===========================

print("\n=== VISUALIZACIONES DE RESULTADOS ===")

plt.figure(figsize=(15, 10))

# Subplot 1: Matriz de confusión
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Sobrevivió', 'Sobrevivió'],
            yticklabels=['No Sobrevivió', 'Sobrevivió'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

# Subplot 2: Curva ROC
plt.subplot(2, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()

# Subplot 3: Distribución de probabilidades predichas
plt.subplot(2, 3, 3)
plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Sobrevivió', color='red')
plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Sobrevivió', color='green')
plt.xlabel('Probabilidad Predicha')
plt.ylabel('Frecuencia')
plt.title('Distribución de Probabilidades Predichas')
plt.legend()

# Subplot 4: Importancia de características (coeficientes)
plt.subplot(2, 3, 4)
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': logistic_model.coef_[0]
})
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)

plt.barh(range(len(feature_importance)), feature_importance['Coefficient'])
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Coeficiente')
plt.title('Importancia de Características\n(Coeficientes del Modelo)')
plt.grid(axis='x', alpha=0.3)

# Subplot 5: Residuos
plt.subplot(2, 3, 5)
residuals = y_test - y_pred_proba
plt.scatter(y_pred_proba, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Probabilidades Predichas')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')

# Subplot 6: Distribución de errores
plt.subplot(2, 3, 6)
errors = np.abs(y_test - y_pred_proba)
plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Error Absoluto')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')

plt.tight_layout()
plt.show()

# ===========================
# INTERPRETACIÓN DEL MODELO
# ===========================

print("\n=== INTERPRETACIÓN DEL MODELO ===")

# Mostrar los coeficientes del modelo
print("Coeficientes del modelo (log-odds):")
for feature, coef in zip(available_features, logistic_model.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"{feature:15}: {coef:8.4f} (OR: {odds_ratio:.4f})")

print(f"\nIntercepto: {logistic_model.intercept_[0]:.4f}")

# Explicar algunos coeficientes importantes
print("\n=== INTERPRETACIÓN DE COEFICIENTES ===")
print("• Coeficiente positivo: aumenta la probabilidad de supervivencia")
print("• Coeficiente negativo: disminuye la probabilidad de supervivencia")
print("• Odds Ratio > 1: factor de riesgo positivo")
print("• Odds Ratio < 1: factor de riesgo negativo")

# ===========================
# PREDICCIONES EN NUEVOS DATOS
# ===========================

print("\n=== EJEMPLO DE PREDICCIÓN EN NUEVOS DATOS ===")

# Crear algunos ejemplos de nuevos pasajeros
new_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Sex_encoded': [0, 1, 0],  # 0=female, 1=male
    'Age': [25, 35, 45],
    'SibSp': [1, 0, 1],
    'Parch': [0, 0, 2],
    'Fare': [80, 15, 50],
    'FamilySize': [2, 1, 4],
    'IsAlone': [0, 1, 0],
    'Embarked_C': [1, 0, 0],
    'Embarked_Q': [0, 0, 1],
    'Embarked_S': [0, 1, 0]
})

# Asegurarse de que todas las características estén presentes
for feature in available_features:
    if feature not in new_passengers.columns:
        new_passengers[feature] = 0

# Seleccionar las características en el orden correcto
new_passengers = new_passengers[available_features]

# Escalar los nuevos datos
new_passengers_scaled = scaler.transform(new_passengers)

# Hacer predicciones
new_predictions = logistic_model.predict(new_passengers_scaled)
new_probabilities = logistic_model.predict_proba(new_passengers_scaled)[:, 1]

print("Predicciones para nuevos pasajeros:")
for i, (pred, prob) in enumerate(zip(new_predictions, new_probabilities)):
    status = "Sobrevivió" if pred == 1 else "No Sobrevivió"
    print(f"Pasajero {i+1}: {status} (Probabilidad: {prob:.4f})")

print("\n=== ANÁLISIS COMPLETADO ===")
print("El modelo de regresión logística ha sido entrenado y evaluado exitosamente!")
print("Puedes usar este modelo para predecir la supervivencia de nuevos pasajeros.")
