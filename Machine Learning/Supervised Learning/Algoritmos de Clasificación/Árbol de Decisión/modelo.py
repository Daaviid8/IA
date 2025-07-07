# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el conjunto de datos (puedes descargar el Titanic dataset de Kaggle)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Echar un vistazo a las primeras filas del conjunto de datos
print(data.head())

# Preprocesamiento de datos
# Rellenar los valores faltantes en las columnas de "Age" con la media
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Rellenar los valores faltantes en la columna "Embarked" con la moda
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Eliminar la columna "Name", "Ticket", "Cabin", que no aportan información útil directamente
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convertir las variables categóricas en variables numéricas
# Convertir "Sex" a 0 (Hombre) y 1 (Mujer)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Convertir "Embarked" en variables dummy (one-hot encoding)
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Mostrar los datos preprocesados
print(data.head())

# Separar las características (X) de la variable objetivo (y)
X = data.drop('Survived', axis=1)  # Características (todas las columnas menos 'Survived')
y = data['Survived']  # Variable objetivo ('Survived')

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el modelo de Árbol de Decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Graficar el árbol de decisión
plt.figure(figsize=(12,8))
from sklearn.tree import plot_tree
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'], rounded=True)
plt.title("Árbol de Decisión del Titanic")
plt.show()
