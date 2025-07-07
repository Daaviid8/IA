# **K-Nearest Neighbors (KNN)**

**K-Nearest Neighbors (KNN)** es un algoritmo de **clasificación supervisada** que predice la clase de un punto en base a las clases de sus **K vecinos más cercanos** en el espacio de características. Es uno de los algoritmos más sencillos y versátiles, utilizado tanto para **clasificación** como para **regresión**.

---

## 🔑 **¿Cómo Funciona KNN?**

1. **Definir K**: 
   - Se selecciona el parámetro **K**, que representa la cantidad de vecinos más cercanos que se usarán para clasificar un nuevo punto. Ejemplo: \( K = 9 \).

2. **Cálculo de la Distancia**: 
   - Para predecir la clase de un nuevo punto, calculamos la distancia entre este punto y los puntos de entrenamiento. Las distancias más comunes son:
   
   - **Distancia Euclidiana** (usada generalmente para datos continuos):
     \[
     d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
     \]
   - **Distancia Manhattan** (cuando las características son discretas):
     \[
     d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
     \]
   
   - Donde \( x \) y \( y \) son dos puntos en el espacio de características, y \(n\) es el número de características.

3. **Seleccionar los K Vecinos Más Cercanos**: 
   - Ordenamos los puntos de entrenamiento según su distancia al punto nuevo y seleccionamos los **K puntos más cercanos**.

4. **Votación Mayoritaria**: 
   - El nuevo punto se clasifica según la clase **mayoría** entre los \(K\) vecinos más cercanos. En caso de empate, se pueden usar métodos adicionales (como la distancia ponderada).

---

## 📊 **Fórmula para Predicción de Clase:**

La predicción de la clase \( \hat{y} \) para un nuevo punto \( x \) es:

\[
\hat{y} = \text{modo}(y_1, y_2, \dots, y_K)
\]

Donde:
- \( y_1, y_2, \dots, y_K \) son las clases de los \(K\) vecinos más cercanos.
- **Modo** significa la clase que más veces aparece entre ellos.

---

## 🏆 **Ventajas de KNN**:
- **Simplicidad**: Muy fácil de entender e implementar.
- **No paramétrico**: No hace suposiciones sobre la distribución de los datos, lo que lo hace útil en muchos escenarios.
- **Flexibilidad**: Puede ser utilizado tanto para **clasificación** como para **regresión**.
- **Modelo intuitivo**: Fácil de explicar y de visualizar.

---

## ⚠️ **Desventajas de KNN**:
- **Computacionalmente costoso**: El cálculo de distancias para cada punto de datos es lento, especialmente para grandes volúmenes de datos.
- **Sensibilidad a la escala**: Si las características tienen diferentes escalas, la distancia puede verse distorsionada. Es necesario **escalar** los datos.
- **Eficiencia de memoria**: El algoritmo requiere almacenar todo el conjunto de entrenamiento, lo que puede ser ineficiente en términos de memoria.
- **Sensibilidad al ruido**: KNN puede verse afectado por características irrelevantes o ruido en los datos.

---

## 💡 **Consideraciones Importantes**:
- **Selección de K**: Elegir el valor correcto de \(K\) es crucial. Un \(K\) pequeño puede ser sensible al ruido, mientras que un \(K\) grande puede suavizar demasiado las fronteras de decisión.
- **Escalado de los datos**: Para que las distancias sean significativas, es importante **escalar o normalizar** las características antes de aplicar el algoritmo (por ejemplo, utilizando `StandardScaler` o `MinMaxScaler`).
- **Votación ponderada**: A veces, es útil ponderar las clases de los vecinos por su **distancia inversa**, dando más peso a los puntos más cercanos.

---

## 📏 **Resumen Matemático de KNN**:

1. Para cada punto \( x \) a clasificar, calculamos la distancia \( d(x, x_i) \) entre \( x \) y todos los puntos \( x_i \) del conjunto de entrenamiento:
   \[
   d(x, x_i) = \sqrt{\sum_{i=1}^{n} (x_i - x)^2}
   \]

2. Seleccionamos los \( K \) vecinos más cercanos y realizamos la **votación mayoritaria** para determinar la clase de \( x \).

---

## 📈 **¿Para Qué Se Usa KNN?**

- **Clasificación de texto**: Como la clasificación de correos electrónicos como "spam" o "no spam".
- **Reconocimiento de imágenes**: Clasificación de imágenes basadas en similitudes de características.
- **Sistemas de recomendación**: Encontrar productos similares a los preferidos por un usuario.
- **Análisis de anomalías**: Identificar outliers o puntos de datos atípicos.

---

## 🧑‍💻 **Ejemplo Visual con KNN**

![image](https://github.com/user-attachments/assets/e714fa67-d310-48ae-ae99-89b85ee7a449)


---

## 📉 **Conclusión**:

**K-Nearest Neighbors (KNN)** es uno de los algoritmos más sencillos pero poderosos para clasificación y regresión. Su eficiencia depende del tamaño del conjunto de datos y de la elección de **K**. Aunque es simple y muy intuitivo, requiere una buena selección de parámetros y un preprocesamiento adecuado (escalado de datos) para obtener buenos resultados.
