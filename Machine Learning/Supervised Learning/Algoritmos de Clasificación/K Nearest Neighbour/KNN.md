# Resumen de **K-Nearest Neighbors (KNN)**

**K-Nearest Neighbors (KNN)** es un algoritmo de **clasificación supervisada** que predice la clase de un punto en base a las clases de sus **K vecinos más cercanos** en el espacio de características.

## ¿Cómo Funciona?

1. **Definición de K**: Se selecciona un número \(K\), que es la cantidad de vecinos más cercanos que se usarán para hacer la predicción.
   
2. **Cálculo de la Distancia**: Para cada nuevo punto que queremos clasificar, se calcula su **distancia** a todos los puntos de entrenamiento. Las distancias comunes son:
   - **Distancia Euclidiana**:
     \[
     d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
     \]
   - **Distancia Manhattan**:
     \[
     d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
     \]
   Donde \( x \) e \( y \) son puntos en el espacio de características y \(n\) es el número de características.

3. **Selección de los K Vecinos**: Una vez calculadas las distancias, se seleccionan los **K puntos más cercanos** al nuevo punto.

4. **Clasificación**: El punto se clasifica según la **mayoría de los votos** de sus \(K\) vecinos. Si los \(K\) vecinos pertenecen a diferentes clases, se elige la clase más frecuente.

## Fórmula para Predicción de Clase:

La predicción de la clase \( \hat{y} \) para un nuevo punto \( x \) es:

\[
\hat{y} = \text{modo}(y_1, y_2, \dots, y_K)
\]

Donde \( y_1, y_2, \dots, y_K \) son las clases de los \(K\) vecinos más cercanos, y **modo** significa la clase que más veces aparece entre ellos.

## Ventajas de KNN:
- **Simplicidad**: Es fácil de entender e implementar.
- **No paramétrico**: No asume nada sobre la distribución de los datos.
- **Adaptable**: Puede ser usado tanto para **clasificación** como para **regresión**.

## Desventajas de KNN:
- **Computacionalmente costoso**: Requiere calcular las distancias de cada punto en el conjunto de datos, lo que puede ser lento para grandes volúmenes de datos.
- **Sensible a la escala de los datos**: Las características deben estar escaladas, ya que KNN depende de la distancia.
- **Eficiencia de memoria**: El algoritmo debe almacenar todo el conjunto de entrenamiento.

## Consideraciones Importantes:
- **Elección de K**: Un valor pequeño de \(K\) puede ser sensible al ruido, mientras que un \(K\) grande puede perder detalles importantes. Generalmente, \(K\) se elige a través de validación cruzada.
- **Escalado de Datos**: Dado que KNN usa distancias, es importante normalizar o estandarizar los datos si las características tienen escalas diferentes.

## Resumen Matemático:

- Para cada punto \(x\) a clasificar, calcular la distancia a cada punto de entrenamiento:
  \[
  d(x, x_i) = \sqrt{\sum_{i=1}^{n} (x_i - x)^2}
  \]
- Seleccionar los \(K\) puntos más cercanos, y hacer una votación mayoritaria sobre sus clases.

Este enfoque es simple, pero puede ser muy efectivo, especialmente cuando los datos están bien distribuidos. Sin embargo, puede ser lento y costoso en términos de memoria para grandes bases de datos.
