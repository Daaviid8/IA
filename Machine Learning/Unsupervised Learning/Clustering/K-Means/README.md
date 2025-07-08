# K-means:

K-means es uno de los algoritmos más populares para **agrupamiento (clustering)**. Su objetivo es dividir un conjunto de datos en **K grupos** o **clusters** donde los puntos dentro de cada grupo son lo más similares posible entre sí.

---

## Objetivo de K-means

Dado un conjunto de datos, el algoritmo busca particionarlos en $K$ clusters minimizando la **varianza intra-cluster**, o lo que es lo mismo, la suma de las distancias cuadradas entre cada punto y el centroide de su cluster.

---

## Cómo funciona K-means (pasos generales)

1. Elegir $K$ centroides iniciales (aleatoriamente o con algún método).
2. Asignar cada punto al cluster cuyo centroide esté más cercano (distancia Euclidiana).
3. Recalcular los centroides como el promedio de los puntos asignados a cada cluster.
4. Repetir los pasos 2 y 3 hasta que las asignaciones no cambien o se alcance un número máximo de iteraciones.

---

## Matemáticas de K-means

Sea:

- $X = \{x_1, x_2, ..., x_n\}$ el conjunto de puntos de datos, con $x_i \in \mathbb{R}^d$.
- $C = \{C_1, C_2, ..., C_K\}$ el conjunto de clusters.
- $\mu_k$ el centroide del cluster $C_k$.

El objetivo es minimizar la siguiente función de costo (inercia o suma de cuadrados intra-cluster):

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

donde:

- $\|x_i - \mu_k\|^2$ es la distancia Euclidiana al cuadrado entre el punto $x_i$ y el centroide $\mu_k$.

El algoritmo actualiza iterativamente las asignaciones de clusters y los centroides para minimizar $J$.

---

## Ventajas de K-means

- **Simple y rápido**: Fácil de implementar y eficiente para grandes conjuntos de datos.
- **Escalable**: Funciona bien con miles o millones de puntos.
- **Interpretabilidad**: Los clusters se definen claramente por sus centroides.

---

## Limitaciones de K-means

- **Se necesita definir $K$ a priori**: Hay que saber o estimar cuántos clusters queremos.
- **Sensibilidad a centroides iniciales**: La elección inicial puede afectar el resultado final (se usan métodos como k-means++ para mejorar esto).
- **Clusters de forma esférica**: K-means asume que los clusters son aproximadamente circulares o esféricos en el espacio.
- **No funciona bien con clusters de tamaños muy diferentes o con ruido/outliers**.
- **Solo usa distancia Euclidiana**: No es adecuado para datos donde otra métrica es más adecuada.

---

## Casos de uso comunes

- **Segmentación de clientes**: Agrupar clientes con comportamientos o características similares.
- **Compresión de imágenes**: Reducir colores agrupando píxeles similares.
- **Agrupamiento de documentos**: Organizar textos similares para facilitar búsquedas.
- **Análisis exploratorio**: Entender la estructura y patrones de datos no etiquetados.

---

## Resumen rápido

| Aspecto         | Descripción                             |
|-----------------|---------------------------------------|
| Tipo            | Clustering no supervisado              |
| Entrada         | Datos numéricos en $\mathbb{R}^d$     |
| Salida          | $K$ clusters con centroides             |
| Parámetros      | Número de clusters $K$, criterio de convergencia |
| Distancia usada | Euclidiana                            |
| Costo a minimizar | Suma de distancias cuadradas intra-cluster |

---
