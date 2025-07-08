# Clustering Jerárquico (Hierarchical Clustering)

El clustering jerárquico es un método de agrupamiento que construye una **jerarquía de clusters** organizados en forma de un árbol llamado **dendrograma**. 

---

## Objetivo del Clustering Jerárquico

Agrupar los datos en una estructura jerárquica donde los clusters más similares se fusionan (o dividen) paso a paso, permitiendo explorar diferentes niveles de agrupamiento sin definir previamente el número de clusters.

---

## Tipos de Clustering Jerárquico

1. **Aglomerativo (bottom-up):**  
   - Comienza con cada punto como un cluster individual.  
   - En cada paso, se fusionan los dos clusters más similares hasta obtener un solo cluster que contiene todos los datos.

2. **Divisivo (top-down):**  
   - Comienza con todos los puntos en un solo cluster.  
   - En cada paso, se divide un cluster en dos, hasta que cada punto esté en un cluster separado.

---

## Cómo funciona el Clustering Jerárquico Aglomerativo (más común)

1. Inicializar con $n$ clusters (cada punto es un cluster).
2. Calcular la matriz de distancias entre todos los clusters.
3. Unir los dos clusters más cercanos según una medida de distancia entre clusters.
4. Actualizar la matriz de distancias (recalculando distancias con el nuevo cluster).
5. Repetir los pasos 3 y 4 hasta que todos los puntos formen un solo cluster.

---

## Matemáticas: Medidas de Distancia entre Clusters

Para fusionar clusters, se necesitan formas de medir la distancia entre grupos de puntos. Algunas comunes son:

- **Enlace sencillo (Single linkage):**

  $$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} \|x - y\|$$

- **Enlace completo (Complete linkage):**

  $$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} \|x - y\|$$

- **Enlace promedio (Average linkage):**

  $$d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} \|x - y\|$$

- **Enlace de Ward (Ward linkage):** Minimiza el aumento en la suma de cuadrados dentro de los clusters al fusionarlos. Es más complejo, pero produce clusters más compactos.

---

## Dendrograma

El resultado final es un árbol llamado dendrograma que muestra cómo los clusters se fueron fusionando (o dividiendo) en cada paso. Cortando el dendrograma en un nivel determinado se obtiene una partición específica con un número deseado de clusters.

---

## Ventajas del Clustering Jerárquico

- No requiere definir $K$ clusters antes de empezar.
- Permite explorar la estructura de los datos a diferentes niveles de granularidad.
- Fácil de interpretar visualmente mediante dendrogramas.
- Funciona con diferentes tipos de distancia y enlace.

---

## Limitaciones del Clustering Jerárquico

- **Escalabilidad:** Es costoso computacionalmente para conjuntos de datos muy grandes (complejidad $O(n^3)$ para algunas implementaciones).
- Sensible al ruido y outliers.
- La decisión de dónde cortar el dendrograma para definir clusters puede ser subjetiva.
- Puede sufrir el efecto "cadena" (enlace sencillo) o crear clusters poco balanceados.

---

## Casos de uso comunes

- **Biología y genética:** Agrupamiento de genes o especies.
- **Análisis de texto:** Agrupar documentos o palabras similares.
- **Marketing:** Segmentación de clientes con exploración jerárquica.
- **Detección de anomalías:** Identificar patrones o grupos atípicos.

---

## Resumen rápido

| Aspecto         | Descripción                              |
|-----------------|----------------------------------------|
| Tipo            | Clustering no supervisado, jerárquico   |
| Entrada         | Datos numéricos o categóricos           |
| Salida          | Dendrograma y clusters jerárquicos      |
| Parámetros      | Tipo de enlace, medida de distancia, criterio de corte |
| Distancia usada | Euclidiana u otras métricas             |
| Complejidad     | Costoso para datos grandes (usualmente $O(n^2)$ a $O(n^3)$) |

---
