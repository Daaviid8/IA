# DBSCAN: Density-Based Spatial Clustering of Applications with Noise

DBSCAN es un algoritmo de clustering basado en la densidad que agrupa puntos que están juntos en regiones de alta densidad y marca como ruido los puntos que están en regiones dispersas.

---

## Objetivo de DBSCAN

Encontrar clusters de forma arbitraria, basados en la densidad de puntos en el espacio, sin necesidad de definir el número de clusters a priori. Además, identifica puntos ruidosos o atípicos (outliers).

---

## Conceptos clave

- **$\varepsilon$ (epsilon):** Radio máximo para considerar vecinos cercanos a un punto.
- **MinPts:** Número mínimo de puntos necesarios dentro del radio $\varepsilon$ para que un punto sea considerado un **punto central** o **core point**.
- **Punto central (core point):** Un punto que tiene al menos `MinPts` vecinos dentro de su vecindad de radio $\varepsilon$.
- **Punto frontera (border point):** Punto que está dentro del vecindario de un core point pero tiene menos de `MinPts` vecinos.
- **Punto ruido (noise point):** Punto que no es core ni frontera, es decir, no pertenece a ningún cluster.

---

## Cómo funciona DBSCAN

1. Para cada punto en el dataset:
   - Si es un core point (tiene al menos `MinPts` vecinos en un radio $\varepsilon$), crear un nuevo cluster o asignarlo a uno existente.
2. Expandir el cluster agregando todos los puntos densamente conectados (vecinos directos y sus vecinos, etc.).
3. Repetir hasta que todos los puntos estén asignados a un cluster o marcados como ruido.

---

## Matemáticas de DBSCAN

- **Vecindad $\varepsilon$-vecino:** Para un punto $x_i$, su vecindad $\varepsilon$ es:

  $$N_\varepsilon(x_i) = \{x_j \mid \|x_j - x_i\| \leq \varepsilon \}$$

- **Condición de punto core:**

  $$|N_\varepsilon(x_i)| \geq \text{MinPts}$$

- **Definición de densidad conectada:**  
  Dos puntos $x_i$ y $x_j$ están densamente conectados si existe una cadena de puntos core conectados por vecindades $\varepsilon$ que los une.

---

## Ventajas de DBSCAN

- No necesita definir el número de clusters a priori.
- Puede encontrar clusters de formas arbitrarias (no solo esféricas).
- Detecta ruido (outliers) automáticamente.
- Funciona bien con datos con densidades variadas (hasta cierto punto).

---

## Limitaciones de DBSCAN

- Sensible a la elección de los parámetros $\varepsilon$ y `MinPts`.
- Difícil de aplicar cuando los clusters tienen densidades muy diferentes.
- No funciona bien en espacios de alta dimensión (la distancia Euclidiana pierde significado).
- Puede ser costoso para datasets muy grandes, aunque existen optimizaciones.

---

## Casos de uso comunes

- **Detección de anomalías:** Identificar puntos atípicos en datos financieros o de seguridad.
- **Geolocalización:** Agrupamiento de puntos geográficos (ej. localización de eventos).
- **Imágenes:** Segmentación basada en densidad.
- **Análisis de redes:** Detección de comunidades densas.

---

## Resumen rápido

| Aspecto         | Descripción                               |
|-----------------|-------------------------------------------|
| Tipo            | Clustering no supervisado basado en densidad |
| Parámetros      | $\varepsilon$ (radio), MinPts (min puntos vecinos) |
| Salida          | Clusters con forma arbitraria + puntos ruido |
| Distancia usada | Usualmente Euclidiana                      |
| Ventaja clave   | Detecta ruido y clusters de forma arbitraria |
| Limitación clave| Sensible a parámetros, dificultad en alta dimensión |

---
