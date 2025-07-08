# Gaussian Mixture Models (GMM)

Un **Gaussian Mixture Model (GMM)** es un modelo probabilístico que asume que los datos se generan a partir de una combinación de varias distribuciones normales (gaussianas) con diferentes medias y covarianzas. Es una técnica muy utilizada para clustering y modelado estadístico.

---

## Objetivo de GMM

Modelar la distribución de los datos como una mezcla de $K$ gaussianas, y asignar probabilísticamente cada punto a uno o más clusters, permitiendo formas de clusters más flexibles que K-means.

---

## Modelo matemático

La densidad de probabilidad de un punto $x$ en un GMM con $K$ componentes es:

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

donde:

- $\pi_k$ es el peso (o proporción) del componente $k$, con $\sum_{k=1}^K \pi_k = 1$ y $\pi_k \geq 0$.
- $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ es la función de densidad de la distribución normal multivariada con media $\mu_k$ y matriz de covarianza $\Sigma_k$:

$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

---

## Entrenamiento: Expectation-Maximization (EM)

Se usa el algoritmo EM para estimar los parámetros $(\pi_k, \mu_k, \Sigma_k)$ que maximizan la probabilidad de los datos:

- **E-step (Expectation):** Calcular la probabilidad de que cada punto pertenezca a cada componente (responsabilidades):

$$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

- **M-step (Maximization):** Actualizar los parámetros usando las responsabilidades:

$$N_k = \sum_{i=1}^n \gamma_{ik}$$

$$\mu_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i$$

$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$

$$\pi_k = \frac{N_k}{n}$$

Se repiten E y M hasta convergencia.

---

## Ventajas de GMM

- Permite clusters con formas elípticas (no solo esféricas).
- Clustering probabilístico: cada punto tiene una probabilidad de pertenecer a cada cluster.
- Más flexible que K-means.
- Puede modelar la distribución completa de los datos.

---

## Limitaciones de GMM

- Requiere definir $K$, el número de componentes, a priori.
- Puede converger a óptimos locales, depende de la inicialización.
- Asume que los datos se ajustan bien a mezclas gaussianas.
- Computacionalmente más costoso que K-means.
- Sensible a outliers.

---

## Casos de uso comunes

- **Segmentación de clientes:** Agrupamiento probabilístico con incertidumbre.
- **Reconocimiento de voz:** Modelado estadístico de señales acústicas.
- **Visión por computadora:** Detección y segmentación de objetos.
- **Análisis de datos:** Modelado flexible de distribuciones complejas.

---

## Resumen rápido

| Aspecto         | Descripción                               |
|-----------------|-------------------------------------------|
| Tipo            | Modelo probabilístico, clustering suave   |
| Parámetros      | Número de gaussianas $K$, medias, covarianzas, pesos |
| Salida          | Probabilidades de pertenencia a clusters  |
| Forma clusters  | Elíptica (no esférica)                     |
| Algoritmo       | Expectation-Maximization (EM)              |
| Ventaja clave   | Flexibilidad en forma y asignación probabilística |
| Limitación clave| Requiere definir $K$, sensibilidad a inicialización |

---
