# 🔗 Elastic Net Regression

## 🤔 ¿Qué es Elastic Net?

Elastic Net es una técnica de regresión lineal regularizada que combina las penalizaciones de **Ridge (L2)** y **Lasso (L1)** en un solo modelo. Su objetivo es obtener lo mejor de ambos métodos: regularización que reduzca la complejidad del modelo y selección automática de variables.

## ⚙️ ¿Cómo funciona?

Elastic Net agrega a la función de pérdida de la regresión lineal dos términos de penalización simultáneamente:

- **L1 (Lasso):** promueve la sparsidad, es decir, hace que algunos coeficientes se vuelvan exactamente cero, facilitando la selección de variables.
- **L2 (Ridge):** reduce el tamaño de los coeficientes, ayudando a manejar multicolinealidad y evitando coeficientes extremadamente grandes.

La función objetivo que minimiza Elastic Net es:

$$
J(\beta) = \frac{1}{2n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2 + \alpha \left( \lambda \|\beta\|_1 + \frac{1-\lambda}{2} \|\beta\|_2^2 \right)
$$

donde:

Donde:

- \( J(\beta) \) es la **función objetivo** que queremos minimizar para encontrar los mejores coeficientes \(\beta\).
- \( n \) es el número total de muestras en el dataset.
- \( y_i \) es el valor real de la variable objetivo para la muestra \(i\).
- \( \hat{y}_i \) es el valor predicho por el modelo para la muestra \(i\).
- \( \beta \) es el vector de coeficientes del modelo (los parámetros que queremos aprender).
- \( \|\beta\|_1 = \sum_{j} |\beta_j| \) es la norma L1 (suma de valores absolutos de los coeficientes).
- \( \|\beta\|_2^2 = \sum_{j} \beta_j^2 \) es la norma L2 al cuadrado (suma de los cuadrados de los coeficientes).
- \( \alpha \) controla la fuerza total de la regularización (qué tanto penalizamos la complejidad del modelo).
- \( \lambda \) (o `l1_ratio`) controla la proporción entre penalización L1 y L2:
  - Cuando \( \lambda = 1 \), Elastic Net es igual a Lasso (solo L1).
  - Cuando \( \lambda = 0 \), Elastic Net es igual a Ridge (solo L2).

## 🧩 ¿En qué casos se usa?

Elastic Net es especialmente útil cuando:

- 📊 Hay **muchas variables** (alta dimensionalidad).
- 🔗 Las variables están **correlacionadas** entre sí (multicolinealidad).
- 🗑️ Se desea hacer **selección de variables**, pero sin perder la estabilidad que ofrece Ridge.
- ⚖️ Se quiere un compromiso entre la **sparsidad** de Lasso y la **estabilidad** de Ridge.

## ✨ Ventajas respecto a otros algoritmos de regresión

- Combina lo mejor de Ridge y Lasso, permitiendo un balance ajustable entre regularización y selección de variables.
- Más robusto que Lasso en presencia de variables altamente correlacionadas.
- Puede seleccionar grupos de variables correlacionadas en lugar de elegir arbitrariamente solo una.
- Flexibilidad para adaptarse mejor a distintos tipos de problemas mediante ajuste de hiperparámetros.

## ⚠️ Limitaciones

- Requiere optimización de dos hiperparámetros (`alpha` y `l1_ratio`), lo que puede aumentar el costo computacional.
- No siempre es trivial interpretar el efecto individual de las variables cuando se combinan penalizaciones.
- Si no se ajustan bien los hiperparámetros, puede no superar en desempeño a Ridge o Lasso por separado.
- No es adecuado si la relación entre variables y objetivo es altamente no lineal sin transformar los datos primero.

