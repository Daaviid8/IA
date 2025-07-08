# Métricas de Evaluación de Modelos

Cuando entrenamos modelos de machine learning, necesitamos medir qué tan bien están funcionando. Para eso, usamos **métricas de evaluación**. Aquí te explico algunas de las más comunes, con sus fórmulas y su objetivo.

---

## Métricas para modelos de clasificación

### 1. Accuracy (Exactitud)
- **¿Qué mide?** La proporción de predicciones correctas (tanto positivas como negativas) sobre el total de datos.
- **Fórmula:**

  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

  Donde:  
  - $TP$: Verdaderos Positivos  
  - $TN$: Verdaderos Negativos  
  - $FP$: Falsos Positivos  
  - $FN$: Falsos Negativos

- **Objetivo:** Saber qué porcentaje de todas las predicciones fueron correctas.

---

### 2. Precision (Precisión)
- **¿Qué mide?** De todas las veces que el modelo dijo "positivo", cuántas veces acertó realmente.
- **Fórmula:**

  $$\text{Precision} = \frac{TP}{TP + FP}$$

- **Objetivo:** Minimizar falsos positivos.

---

### 3. Recall (Sensibilidad o Exhaustividad)
- **¿Qué mide?** De todos los positivos reales, cuántos fue capaz de detectar el modelo.
- **Fórmula:**

  $$\text{Recall} = \frac{TP}{TP + FN}$$

- **Objetivo:** Minimizar falsos negativos.

---

### 4. F1-Score
- **¿Qué mide?** El balance entre precisión y recall, es su promedio armónico.
- **Fórmula:**

  $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Objetivo:** Tener un solo número que combine precisión y recall.

---

### 5. ROC-AUC (Área Bajo la Curva ROC)
- **¿Qué mide?** La capacidad del modelo para distinguir entre clases (positivo vs negativo) a diferentes umbrales.
- **Concepto:**

  Se grafica la tasa de verdaderos positivos (Recall) frente a la tasa de falsos positivos $\left(\frac{FP}{FP + TN}\right)$ para distintos umbrales. El área bajo esta curva (AUC) indica el rendimiento general.

- **Valor:**  
  - 1 = perfecto  
  - 0.5 = modelo aleatorio  
  - Menor a 0.5 = peor que aleatorio

- **Objetivo:** Evaluar la discriminación del modelo.

---

### 6. Matriz de Confusión
- **¿Qué es?** Tabla que muestra los conteos de predicciones correctas e incorrectas, divididas en:
  
  |               | Predicción Positiva | Predicción Negativa |
  |---------------|---------------------|---------------------|
  | **Real Positivo** | TP                  | FN                  |
  | **Real Negativo** | FP                  | TN                  |

- **Objetivo:** Entender en detalle qué errores comete el modelo.

---

## Métricas para modelos de regresión

### 7. MSE (Error Cuadrático Medio)
- **¿Qué mide?** El promedio de los cuadrados de los errores (diferencias entre valor real y predicho).
- **Fórmula:**

  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$$

- **Objetivo:** Penalizar errores grandes, ideal para medir la precisión del modelo.

---

### 8. RMSE (Raíz del Error Cuadrático Medio)
- **¿Qué mide?** Es la raíz cuadrada del MSE, para devolver el error a las mismas unidades de la variable original.
- **Fórmula:**

  $$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}$$

- **Objetivo:** Interpretar el error en las mismas unidades de la variable objetivo.

---

### 9. MAE (Error Absoluto Medio)
- **¿Qué mide?** El promedio de las diferencias absolutas entre valor real y predicho.
- **Fórmula:**

  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|$$

- **Objetivo:** Medir el error promedio sin penalizar tanto los errores grandes como el MSE.

---

### 10. R² (Coeficiente de Determinación)
- **¿Qué mide?** Qué proporción de la variabilidad total de los datos es explicada por el modelo.
- **Fórmula:**

  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y_i})^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

  Donde $\bar{y}$ es el promedio de los valores reales.

- **Valor:**  
  - 1 = modelo perfecto  
  - 0 = modelo que no explica nada mejor que la media  
  - Puede ser negativo si el modelo es peor que simplemente predecir la media

- **Objetivo:** Evaluar qué tan bien el modelo explica los datos.

---

# Resumen visual rápido

| Métrica     | Tipo        | Rango/Valor       | Objetivo principal                     |
|-------------|-------------|-------------------|--------------------------------------|
| Accuracy    | Clasificación | 0 a 1            | % total de aciertos                  |
| Precision   | Clasificación | 0 a 1            | Evitar falsos positivos              |
| Recall      | Clasificación | 0 a 1            | Evitar falsos negativos              |
| F1-Score    | Clasificación | 0 a 1            | Balance entre precision y recall    |
| ROC-AUC     | Clasificación | 0.5 a 1          | Calidad general discriminativa      |
| Conf. Matrix| Clasificación | -                | Entender errores detalladamente     |
| MSE         | Regresión    | 0 a ∞             | Penaliza errores grandes             |
| RMSE        | Regresión    | 0 a ∞             | Error promedio en unidades originales|
| MAE         | Regresión    | 0 a ∞             | Error promedio absoluto              |
| R²          | Regresión    | Puede ser negativo a 1 | Explicación de la varianza          |

---
