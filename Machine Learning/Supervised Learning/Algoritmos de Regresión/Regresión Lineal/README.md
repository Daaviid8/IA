# 📈 Regresión Lineal

La **regresión lineal** es uno de los algoritmos más sencillos y fundamentales del **aprendizaje supervisado**. Su objetivo es **modelar la relación lineal** entre una o más variables independientes (características) y una variable dependiente (resultado numérico).

---

## 🎯 ¿Qué hace?

Busca encontrar la **mejor recta (o hiperplano)** que predice un valor continuo ŷ a partir de variables de entrada x, utilizando la fórmula:

ŷ = wᵀ·x + b

Donde:

- x: vector de características (inputs)
- wᵀ: pesos (coeficientes del modelo)
- b: término independiente (bias)
- ŷ: predicción del modelo

---

## ⚙️ ¿Cómo se entrena?

Se ajustan los parámetros wᵀ y b minimizando una **función de pérdida**, generalmente el **error cuadrático medio (MSE)**:

MSE = (1/m) * Σ (ŷᵢ - yᵢ)²

Métodos comunes:

- **Ecuación normal**: solución directa (eficiente en pocos datos)
- **Descenso por gradiente**: iterativo (ideal para grandes datasets)

---

## ✅ Utilidades

La regresión lineal es útil cuando:

- Quieres **entender relaciones lineales** entre variables
- Necesitas un **modelo interpretable**
- Estás resolviendo un problema de **predicción numérica**
- Buscas una **solución rápida y eficiente**
- Quieres una **línea base (baseline)** para comparar modelos más complejos

---

## 🧪 Casos de Uso

| Área | Aplicación |
|------|------------|
| Economía | Predicción del PIB, salarios según experiencia |
| Salud | Relación entre dosis y efecto, predicción de presión arterial |
| Marketing | Estimación de ventas a partir de campañas publicitarias |
| Inmobiliaria | Predicción de precios de vivienda |
| Educación | Predicción del rendimiento académico según horas de estudio |

---

## ⚠️ Limitaciones

1. **Solo captura relaciones lineales**  
   No se ajusta bien a patrones no lineales.
   
2. **Sensible a outliers**  
   Los valores atípicos pueden sesgar el modelo.

3. **Multicolinealidad**  
   Si las variables independientes están muy correlacionadas entre sí, el modelo se vuelve inestable.

4. **No maneja bien relaciones complejas o interacciones no explícitas**

5. **Asume homoscedasticidad y errores independientes**  
   (en contextos estadísticos)

---

## 📊 Métricas comunes de evaluación

- **MSE** (Error cuadrático medio)
- **RMSE** (Raíz del MSE)
- **MAE** (Error absoluto medio)
- **\( R^2 \)**: porcentaje de la varianza explicada por el modelo

---

## 📈 Ejemplo gráfico

![Regresión Lineal](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/512px-Linear_regression.svg.png)

*Relación lineal entre dos variables con la recta ajustada.*

---

## 🧠 En el contexto de Machine Learning

- Es un modelo base ideal para comenzar un proyecto.
- Útil para interpretar relaciones entre variables antes de aplicar modelos más complejos.
- Puede mejorarse con regularización (Ridge, Lasso).
- Se incluye frecuentemente en pipelines automatizados.

---

## 📌 En resumen

| Característica | Detalle |
|----------------|---------|
| Tipo de modelo | Supervisado, regresión |
| Tipo de salida | Variable continua |
| Complejidad | Baja |
| Interpretabilidad | Alta |
| Velocidad de entrenamiento | Muy rápida |
| Recomendado como baseline | ✅ Sí |

---
