# 📊 Regresión Logística: 

## 📌 ¿Qué es la regresión logística?

La **regresión logística** es un modelo estadístico usado para **predecir la probabilidad** de que ocurra un evento binario (por ejemplo, **éxito o fracaso**, **sí o no**, **enfermo o sano**).

A diferencia de la regresión lineal, que predice valores continuos, la regresión logística **predice probabilidades entre 0 y 1**.

---

## 🔢 La forma del modelo

La regresión logística usa la **función sigmoide (o logística)** para transformar una combinación lineal de las variables predictoras en una probabilidad.

**Modelo lineal (como en regresión lineal):**  
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

**Función logística (sigmoide):**  
P(y = 1 | x) = 1 / (1 + e^(-z)) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))

Esto garantiza que el resultado esté siempre entre 0 y 1, como una probabilidad.

---

## 📈 ¿Qué se ajusta realmente?

Se estima el conjunto de coeficientes `βᵢ` que mejor ajusta los datos.  
No se minimiza el error cuadrático como en la regresión lineal, sino que se **maximiza la verosimilitud** (*maximum likelihood*).

La **función de verosimilitud** mide cuán probable es observar los datos reales dados los parámetros del modelo.

---

## 🧮 Interpretación de los coeficientes

Cada `βᵢ` representa el **log-odds** (logaritmo del cociente de probabilidades) de que `y = 1` por unidad de cambio en `xᵢ`, manteniendo las demás variables constantes:

log(P(y=1)/P(y=0)) = β₀ + β₁x₁ + ... + βₙxₙ

El término `P(y=1)/P(y=0)` se llama **odds** (probabilidad entre su complemento).

---

## ✅ ¿Cuándo se usa?

- Diagnóstico médico (¿el paciente tiene una enfermedad?)
- Clasificación binaria (¿un correo es spam?)
- Predicción de comportamiento (¿un cliente comprará o no?)
- Cualquier situación con una variable de salida binaria (0 o 1)

---

## 🧠 En resumen

- **Entrada:** Variables `x₁, x₂, ..., xₙ`
- **Salida:** Probabilidad de una clase (0 o 1)
- **Modelo:** Aplica una función logística a una combinación lineal
- **Estimación:** Se hace vía **máxima verosimilitud**
- **Interpretación:** Coeficientes afectan los log-odds de la clase positiva

![image](https://github.com/user-attachments/assets/96cf7473-d7bc-44ae-a213-584f00c309df)

