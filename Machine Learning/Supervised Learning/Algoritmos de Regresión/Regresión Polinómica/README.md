# 🧮 Regresión Polinómica en Machine Learning

La regresión polinómica es una extensión de la regresión lineal que permite modelar relaciones no lineales entre las variables de entrada y la variable objetivo. Aunque el modelo sigue siendo lineal en los coeficientes, se introducen potencias (polinomios) de las variables de entrada para capturar curvas en los datos.

---

## 🎯 ¿Qué hace?

En lugar de ajustar una línea recta, ajusta una curva (parábola, cúbica, etc.) generando nuevos atributos a partir de las potencias de las variables originales. De esta forma, permite modelar relaciones más complejas que no pueden ser representadas por una regresión lineal simple.

---

## 🧠 ¿Por qué usar regresión polinómica?

- Captura relaciones no lineales entre las variables.
- Es flexible y puede aproximarse a muchas funciones.
- Permite mejorar el rendimiento sin recurrir a modelos más complejos como redes neuronales o árboles de decisión.

---

## ⚙️ ¿Cómo se implementa?

En `scikit-learn`, se utiliza `PolynomialFeatures` para generar los términos polinomiales, que luego se usan como entrada en un modelo de regresión lineal.

Además, es común utilizar `Pipeline` para combinar el preprocesamiento (generación de polinomios) con el modelo de entrenamiento, facilitando la implementación y validación.

---

## ✅ Utilidades

- Predecir fenómenos que presentan una tendencia curva.
- Mejorar un modelo de regresión lineal cuando los residuos muestran patrones.
- Funciona bien en conjuntos de datos con relaciones suaves y continuas.

---

## 🧪 Casos de Uso

| Área         | Aplicación                                          |
|--------------|-----------------------------------------------------|
| Salud        | Modelar la relación dosis-efecto con curvatura      |
| Finanzas     | Predecir precios de activos con comportamiento no lineal |
| Física       | Modelar trayectorias o comportamientos cuadráticos  |
| Ingeniería   | Modelar fenómenos físicos complejos (resistencia, velocidad, fricción) |

---

## ⚠️ Limitaciones

1. Riesgo alto de sobreajuste si se usa un grado muy alto.
2. Requiere escalado o normalización en algunos casos.
3. Poca capacidad de generalización fuera del rango de entrenamiento.
4. Puede volverse ineficiente con muchos atributos y grados altos.
5. Menor interpretabilidad que la regresión lineal simple.

---

## 📊 Métricas comunes

- Error cuadrático medio (MSE)
- Error absoluto medio (MAE)
- R² (coeficiente de determinación)

---

## 📈 Ejemplo gráfico

![image](https://github.com/user-attachments/assets/ffa86729-2698-4f85-9047-e1c12fe1d5ee)
*Comparación entre regresión lineal (línea recta) y polinómica (curva).*

---

## 📌 En resumen

| Característica       | Detalle                              |
|----------------------|--------------------------------------|
| Tipo de modelo       | Supervisado, regresión no lineal     |
| Tipo de salida       | Variable continua                    |
| Complejidad          | Media                                |
| Interpretabilidad    | Media-baja                           |
| Requiere preprocesado| Sí (PolynomialFeatures)              |
| Riesgo de overfitting| Alto si el grado no se regula        |

---
