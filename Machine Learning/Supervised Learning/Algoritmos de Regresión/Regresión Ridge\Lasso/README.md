# 🧷 Ridge y Lasso Regression en Machine Learning

La regresión lineal tradicional puede sufrir **sobreajuste (overfitting)** cuando el modelo es muy complejo, hay demasiadas variables o los datos tienen ruido. Las técnicas de **Ridge** y **Lasso Regression** introducen **regularización**, que ayuda a controlar la complejidad del modelo penalizando los coeficientes grandes.

---

## 🎯 ¿Qué hacen?

Ambas técnicas modifican el entrenamiento de un modelo de regresión lineal agregando una penalización al tamaño de los coeficientes:

- **Ridge Regression**: reduce el tamaño de todos los coeficientes, sin llegar a eliminarlos completamente.
- **Lasso Regression**: puede forzar algunos coeficientes a exactamente cero, eliminando variables del modelo (selección de características).

---
##  ✅ Ventajas
Técnica	Ventajas principales
Ridge	- Funciona bien con muchas variables correlacionadas
- Reduce la complejidad sin eliminar variables
Lasso	- Elimina variables irrelevantes automáticamente
- Mejora interpretabilidad al generar modelos más simples
---

##  🧪 Casos de Uso
Escenario	Técnica recomendada
Muchas variables correlacionadas	Ridge
Reducción automática de características	Lasso
Prevención de sobreajuste	Ambos
---

##  ⚠️ Limitaciones
Ridge no realiza selección de variables: incluye todas las características.

Lasso puede comportarse de forma inestable si las variables están altamente correlacionadas.

El valor de alpha debe seleccionarse cuidadosamente mediante validación cruzada.

Ambos pueden tener bajo rendimiento si no se ajusta adecuadamente la regularización.
---

##  📊 Comparación rápida
Característica	Ridge Regression	Lasso Regression
Penalización	L2	L1
Reduce coeficientes	Sí	Sí
Coeficientes en cero	No	Sí (puede eliminar variables)
Selección de variables	No	Sí
Interpretabilidad	Media	Alta
Multicolinealidad	Bien manejada	Puede ser un problema
---

##  📈 Visualización


Comparación geométrica entre la penalización L1 (Lasso) y L2 (Ridge)

