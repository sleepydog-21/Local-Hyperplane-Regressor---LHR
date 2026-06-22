# Local Hyperplane Regressor (LHR) & Geometrically Coherent LHR (LHR-C)

Este repositorio contiene la implementación y evaluación experimental de **LHR (Local Hyperplane Regressor)** y su extensión optimizada **LHR-C (Geometrically Coherent Local Hyperplane Regressor)**, diseñado para la predicción de valores numéricos e incertidumbre local mediante interpolaciones geométricas adaptativas.

---

## 1. ¿Qué es el Local Hyperplane Regressor (LHR)?

El **LHR** es un regresor no paramétrico que generaliza el algoritmo de k-vecinos más cercanos (k-NN) y la Regresión Lineal Local (LLR). En lugar de ajustar un único modelo sobre todos los vecinos, LHR:
1. Encuentra los $k$ vecinos más cercanos del punto de consulta $x^*$.
2. Muestrea múltiples subconjuntos pequeños de tamaño $n+1$ (donde $n$ es la dimensión del espacio de características).
3. Ajusta un hiperplano exacto para cada subconjunto.
4. Genera una predicción local combinando los resultados de todos los hiperplanos, lo que también permite estimar la incertidumbre empírica (desviación estándar local $\sigma$) de la predicción.

---

## 2. El Problema del Aplanamiento (*Smoothing*) y la Solución LHR-C

En el LHR clásico, los subconjuntos de vecinos se seleccionan de forma puramente aleatoria. Cuando el punto de consulta $x^*$ se encuentra cerca de un valor extremo (un pico o un valle de la función), sus vecinos tienden a rodearlo distribuyéndose a ambos lados del pico. Al mezclar puntos de pendientes opuestas, los hiperplanos resultantes se "aplanan", subestimando severamente la altura del pico.

Para solucionar esto, introducimos **Coherencia Geométrica (LHR-C)** a través de tres métodos:

### A. Coherencia Angular (Similitud de Coseno) - *Recomendado*
Definimos los vectores relativos de cada vecino respecto al punto de consulta $x^*$:
$$v_i = x_i - x^* \implies u_i = \frac{v_i}{\|v_i\|}$$

Definimos la **Coherencia Angular $C(S)$** para un subconjunto $S$ de tamaño $p$ como la similitud de coseno promedio de todos los pares:
$$C(S) = \frac{2}{p(p-1)} \sum_{i < j \in S} u_i \cdot u_j$$

Optimizamos este cálculo a complejidad $O(p)$ mediante la identidad algebraica:
$$\sum_{i < j \in S} u_i \cdot u_j = \frac{1}{2} \left( \left\| \sum_{i \in S} u_i \right\|^2 - p_{\text{non\_zero}} \right)$$

Muestreamos subconjuntos con probabilidad ponderada por coherencia y distancia:
$$P(S) \propto \exp(\lambda C(S)) \prod_{i \in S} \exp\left( -\frac{\|x_i - x^*\|^2}{h^2} \right)$$

Esto penaliza los subconjuntos con direcciones opuestas, asegurando que los hiperplanos provengan del mismo lado de la cima y preserven la curvatura local.

### B. Vecindad Jerárquica ("Anchor KNN")
Encuentra el vecino más cercano a $x^*$, y luego selecciona los siguientes $n$ puntos recursivamente a partir de los vecinos más cercanos de ese primer punto (agrupación local extrema).

### C. Partición por Signos/Ortantes
Proyecta los vecinos en el primer componente principal (PCA) del vecindario y divide rígidamente la selección de hiperplanos a un solo lado del eje.

---

## 3. Estructura del Repositorio

- `local_hyperplane_regressor.ipynb`: Cuaderno original con la formulación clásica de LHR.
- `local_hyperplane_regressor_coherent.ipynb`: Cuaderno principal con:
  - Implementación optimizada de `CoherentLocalHyperplaneRegressor`.
  - Experimento sintético 1D (cima de montaña $y = -|x| + \text{ruido}$) donde se demuestra visualmente cómo el LHR-C preserva el pico frente al aplanamiento del LHR clásico.
  - Evaluación rigurosa sobre 10 particiones del dataset real **California Housing**.
  - Validación estadística de calibración de incertidumbre ($\sigma$ local vs. error real) y cobertura de cuantiles.
  - Benchmark comparativo contra modelos del estado del arte (SOTA).
  - **Experimento Geométrico Sintético (Sección 9)**: Comparativa de LHR-C Angular contra LHR clásico, KNN, LLR, GPR y QRF en 8 funciones geométricas 1D.
  - **Mitigación del Mal Condicionamiento Numérico (Sección 10)**: Análisis del impacto del número de condición local $\kappa(A)$ en los errores de predicción y mitigación por ponderación suave ($w_j = \kappa_j^{-\gamma}$).

---

## 4. Resumen de Modelos en el Benchmark

En los cuadernos comparamos LHR-C contra los siguientes métodos del estado del arte para regresión y cuantificación de incertidumbre:
1. **Quantile Regression Forest (QRF)**: Estimación de la distribución empírica de hojas de un bosque aleatorio.
2. **Gaussian Process Regression (GPR)**: Regresión Bayesiana no paramétrica kernelizada.
3. **Gradient Boosting con Pérdida por Cuantiles**: Ajuste directo de cuantiles con GBDT.
4. **Predicción Conforme sobre LLR (Split Conformal LLR)**: Intervalos con garantía de cobertura libre de distribución.
5. **Bootstrap de Regresión Lineal Local (Bootstrap LLR)**: Remuestreo bootstrap local de vecinos.
6. **Modelos Lineales Locales Bayesianos (Local Bayesian Ridge)**: Regresión lineal Bayesiana ajustada localmente.

---

## 5. Mitigación del Mal Condicionamiento Numérico (Ponderación por $\kappa$)

En LHR, resolver sistemas lineales con pocos vecinos cercanos puede producir matrices de diseño locales $A$ mal condicionadas (cuando los vecinos están alineados o son casi coplanares). Esto provoca que pequeños ruidos en las etiquetas generen coeficientes gigantescos y predicciones erróneas atípicas (outliers).

Para solucionar esto, introducimos una **ponderación suave** basada en el número de condición de la matriz local:
$$w_j = \frac{1}{\kappa(A_j)^\gamma}$$
donde $\kappa(A_j) = \sqrt{\kappa(A_j^T A_j)}$ es el número de condición del sistema local y $\gamma \ge 0$ es la potencia de penalización.

### Resultados en California Housing (3 Splits, `cond_threshold=1000`):
- **Sin Mitigación ($\gamma = 0$)**: MAE = **0.5853**, RMSE = **0.9124** (los planos inestables ensanchan la distribución y aumentan el error).
- **Con Mitigación ($\gamma = 2$)**: MAE = **0.5184**, RMSE = **0.7571** (la ponderación por condición elimina eficazmente el impacto de los outliers sin descartar rígidamente la información).

Esta estrategia de ponderación suave supera a un filtro rígido (hard thresholding), permitiendo que la agregación del LHR-C sea extremadamente robusta.

---

## Requisitos de Instalación

Para ejecutar los notebooks, asegúrate de tener instalado:

```bash
pip install numpy scipy scikit-learn matplotlib jupyter nbformat nbconvert
```
