---
layout: default
title: Metodología
---

# Metodología

## Diseño experimental

Cinco fuentes × tres modos de entrenamiento × cuatro escalones = 60
configuraciones, cada una comparada contra su cota teórica y contra el mejor
baseline clásico a la misma tasa.

## Las fuentes

Todas producen vectores de 500 símbolos en {−1, +1}. La estructura (matrices
`W`, `P`) es **la misma en entrenamiento y prueba**; lo que cambia son los
símbolos. Esto es esencial: si la estructura cambiara entre particiones, no
habría nada que generalizar; si los símbolos se repitieran, el modelo podría
memorizar.

- **`random`** — bits i.i.d. uniformes. Control negativo: incompresible.
- **`oversamp`** — 125 símbolos, cada uno repetido 4 veces. Modela una señal
  sobremuestreada antes de diezmar, situación habitual en un front-end de
  receptor.
- **`markov`** — cadena de Markov con probabilidad de transición 0.05.
  Redundancia correlacional.
- **`lowdim`** — $\mathrm{sign}(Wz)$ con $z \in \mathbb{R}^{32}$. Redundancia
  geométrica: la fuente vive en un manifold de baja dimensión.
- **`code`** — código lineal sistemático de tasa 1/2 sobre GF(2), con checks de
  grado 3. Redundancia algebraica, el caso crítico.

## Modos de entrenamiento

**`direct`** — un autoencoder independiente por escalón. Es el control: mide el
mejor rendimiento alcanzable a cada tasa sin restricción de anidamiento.

**`nested`** — un solo encoder produce 250 bits; para el escalón L se anulan los
bits posteriores y el decoder reconstruye desde el prefijo. Todos los escalones
se optimizan en el mismo paso, lo que fuerza a que los primeros bits sean el
mejor sub-código posible. Produce un códec compatible en tasa.

**`stacked`** — preentrenamiento voraz por capas siguiendo a Hinton y
Salakhutdinov (2006): 500 → 250 → 125 → 70 → 35, cada etapa comprimiendo el
código binario de la anterior, con ajuste fino extremo a extremo posterior.

## Baselines clásicos

La vara. Todos usan **el mismo número de bits** que el autoencoder.

**PCA + cuantización uniforme.** El competidor justo: aprendido de datos, sin
oráculo, sin red neuronal. Se prueban combinaciones de $d$ componentes por $b$
bits tales que $d \cdot b = L$, y se reporta la mejor. La eigendescomposición se
calcula una sola vez por fuente.

**Decimación.** Envía 1 de cada $m$ símbolos. Se evalúan **dos**
reconstrucciones: *sample-and-hold* y *vecino más cercano*. La distinción
importa: sobre `oversamp` a 125 bits, hold da BER = 0.0000 y vecino da 0.1241.
Usar solo vecino subestimaría la vara y haría parecer ganador a un autoencoder
que en realidad pierde.

**Oráculos.** Cotas superiores de lo alcanzable con conocimiento perfecto de la
estructura. Sobre `code`, un oráculo que conoce la matriz de paridad transmite
los 250 bits sistemáticos y recalcula las paridades: BER exactamente 0 a tasa
0.5. Sobre `lowdim`, un oráculo que conoce $W$ transmite $z$ cuantizado.

## Métricas

**BER** — fracción de posiciones donde $\mathrm{sign}(\hat{x}) \neq x$.

**BER por confianza** — BER restringido al 50 % y al 90 % de los bits con mayor
$|$logit$|$. Aproxima lo que conseguiría un FEC de decisión blanda.

**Brecha entrenamiento/prueba** — sobre `random` debe ser grande y positiva: la
red memoriza el conjunto de entrenamiento pero no puede generalizar, que es
precisamente la huella de una fuente sin estructura explotable.

## Controles de validez

Antes de interpretar cualquier resultado, el verificador comprueba:

1. Ninguna configuración viola la cota de Shannon.
2. `random` no comprime en ningún escalón.
3. La brecha entrenamiento/prueba sobre `random` es positiva.
4. No hay NaN (divergencia del entrenamiento).
5. El barrido está completo: 60 filas, sin combinaciones faltantes.

## Test de calibración

Añadido tras detectar un fallo en la primera corrida completa. Entrena
`oversamp` con latente de 250 bits para una entropía de 125: al autoencoder le
sobra el doble de capacidad y la respuesta correcta es BER ≈ 0.

El razonamiento es el de calibrar un instrumento contra un patrón conocido. Si
una balanza indica 0.5 kg para una pesa patrón de 1 kg, la conclusión no es que
la gravedad cambió. Mientras el arnés no apruebe este test, sus números miden
defectos de implementación y no propiedades de los autoencoders.

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
