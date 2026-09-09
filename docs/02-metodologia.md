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
- **`lowdim`** — $$\mathrm{sign}(Wz)$$ con $$z \in \mathbb{R}^{32}$$. Redundancia
  geométrica: la fuente vive en un manifold de baja dimensión.
- **`code`** — código lineal sistemático de tasa 1/2 sobre GF(2). Cada bit de
  paridad es el XOR de 3 bits sistemáticos elegidos al azar (peso de columna 3
  en `P`; equivalentemente, peso de fila 4 en la matriz de chequeo
  `H = [Pᵀ | I]`). Redundancia algebraica, el caso crítico.

## Por qué estas fuentes y no variantes de ellas

Tres restricciones acotan el espacio de diseño de cualquier fuente del arnés, y
conviene tenerlas explícitas porque son las que descartan la mayoría de las
alternativas antes de discutirlas:

1. **Alfabeto fijo.** Todas producen vectores en {−1,+1}⁵⁰⁰. Si una fuente
   saliera de ese alfabeto, la escalera, la cota y la vara dejarían de ser
   comparables entre fuentes.
2. **Entropía exacta, no estimada.** El piso de Shannon se dibuja bajo cada
   fuente. Una fuente cuya H solo se pueda estimar no puede llevar piso, y sin
   piso el BER vuelve a no significar nada.
3. **Redundancia demostrablemente explotable.** Cada fuente estructurada tiene
   un oráculo constructivo que la explota. Sin él, un fracaso del autoencoder
   sería ambiguo entre "no la vio" y "no había nada que ver".

### `lowdim`: por qué `sign(Wz)`

Sobre vectores binarios, "vive en un espacio de baja dimensión" admite
exactamente dos lecturas, y ambas están en el arnés:

- **Baja dimensión sobre GF(2)** — un subespacio lineal de {0,1}⁵⁰⁰. Eso *es*
  un código lineal, por definición: la lectura la ocupa `code`.
- **Baja dimensión sobre ℝ, proyectada al hipercubo** — un latente continuo
  z ∈ ℝᵏ y un mapa a signos. Es `lowdim`.

No hay una tercera. Las alternativas concretas que se consideraron mueren cada
una por una razón distinta:

| Alternativa | Por qué se descartó |
|---|---|
| Cuantizar `z` a *b* bits por coordenada | La salida deja de ser ±1: rompe el alfabeto común |
| Señal de banda limitada muestreada, luego signo | Introduce localidad temporal, que ya miden `oversamp` y `markov` |
| Mezcla de *M* centroides | Redundancia nominal, no geométrica; sin latente continuo ni relación entre Hamming y ángulo |
| Vector disperso + proyección + signo | Añade la dispersión como segunda estructura, confundida con la dimensión |
| `sign(f(z))` con `f` no lineal aleatoria | Sin entropía en forma cerrada: adiós al piso |

El generador `sign(Wz)` no es una construcción *ad hoc*: es el modelo de
medición de un bit, `y = sign(Ax)`, introducido por Boufounos y Baraniuk (2008),
y el régimen de los ADCs de un bit en massive MIMO (Li et al., 2017). La
identidad de hiperplanos aleatorios de Charikar (2002) —probabilidad de
discrepancia de signo igual a θ/π— añade que el BER sobre esta fuente **mide
error angular en ℝ³²** salvo un cambio de escala, de modo que la métrica no es
solo la del campo sino la natural de la fuente. Detalle en
[Bibliografía](04-bibliografia.md).

**Límite declarado.** `W` es gaussiana i.i.d. —hiperplanos en posición general,
que es la hipótesis bajo la cual el conteo de Cover es igualdad y no cota. Una
matriz de canal real es correlacionada y dispersa en el dominio angular. La
fuente modela la geometría de la medición, no la estadística de un canal.

### `code`: por qué grado 3

La elección original fue por hipótesis —se creía que grado 3 era donde el
descenso de gradiente se rompe— y **esa hipótesis fue refutada** por el
diagnóstico. La justificación correcta es otra, y es un estrujón por los dos
lados:

| grado | ¿PCA lo ve? | ¿lo ve una vara cuadrática? | ¿lo aprende un MLP supervisado? |
|---|---|---|---|
| 1 | **sí** | sí | sí (1.0000) |
| 2 | no | **sí** | sí (1.0000) |
| **3** | no | no | **sí (1.0000)** |
| 4 | no | no | **no (0.4995)** |

- **Grado 1 no es álgebra.** El bit de paridad es una copia; la covarianza tiene
  un 1 fuera de la diagonal y PCA lo resuelve. Desaparecería el hallazgo firme
  del proyecto.
- **De grado 2 en adelante PCA queda ciego**, y esto es demostrable en una
  línea: en representación ±1 el XOR es el producto, así que si xₚ = xᵢ·xⱼ
  entonces E[xₚ·xᵢ] = E[xᵢ²·xⱼ] = E[xⱼ] = 0. Todas las correlaciones por pares
  se anulan y la covarianza es la identidad.
- **Grado 2 se descarta por la vara.** Una interacción bilineal la recupera un
  mapa de características cuadrático, o simplemente probar los 31 125 pares. El
  resultado sería vulnerable a "tu baseline era débil". Con grado 3 haría falta
  un mapa cúbico —~2.6 × 10⁶ monomios por bit de paridad—, que ya no es una
  extensión razonable de PCA sino el oráculo disfrazado.
- **Grado 4 o más hace el resultado ininterpretable.** A grado 4 el MLP no
  aprende ni con supervisión directa, así que el fracaso del autoencoder no
  podría distinguirse de una barrera de aprendibilidad.

**Grado 3 es el único valor simultáneamente demasiado alto para cualquier vara
clásica razonable y suficientemente bajo para que el gradiente lo aprenda con
supervisión.** Es lo que permite atribuir el fracaso al objetivo de
reconstrucción y no a la dificultad de la fuente.

**Por qué `P` aleatoria y no un código estructurado.** Un Hamming, BCH o
Reed–Muller trae estructura cíclica o de cuerpo finito además de la paridad: si
el autoencoder mejorara, no se sabría cuál explotó. Además `code` funciona como
control de *no localidad* en el experimento de convolución precisamente porque
`P` es aleatoria; un código de banda o cíclico invalidaría ese control. Y la
construcción sistemática da k = 250 exacto, que los pares (n,k) de los códigos
estructurados no alcanzan. Un LDPC estándar es la extensión declarada en
[Hacia un paper](07-hacia-paper.md), no una alternativa descartada.

**Lo que no está diseñado, y conviene declarar.** El peso de columna es
exactamente 3, pero el peso de fila —en cuántos checks participa cada bit
sistemático— es aleatorio, aproximadamente Poisson(3); algunos sistemáticos
pueden no aparecer en ningún check. No es un ensemble LDPC regular ni una
distribución optimizada. La distancia mínima nunca se verificó: no afecta al
experimento, porque al ser sistemático H = 250 sale de la construcción
cualquiera que sea el rango de `P`, y el oráculo transmite los sistemáticos. Se
sortea una `P` por semilla, de modo que el resultado no cuelga de un código
particular.

## Modos de entrenamiento

**`direct`** — un autoencoder independiente por escalón. Es el control: mide el
mejor rendimiento alcanzable a cada tasa sin restricción de anidamiento.

**`nested`** — un solo encoder produce 250 bits; para el escalón L se anulan los
bits posteriores y el decoder reconstruye desde el prefijo. Todos los escalones
se optimizan en el mismo paso, lo que fuerza a que los primeros bits sean el
mejor sub-código posible. Produce un códec compatible en tasa.

**`stacked`** — cascada de compresores binarios: 500 → 250 → 125 → 70 → 35,
cada etapa un autoencoder profundo comprimiendo el código **binario** de la
anterior, con ajuste fino posterior. Se documentó originalmente como «la receta
de Hinton y Salakhutdinov (2006)», pero no lo es: la reproducción fiel (RBM con
divergencia contrastiva sobre activaciones continuas) está en `rbm_stack.py` y
da resultados opuestos.

## Baselines clásicos

La vara. Todos usan **el mismo número de bits** que el autoencoder.

**PCA + cuantización uniforme.** El competidor justo: aprendido de datos, sin
oráculo, sin red neuronal. Se prueban combinaciones de $$d$$ componentes por $$b$$
bits tales que $$d \cdot b = L$$, y se reporta la mejor. La eigendescomposición se
calcula una sola vez por fuente.

**Decimación.** Envía 1 de cada $$m$$ símbolos. Se evalúan **dos**
reconstrucciones: *sample-and-hold* y *vecino más cercano*. La distinción
importa: sobre `oversamp` a 125 bits, hold da BER = 0.0000 y vecino da 0.1241.
Usar solo vecino subestimaría la vara y haría parecer ganador a un autoencoder
que en realidad pierde.

**Oráculos.** Cotas superiores de lo alcanzable con conocimiento perfecto de la
estructura. Sobre `code`, un oráculo que conoce la matriz de paridad transmite
los 250 bits sistemáticos y recalcula las paridades: BER exactamente 0 a tasa
0.5. Sobre `lowdim`, un oráculo que conoce $$W$$ transmite $$z$$ cuantizado.

## Métricas

**BER** — fracción de posiciones donde $$\mathrm{sign}(\hat{x}) \neq x$$.

**BER por confianza** — BER restringido al 50 % y al 90 % de los bits con mayor
$$|$$logit$$|$$. Aproxima lo que conseguiría un FEC de decisión blanda.

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
