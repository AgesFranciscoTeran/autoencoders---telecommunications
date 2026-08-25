---
layout: default
title: Marco teórico
---

# Marco teórico

## Por qué el BER solo no dice nada

Con un latente de 500 bits, un autoencoder aprende la identidad y obtiene
BER = 0. No comprimió nada. Cualquier afirmación de viabilidad tiene que ser un
par **(tasa, BER)**, nunca un BER suelto.

La tasa se define como

$$R = \frac{\text{bits del latente}}{500}$$

y para que sea contable en bits el latente debe ser **binario**. Un latente de
50 dimensiones en `float32` ocupa 1600 bits: más que los 500 originales. Por eso
el arnés usa el estimador straight-through (Bengio et al., 2013) en lugar de
latentes continuos.

## La cota inferior de Shannon

Para una fuente binaria estacionaria con distorsión de Hamming, la función
tasa-distorsión satisface

$$R(D) \;\geq\; \frac{H(X)}{n} - H_b(D)$$

donde $H_b$ es la entropía binaria. Despejando la distorsión:

$$D \;\geq\; H_b^{-1}\!\left(\frac{H(X)}{n} - R\right)$$

Esta cota vale para **cualquier** fuente estacionaria binaria, no solo i.i.d.,
por lo que se aplica uniformemente a las cinco fuentes del estudio. Se implementa
en `slb()` invirtiendo $H_b$ por bisección en la rama $[0, 0.5]$.

Consecuencia operativa: si un experimento reporta un BER por debajo de esta
línea, **tiene un bug** — típicamente fuga entre entrenamiento y prueba, o una
fuente que cambia entre particiones. El verificador comprueba esto antes que
cualquier otra cosa.

## Entropía de cada fuente

| Fuente | H (bits) | H/n | Cómo se calcula |
|---|---|---|---|
| `random` | 500.0 | 1.000 | trivial: 500 bits i.i.d. |
| `code` | 250.0 | 0.500 | 250 bits sistemáticos; las paridades son deterministas |
| `lowdim` | 164.9 | 0.330 | conteo de dicotomías de Cover (1965) |
| `markov` | 143.9 | 0.288 | $1 + (n-1) H_b(p)$ con $p = 0.05$ |
| `oversamp` | 125.0 | 0.250 | 125 símbolos, cada uno repetido 4 veces |

El caso de `lowdim` merece atención. La fuente es $x = \mathrm{sign}(Wz)$ con
$z \in \mathbb{R}^{32}$, y la intuición ingenua sugiere que su entropía son 32
bits. Es falso: lo que importa es **cuántos vectores de signos distintos** puede
producir, es decir el número de dicotomías linealmente separables de 500 puntos
en 32 dimensiones,

$$C(n,k) = 2\sum_{i=0}^{k-1}\binom{n-1}{i}$$

cuyo logaritmo da 164.9 bits. Usar 32 habría puesto la cota en el lugar
equivocado y habría hecho parecer que el autoencoder viola la teoría.

## La escalera y por qué esos valores

Los cuatro escalones — 35, 70, 125 y 250 bits — no son arbitrarios: **cortan las
entropías de las fuentes**.

| | L=35 | L=70 | L=125 | L=250 |
|---|---|---|---|---|
| `random` (H=500) | ≥ 0.3455 | ≥ 0.2834 | ≥ 0.2145 | ≥ 0.1100 |
| `code` (H=250) | ≥ 0.0881 | ≥ 0.0684 | ≥ 0.0417 | **≥ 0** |
| `lowdim` (H=164.9) | ≥ 0.0439 | ≥ 0.0291 | ≥ 0.0099 | **≥ 0** |
| `markov` (H=143.9) | ≥ 0.0348 | ≥ 0.0211 | ≥ 0.0040 | **≥ 0** |
| `oversamp` (H=125) | ≥ 0.0272 | ≥ 0.0146 | **≥ 0** | **≥ 0** |

Las celdas en negrita marcan los puntos donde la compresión **sin pérdida es
teóricamente posible**. L=250 es por tanto el test de "¿alcanza el óptimo?", y
los escalones inferiores miden qué tan grácil es la degradación. `oversamp` a
125 bits es especialmente informativo: es exactamente su entropía, así que
BER = 0 es alcanzable y de hecho un sample-and-hold trivial lo consigue.

## Refinamiento sucesivo

La escalera anidada entrena **un solo modelo** cuyo prefijo de 35 bits es
decodificable, igual que el de 70, el de 125 y el de 250. Esto corresponde a la
noción de refinamiento sucesivo de Equitz y Cover (1991), y en telecomunicaciones
equivale a un códec compatible en tasa: capa base más capas de mejora.

La teoría advierte que puede existir una **brecha** entre el rendimiento anidado
y el de un modelo dedicado a cada tasa. Por eso el arnés compara explícitamente
los modos `nested` y `direct` en lugar de suponer que anidar es gratis, y la
diferencia se reporta como resultado en sí misma.

## Umbral de viabilidad

En un enlace real el autoencoder no sería la última etapa: iría seguido de un
código corrector de errores. Con codificación moderna (LDPC, turbo), un BER
**previo a la decodificación** del orden de $10^{-2}$ se corrige hasta
esencialmente libre de errores. Sin código externo, los objetivos habituales
están entre $10^{-3}$ y $10^{-6}$.

Por eso el arnés reporta, además del BER, el **BER restringido a los bits más
confiables**. El decoder emite logits que funcionan como razones de verosimilitud
(LLR); si los errores se concentran en los bits de baja confianza, un decodificador
de decisión blanda los limpia. Un cociente BER/BER-top90 muy superior a 1 es el
argumento de viabilidad; un cociente cercano a 1 significa que los errores están
repartidos uniformemente y ni el FEC ayudará.
