---
layout: default
title: Resultados
---

# Resultados

Todo lo de esta página está validado: el arnés pasa una prueba de calibración
con respuesta conocida, y los resultados de autoencoder se reportan sobre
**cuatro semillas independientes** que varían tanto la instancia de la fuente
como la inicialización del modelo.

---

## La respuesta

**Los autoencoders no son viables para comprimir flujos de bits BPSK a tasas
agresivas, y son viables en una franja estrecha con decisión blanda.** Con el
umbral absoluto de telecomunicaciones —un BER previo a la decodificación de
10⁻², corregible con LDPC o turbo— **18 de 20 puntos de operación del barrido
principal no lo alcanzan**. Pero el mejor punto no trivial, `markov` L=250 con
escalera estrecha y preentrenamiento RBM, queda en BER 0.0107 con una
estructura de confianza que un decodificador blando aprovecha: el 90 % de los
bits más confiables llega a **0.0011** y el 10 % restante está marcado como
dudoso. Ver «El punto que roza el umbral».

| categoría | criterio | puntos |
|---|---|---|
| Buena sin FEC | BER < 10⁻³ | 2/20 |
| Usable con FEC | BER < 10⁻² | 2/20 |
| Marginal | BER < 10⁻¹ | 7/20 |
| Inservible | BER ≥ 10⁻¹ | 13/20 |

Los dos que pasan son `oversamp` a 125 y 250 bits, donde un *sample-and-hold* de
tres líneas consigue lo mismo. El mejor resultado sobre estructura no trivial es
`markov` a 250 bits: BER 0.0344, es decir **17 bits errados de cada 500**,
equivalente a un enlace BPSK sin codificar a 2.2 dB.

Con escalera estrecha y preentrenamiento RBM (ver más abajo), ese mismo punto
baja a **0.0107** — a siete diezmilésimas del umbral. Sugiere que la
inviabilidad es de las tasas agresivas, no de todo el mapa.

---

## Pero hay una franja donde sí superan a lo clásico

Seis configuraciones ganan al mejor método clásico en **4 de 4 semillas**, con
la vara recalculada (ver «La vara estaba débil», más abajo):

| fuente | L | autoencoder | vara | margen |
|---|---|---|---|---|
| `markov` | 250 | **0.0107** | 0.0250 | **+0.0143** (57 %) |
| `lowdim` | 35 | 0.1870 | 0.1932 | +0.0062 |
| `lowdim` | 250 | 0.0346 | 0.0403 | +0.0057 |
| `markov` | 35 | 0.1498 | 0.1529 | +0.0031 |
| `markov` | 125 | 0.0464 | 0.0490 | +0.0026 |
| `markov` | 70 | 0.0902 | 0.0915 | +0.0013 |

Y dos configuraciones **pierden** con la vara corregida, ambas en la zona media
de `lowdim`:

| fuente | L | autoencoder | vara | |
|---|---|---|---|---|
| `lowdim` | 70 | 0.1051 | **0.0977** | pierde por 0.0074 |
| `lowdim` | 125 | 0.0607 | **0.0513** | pierde por 0.0094 |

El patrón que queda es más nítido que el anterior:

> **Sobre estructura correlacional (`markov`) el autoencoder gana a las cuatro
> tasas. Sobre estructura geométrica (`lowdim`) gana solo en los extremos: en la
> zona media, PCA con cuantizador óptimo lo supera.**

Tiene explicación. `lowdim` es sign(Wz), una fuente **casi lineal**, que es el
terreno de PCA — Baldi y Hornik otra vez, pero ahora con la vara bien puesta.
`markov` tiene correlación local que PCA no captura, y ahí la vara real no es
PCA sino la decimación.

El punto más fuerte del proyecto pasa a ser **`markov` L=250 con
preentrenamiento RBM**: margen del 57 % y, además, el único punto no trivial que
roza el umbral operativo. El mejor resultado es también el más útil.

### Control negativo

`random nested` L=250 gana en 1 de 4 semillas, con la varianza más alta de toda
la tabla (0.2453 a 0.2611, sd 0.0066). Es ruido oscilando alrededor de la vara,
que es exactamente lo que debe ocurrir sobre datos incompresibles. Una victoria
consistente ahí habría invalidado el experimento.

---

## La vara estaba débil, y eso invalidó dos victorias

El baseline de PCA cuantizaba las proyecciones con un rango **min/max de las
muestras de entrenamiento**. Ese rango crece con el número de muestras —se
observan colas más extremas— así que los intervalos se ensanchan y la vara
empeora cuantos más datos se le dan:

| n_train | rango min/max | rango por percentiles |
|---|---|---|
| 20 000 | 0.1772 | 0.1316 |
| 100 000 | 0.1836 | 0.1318 |
| 400 000 | **0.1866** | 0.1317 |

Un baseline que depende del tamaño de muestra no es un baseline. Y explica por
qué circulaban dos valores para `lowdim` L=70: 0.1776 venía de las tablas
generadas con 20 000 muestras y 0.1861 del barrido con 400 000.

Se recalculó con el cuantizador escalar óptimo en MSE (Lloyd-Max por
componente, ajustado en entrenamiento) y, por si la reconstrucción también era
subóptima, con **síntesis por mínimos cuadrados** en lugar de transponer la
base:

| | uniforme + Vᵀ | uniforme + LS | Lloyd + Vᵀ | Lloyd + LS |
|---|---|---|---|---|
| `lowdim` L=70 | 0.0985 | 0.0985 | **0.0977** | 0.0977 |
| `markov` L=250 | 0.0311 | 0.0311 | 0.0307 | 0.0307 |

La síntesis por mínimos cuadrados **no mejora nada** en ninguna de las 16
celdas, lo cual es esperable: con PCA la base de análisis es ortonormal, así que
la transpuesta ya era la síntesis óptima. Dos rondas independientes de
fortalecimiento convergen al mismo número, de modo que el resultado ya no es
plausiblemente atribuible a un baseline débil.

**Consecuencia.** La vara de `lowdim` L=70 pasó de 0.1861 a **0.0977**, y con
ella cayó la victoria que se había reportado a 216 σ. `lowdim` L=125 también
cae. Quedan 6 de 8.

**Asimetría que conviene declarar.** Las varas se recalcularon *después* de
medir el autoencoder, con una implementación mejor. No es injusto —la vara no
depende del autoencoder ni se ajustó mirando sus resultados— pero el orden
cronológico debe constar.

**Límite conocido.** El cuantizador es óptimo en MSE de la proyección, no en
BER. Uno que minimizara BER directamente podría ser algo más fuerte; es un
problema discreto no convexo sin solución estándar. La evidencia sugiere
retornos decrecientes: de min/max a percentiles el cambio fue grande, de
percentiles a Lloyd pequeño, y de Lloyd a mínimos cuadrados nulo.

---

## La frontera se mueve con la arquitectura

Una **escalera estrecha** (500 → 250 → 125 → 70 → 35, sigmoides intermedias,
mismo cuello binario) comparada contra el MLP ancho con **protocolo idéntico**
—mismos pasos, misma tasa de aprendizaje, misma selección de checkpoint por
validación— sobre 4 semillas:

| caso | vara | MLP ancho | escalera | quién gana a la vara |
|---|---|---|---|---|
| markov L=35 | 0.1531 | **0.1564** | 0.1979 | ninguno, con este protocolo |
| markov L=125 | 0.0488 | 0.0614 | **0.0464** | escalera |
| markov L=250 | 0.0250 | 0.0288 | **0.0135** | escalera, por 46 % |
| lowdim L=250 | 0.0492 | 0.0553 | **0.0386** | escalera |

**Interacción arquitectura–tasa.** A tasa agresiva (L=35) el MLP ancho gana por
0.04; a tasas holgadas (L ≥ 125) la escalera gana por ~0.015. El signo se
invierte. Hipótesis: cuando el cuello es apretado la capacidad ayuda; cuando
sobra, el MLP ancho encuentra soluciones peores y la escalera —que ya comprime
en su primera capa— no tiene con qué sobreajustar.

**La escalera es estable; el MLP ancho no.** Diferencia entre BER final y BER al
mejor checkpoint: +0.0000 en las 16 corridas de la escalera; hasta +0.009 en el
MLP ancho, que en `markov` L=35 alcanza su pico en el paso 750 de 30 000 y luego
se degrada. Los resultados de v4 a tasas altas medían el MLP ancho degradado,
pero corregir eso explica solo un tercio de la brecha; el resto es arquitectura.

**Consecuencia sobre el mapa.** Con la arquitectura elegida por tasa, el
autoencoder supera a la vara clásica en `markov` a **todas** las tasas y en
`lowdim` a tres de cuatro. Elegir arquitectura por punto es un ajuste y debe
reportarse como tal; cada punto está validado sobre 4 semillas.

**Una victoria previa se rebaja.** `markov` L=35 con el MLP ancho ganaba en v4
por 4σ; con este protocolo pierde (0.1564 contra 0.1531). Es sensible al
protocolo de entrenamiento. Con preentrenamiento sí es robusta.

---

## El preentrenamiento por capas sí ayuda

El modo `stacked` de v4 era una **cascada de compresores binarios** —cada etapa
un autoencoder profundo comprimiendo el código binario de la anterior— y fue el
peor modo. Se etiquetó erróneamente como «la receta de Hinton y Salakhutdinov».

La reproducción fiel —**RBM con divergencia contrastiva**, decoder `Wᵀ`, o
autoencoder superficial por capa sobre activaciones continuas— sobre la
escalera estrecha, 4 semillas:

| caso | aleatoria | pre. AE | pre. RBM | mejora |
|---|---|---|---|---|
| lowdim L=35 | 0.2610 | **0.1833** | 0.1911 | 29.8 % |
| lowdim L=70 | 0.1376 | **0.1140** | 0.1146 | 17.2 % |
| lowdim L=250 | 0.0386 | 0.0394 | **0.0369** | 4.4 % |
| markov L=35 | 0.1979 | **0.1418** | 0.1732 | 28.3 % |
| markov L=70 | 0.1010 | **0.0838** | 0.0891 | 17.0 % |
| markov L=250 | 0.0135 | 0.0143 | **0.0107** | 20.7 % |

Preentrenar gana en **16 de 16** celdas de `lowdim` y `markov`, 4/4 semillas. La
predicción registrada antes de correrlo —que no ayudaría porque la optimización
moderna ya resuelve lo que resolvía en 2006— fue **refutada**.

- **RBM y AE no son equivalentes.** El AE gana a tasas bajas (`markov` L=35:
  0.1418 contra 0.1732); la RBM a tasas altas (`markov` L=250: 0.0107 contra
  0.0143). El objetivo de la etapa importa, en dirección opuesta según la tasa.
- **El ajuste fino daña la inicialización.** Los preentrenados alcanzan su mejor
  checkpoint entre los pasos 750 y 6 000 de 30 000 y luego se degradan. Hinton
  usaba tasas muy pequeñas para ajustar; con `lr = 1e-3` el ajuste destruye
  parte de lo ganado. Sin selección de checkpoint habría parecido inútil.
- **Sobre `code`, nada cambia.** Los tres brazos quedan al nivel de ruido. Ni
  una RBM —generativa, binaria— ve la paridad de grado 3.

**El dato que más importa:** `markov` L=250 con RBM da BER **0.0107**, a siete
diezmilésimas del umbral corregible por FEC. Es el primer punto no trivial del
proyecto que roza calidad operativa.

---

## El punto que roza el umbral

`markov` L=250, escalera estrecha, preentrenamiento RBM, 4 semillas:

| ajuste fino | BER total | BER top 90 % | BER top 50 % | cruza 10⁻² |
|---|---|---|---|---|
| `lr = 1e-3` | 0.0107 ± 0.0001 | **0.0011 ± 0.0000** | 0.0001 | con salida blanda, 4/4 |
| `lr = 1e-4` | 0.0186 ± 0.0001 | 0.0029 | 0.0004 | con salida blanda, 4/4 |

**Con BER total, no cruza** en ninguna semilla: 0.0107 contra 0.0100.

**Con salida blanda, cruza en 4/4 y por un orden de magnitud.** Los errores están
concentrados: despejando la mezcla, el 10 % de bits menos confiables tiene BER
≈ 0.097 —casi moneda al aire— mientras el 90 % restante está en 10⁻³. Un
decodificador de decisión blanda ve un canal donde nueve de cada diez bits
llegan limpios y el décimo viene **marcado como dudoso** por su propio LLR. Es
la estructura que LDPC y turbo aprovechan mejor.

**Lo que esto demuestra y lo que no.** Demuestra que la estructura de confianza
es favorable a FEC de decisión blanda. No es una simulación post-FEC: el BER
final con un decodificador real requiere implementarlo. La afirmación defendible
es *«viable con FEC de decisión blanda, en una franja estrecha»*, no *«alcanza
10⁻²»*.

**Una hipótesis más refutada.** Se esperaba que un ajuste fino con `lr = 1e-4`
mejorara el 0.0107, porque a L=35 los preentrenados se degradaban tras un pico
temprano. En L=250 no ocurre: el mejor checkpoint está en el paso ~27 000 de
30 000 y el ajuste sigue mejorando hasta el final. Con `lr = 1e-4` el modelo
queda subentrenado y el BER es un 74 % peor. La degradación temprana era
específica de tasas agresivas.

---

## La arquitectura debe coincidir con el tipo de estructura

Un encoder convolucional 1D frente al MLP, con la localidad como variable:

| fuente | localidad | efecto de la convolución |
|---|---|---|
| oversamp | sí (período 4) | **+10.3 %** |
| markov | sí (correlación local) | +3.5 % |
| code | no (P aleatoria) | +2.4 % |
| lowdim | no (W aleatoria) | **−21.4 %** |

La convolución ayuda donde hay estructura local y **estorba** donde la
estructura es global: sin localidad que explotar, pierde la capacidad de mezcla
global del MLP sin ganar nada a cambio. La confirmación en las dos direcciones
es lo que hace sólido el resultado.

Aporta una **victoria adicional**: `oversamp` L=70 con convolución da 0.2271
frente a una vara de 0.2330, un punto que el MLP perdía.

---

## El resultado negativo sobre redundancia algebraica

Sobre `code` —código de bloque con checks de grado 3— el autoencoder fracasa en
los cuatro escalones y en todas las semillas.

| estrategia | BER a L=250 |
|---|---|
| copiar 250 bits, adivinar el resto | 0.2500 |
| **autoencoder observado** | **0.1982** |
| oráculo que conoce la matriz de paridad | 0.0000 |

El autoencoder recorre solo el **21 %** del camino entre la estrategia trivial y
el óptimo. Medido contra su propia cota queda **más lejos del óptimo en `code`
(+0.198) que sobre ruido puro (+0.148)**.

Tres evidencias convergentes:

1. **PCA es igualmente ciego.** Sobre 10 semillas, su BER en `code` y en
   `random` son indistinguibles dentro de 1σ en los cuatro escalones.
2. **Un FEC de decisión blanda no ayudaría.** El cociente BER/BER-top90 es
   1.0–1.2× sobre `code`, frente a 3.9× sobre `markov`: los errores están
   repartidos uniformemente, no concentrados en bits de baja confianza.
3. **La redundancia es explotable.** El oráculo alcanza BER exactamente 0 con la
   mitad de los bits.

### La explicación, corregida

La hipótesis inicial era que el descenso de gradiente no aprende XOR de grado
alto. **El diagnóstico la refutó:** con 500 bits de entrada y grado 3, un MLP
alcanza acc_test = 1.0000. Solo falla en grado 4 (acc_test 0.4995, con acc_train
1.0 — memoriza sin generalizar), y la fuente usa grado 3.

Conviene registrar que **un piloto anterior la había confirmado**: con 60 000
muestras y 25 épocas, el grado 3 daba acc_test = 0.5069, azar puro. La
diferencia con la corrida final son ocho veces más pasos de entrenamiento. Un
resultado negativo obtenido con presupuesto insuficiente es indistinguible de
uno real; el detalle está en la [bitácora](00-bitacora.md).

La diferencia entre el diagnóstico y el autoencoder es la **supervisión**. El
diagnóstico recibe una etiqueta que *es* la paridad, y el gradiente apunta
directamente a ella. El autoencoder solo tiene pérdida de reconstrucción, y debe
descubrir 250 funciones de paridad simultáneas como subproducto de comprimir.

**El obstáculo es el objetivo, no la capacidad ni la aprendibilidad.** Es una
conclusión más débil que la original pero mejor sustentada, y deja una vía
concreta de trabajo futuro: un objetivo auxiliar que supervise la estructura.

---

## El denoising abre el camino de gradiente, pero no sobre álgebra

Si el obstáculo en `code` es que la reconstrucción no genera gradiente hacia la
estructura, cambiar el objetivo debería resolverlo. El enmascarado de entrada
(Vincent et al., 2008) es la forma concreta: se tapan al azar una fracción *p*
de los 500 bits y se exige reconstruir los 500. Acertar un bit tapado obliga a
usar los demás, y en `code` eso es exactamente aprender el código.

**No funciona.** Sobre `code` L=250 el BER es plano: 0.2477 con *p*=0 y 0.2481
con *p*=0.50.

Lo que hace interpretable el resultado es una métrica añadida para el caso,
`ber_tapados`: el BER medido **solo en las posiciones corrompidas**, que mide
directamente si el modelo infiere los bits ocultos.

| fuente | L | `ber_tapados` | |
|---|---|---|---|
| `code` | 125 | 0.5000 ± 0.0005 | prácticamente azar |
| `code` | 250 | **0.5001 ± 0.0003** | prácticamente azar |
| `random` | 250 | 0.5000 ± 0.0004 | correcto, es imposible |
| `lowdim` | 70 | 0.1713 ± 0.0041 | infiere con fuerza |
| `lowdim` | 250 | 0.1129 ± 0.0044 | infiere con más fuerza |
| `markov` | 70 | 0.1197 ± 0.0014 | infiere con fuerza |
| `markov` | 250 | 0.0762 ± 0.0056 | infiere con más fuerza |

*(Todas las celdas salen de la misma corrida de denoising: media sobre las tres
*p* > 0 y las cuatro semillas. La rejilla usa dos escalones por fuente, y no son
los mismos en todas. La desviación es entre semillas; promediar los dos
escalones en una sola cifra la inflaría hasta siete veces, porque casi toda la
variación es entre escalones.)*

El gradiente por escalones dice algo por sí solo: en `lowdim` y `markov`, más
bits significan mejor inferencia de lo tapado. En `code` la cifra no se mueve.

**El mecanismo funciona; sobre estructura algebraica, apenas.** En `lowdim` y
`markov` el modelo aprende a rellenar lo que se le tapa. En `code` se queda al
borde del azar.

**Una precisión que el dato exige.** En el factorial posterior, con 64 celdas de
`code` y las dos inicializaciones, el valor es **0.4946 ± 0.0074**: está 5.9
desviaciones por debajo de 0.5, así que *no es azar exacto*. Y el desvío crece
con la tasa de forma monótona; a L=250 con *p*=0.10 llega a 0.4884, o sea que
recupera un **2.3 % del camino** hacia la inferencia perfecta. Comparado con
`random`, cuya desviación estándar es veinte veces menor, la diferencia es real.

Es una cifra pequeña, y rima con el 21 % del camino que el autoencoder recorre
en BER sobre esa misma fuente. La afirmación defendible es *«recupera en torno
al 2 % de la estructura disponible»*, no *«no infiere nada»*: esta última es
falsable con tres líneas de pandas sobre el CSV publicado.

Como regularizador sí sirve en un punto: `lowdim` L=250 con *p*=0.10 da
**0.0346**, mejor que el preentrenamiento RBM (0.0370) y que la inicialización
aleatoria (0.0386). No cambia ningún veredicto —ese punto ya ganaba— pero
amplía su margen.

### El enmascarado ayuda en `code`, pero no por el mecanismo de Vincent

En `code` L=125 el BER baja de forma **monótona con el ruido**, y gana en 4/4
semillas en las tres *p*:

| *p* | BER |
|---|---|
| 0.00 | 0.34397 ± 0.00030 |
| 0.10 | 0.34331 ± 0.00015 |
| 0.25 | 0.34191 ± 0.00009 |
| **0.50** | **0.33987 ± 0.00017** |

La mejora es de 0.0041, unas **24 desviaciones**. Refuta P4 («*p*=0.50 empeora
en todas las fuentes») mucho más directamente que el caso de `lowdim` L=70.

Lo relevante es el mecanismo: mejora **sin que `ber_tapados` se mueva de 0.495**.
El enmascarado actúa aquí como regularizador puro, no por la vía que predice la
hipótesis de Vincent. Matiza la frase anterior: sobre álgebra el enmascarado
**sí produce un efecto, solo que no el que se esperaba**. Sigue perdiendo contra
la vara (0.3350), así que no cambia ningún veredicto.

*(Mismo comentario para `lowdim` L=70, donde *p*=0.50 gana 4/4 de 0.1376 a
0.1327 mientras *p*=0.10 y *p*=0.25 pierden 0/4. Con sd de 0.0022 en el control,
esa victoria es de 2 a 3 sigmas: más frágil de lo que sugiere el 4/4.)*

### Limitación: sobre `code` el arnés rinde por debajo del mejor conocido

El factorial usa la **escalera estrecha**, y sobre `code` rinde peor que el MLP
ancho del barrido v4:

| | `code` L=250 |
|---|---|
| copiar 250 bits y adivinar el resto | 0.2500 |
| **factorial, escalera estrecha, *p*=0** | **0.2477** |
| barrido v4, MLP ancho | 0.1982 |

El valor del factorial está prácticamente en la solución trivial. Y el mejor
checkpoint llega en el paso **~1 500 de 30 000** sobre `code` y `random`, frente
a ~27 750 sobre `lowdim` y `markov`: el modelo alcanza la solución trivial
pronto y se degrada después. Entrenó los 30 000 pasos —no se cortó— pero dejó de
mejorar en el 5 % inicial.

**La línea plana de `code` está medida en un régimen donde esa arquitectura no
progresa**, y por debajo del mejor punto que el propio proyecto ya tenía para esa
celda. No creo que cambie el veredicto, porque `ber_tapados` es evidencia más
directa que el BER y apunta igual, pero es la primera objeción que un revisor
plantearía y debe constar.

### Preentrenamiento y ruido compiten

Factorial 2×3 completo (inicialización aleatoria o RBM × *p* ∈ {0, 0.10, 0.25}),
13 puntos, 4 semillas, 312 celdas. Hipótesis primaria registrada de antemano
sobre `markov` L=250; el resto declarado exploratorio.

| inicialización | *p* = 0 | *p* = 0.10 | *p* = 0.25 |
|---|---|---|---|
| aleatoria | 0.0135 ± 0.0004 | 0.0140 ± 0.0002 | 0.0207 ± 0.0006 |
| **RBM** | **0.0107 ± 0.0001** | 0.0116 ± 0.0002 | 0.0168 ± 0.0004 |

Ninguna condición con ruido mejora sobre RBM solo, en **0 de 4 semillas en las
seis celdas**. El efecto del RBM es casi constante (−0.0028) y el del ruido
crece con *p*.

**La interacción es positiva en 11 de 13 puntos** (mediana +0.0018): combinar da
peor que la suma de los efectos por separado. La lectura mecánica es que el
preentrenamiento coloca los pesos en una configuración estructurada y el
enmascarado la perturba — **construir y perturbar compiten**. El caso extremo es
`lowdim` L=35, con interacción +0.0120.

Incluso donde el ruido es lo mejor que hay (`lowdim` L=250, 0.0346), combinarlo
con RBM da 0.0347: ni suma ni resta. Cada mecanismo funciona por separado y se
estorban juntos.

**Lo que no se probó.** Solo se ejecutó el modo `mask`; no se ejecutó `flip`
(voltear bits en vez de borrarlos). Sobre `code`, `flip` es la corrupción
teóricamente más adecuada: con `mask` el modelo sabe *qué* posiciones faltan,
mientras que con `flip` no lo sabe, y corregir volteos usando los checks es
literalmente **decodificación por síndrome** — para lo que existe un código de
bloque. Es la prueba más adversa que se le puede hacer a esta conclusión, está
implementada y sin correr. Tampoco se incluyó `oversamp` en la rejilla.

---

## Hallazgos metodológicos

Probablemente lo más transferible del proyecto.

### Inestabilidad de escala en latentes binarios

Con `sign()` en el paso hacia adelante, escalar las preactivaciones no cambia la
salida: **la pérdida es ciega a la escala**. Pero el paso hacia atrás no lo es.
Existe una dirección degenerada por la que los pesos crecen sin penalización
mientras van anulando su propio gradiente.

Observado: el modelo alcanza BER 0.0003 y noventa épocas después está en 0.4725
con el 47 % de los bits muertos. Confirmado por medición — `pre_max` final
correlaciona monótonamente con el BER final:

| normalización | pre_max | BER final |
|---|---|---|
| BatchNorm sin afín | **3.8** | **0.0000** |
| LayerNorm sin afín | 13.3 | 0.0640 |
| tanh | 16.9 | 0.1706 |
| LayerNorm con afín | 28.6 | 0.1167 |

`BatchNorm1d(affine=False)` lo controla porque normaliza **por dimensión sobre
el batch**, aplicando realimentación negativa a cada bit por separado. LayerNorm
normaliza cada muestra contra sus propias dimensiones y no controla ninguna en
particular. Aun así, **1 de 4 semillas divergió**: la selección de checkpoint
por validación es necesaria, no opcional.

### La redundancia del latente es capacidad excedente

Medida con un modelo autoregresivo (entropía cruzada de test como cota superior,
verificada contra la cota de Fano):

| fuente | L | L / H_fuente | redundancia |
|---|---|---|---|
| oversamp | 250 | 2.00 | 36.1 % |
| markov | 250 | 1.74 | 33.8 % |
| markov | 70 | 0.49 | 16.3 % |
| lowdim | 70 | 0.42 | 3.6 % |
| lowdim | 35 | 0.21 | **1.0 %** |

Relación monótona: cuando sobran bits, el autoencoder los desperdicia; cuando
faltan, el latente sale casi incompresible. **A tasas bajas —donde el
autoencoder gana— no hay nada que ganar por codificación entrópica.**

Excepción informativa: `markov` L=70 tiene 16.3 % frente al 3.6 % de `lowdim`
L=70, con menos capacidad relativa. El autoencoder codifica estructura
geométrica más eficientemente que correlacional.

### Modos de entrenamiento

Anidar sale **esencialmente gratis** (costo mediano +0.0011 BER) y produce un
códec compatible en tasa: un modelo con cuatro puntos de operación. El
modo `stacked` de v4 —una cascada de compresores binarios profundos— es
**consistentemente el peor y el menos reproducible** (|Δ| máxima entre semillas
de 0.079). Ese modo **no era** la receta de Hinton y Salakhutdinov; la
reproducción fiel sí ayuda (ver «El preentrenamiento por capas sí ayuda»).

### Los datos están saturados

Con **pasos fijos** (30 000 en todas las corridas, para no confundir el efecto
de los datos con el de la optimización):

| n_train | BER |
|---|---|
| 200k | 0.1106 |
| 400k | 0.1053 |
| 800k | 0.1056 |
| 1.6M | 0.1040 |

Plano desde 400k. Un análisis previo con épocas fijas sugería una mejora del
14.5 % al duplicar los datos; con pasos fijos se ve que venía de **más
actualizaciones**, no de más diversidad.

---

## Resultados clásicos verificados

Independientes del arnés de autoencoders: NumPy puro, semillas fijas,
regenerables en segundos con `python3 code/generar_tablas.py`.

### Cotas teóricas

| Fuente | H (bits) | H/n | L=35 | L=70 | L=125 | L=250 |
|---|---|---|---|---|---|---|
| `random` | 500.0 | 1.000 | 0.3455 | 0.2834 | 0.2145 | 0.1100 |
| `code` | 250.0 | 0.500 | 0.0881 | 0.0684 | 0.0417 | **0** |
| `lowdim` | 164.9 | 0.330 | 0.0439 | 0.0291 | 0.0099 | **0** |
| `markov` | 143.9 | 0.288 | 0.0348 | 0.0211 | 0.0040 | **0** |
| `oversamp` | 125.0 | 0.250 | 0.0272 | 0.0146 | **0** | **0** |

### PCA es ciego a la redundancia algebraica

Sobre **10 semillas**, media ± desviación estándar:

| Latente | `code` | `random` | diferencia |
|---|---|---|---|
| 35 bits | 0.4160 ± 0.0006 | 0.4161 ± 0.0005 | −0.0000 ± 0.0008 |
| 70 bits | 0.3795 ± 0.0004 | 0.3794 ± 0.0006 | +0.0001 ± 0.0008 |
| 125 bits | 0.3350 ± 0.0004 | 0.3352 ± 0.0006 | −0.0003 ± 0.0007 |
| 250 bits | 0.2526 ± 0.0004 | 0.2530 ± 0.0005 | −0.0003 ± 0.0006 |

Dentro de 1σ en los cuatro escalones. Datos en
[`data/pca_code_vs_random.csv`](../data/pca_code_vs_random.csv), procedencia en
[`data/procedencia.json`](../data/procedencia.json).

**Medido con el cuantizador antiguo.** Esta tabla se calculó con la
cuantización uniforme min/max, antes del recálculo con Lloyd-Max, y no se
rehízo. La conclusión no debería cambiar, porque es una comparación entre
`code` y `random` con el mismo cuantizador en los dos lados: el sesgo se
cancela. Queda como pendiente barato, no como duda abierta.

![Zona de viabilidad por fuente](../figs/zona_viabilidad.png)

---

## Cuestiones de medición que cambiaron conclusiones

| Corrección | Por qué importaba |
|---|---|
| Latente binario con straight-through | Un latente `float32` de 50 dimensiones ocupa 1600 bits: reportar "compresión 10×" habría sido falso |
| Entropía de `lowdim` por conteo de Cover | Usar la dimensión del manifold (32) en lugar de 164.9 habría puesto la cota en el lugar equivocado |
| Decimación con *hold* además de vecino | Sobre `oversamp` a 125 bits, hold da 0.0000 y vecino 0.1241: usar solo vecino subestimaba la vara |
| Modo fijo al promediar semillas | Elegir el mejor modo por semilla es selección posterior: infló una victoria inexistente |
| Pasos fijos al escalar datos | Con épocas fijas, más datos son también más actualizaciones |
| Cota de Fano como verificación | Detectó que el primer estimador de entropía daba resultados imposibles |
| Control de monotonía | Más bits siempre debe dar menos BER; las violaciones delatan divergencia |

La decimación **no es monótona en la tasa** sobre `oversamp`: m=4 (tasa 0.250)
da BER 0 y m=3 (tasa 0.334) da 0.083, porque m=4 se alinea con el período de
repetición. Un detalle a reportar si se usa decimación como vara.
