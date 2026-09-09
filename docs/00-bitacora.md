---
layout: default
title: Bitácora — cómo llegamos hasta aquí
---

# Bitácora: cómo llegamos hasta aquí

Esta página es el registro cronológico del proyecto. Cada sección dice **qué
creíamos**, **qué lo rompió** y **qué cambiamos**. Si algo del resto del sitio
no cuadra, probablemente la explicación esté aquí.

Leer de arriba abajo reconstruye todo el razonamiento.

---

## Punto de partida

**El objetivo original.** Un vector de 500 símbolos BPSK (±1). Comprimirlo lo
más posible con un encoder y reconstruirlo con un decoder preservando signos y
posiciones. Usar una métrica de telecomunicaciones para decidir si es viable.

**Primera decisión.** La métrica es el **BER** (bit error rate): fracción de
posiciones donde el signo reconstruido no coincide con el original. Es la medida
natural del campo y no hay que inventar nada.

---

## Etapa 1 — El primer arnés

Se construyó un barrido simple: dos tipos de datos (bits aleatorios y una señal
con estructura de baja dimensión), dos arquitecturas (profunda y lineal), y
varios tamaños de latente.

**Resultado.** Con bits aleatorios el BER subía hacia 0.5 al comprimir, y con
una brecha enorme entre entrenamiento y prueba. Con datos estructurados,
compresión 10× con ~5 % de BER.

**Lo aprendido.** *La compresibilidad no es una propiedad del autoencoder sino
de la señal.* Bits i.i.d. no se pueden comprimir por debajo de su entropía —
teorema de codificación de fuente de Shannon — y ninguna arquitectura lo
cambia. Ese fue el hallazgo que orientó todo lo demás.

---

## Etapa 2 — El autoengaño de la compresión

**Qué se rompió.** Un latente de 50 dimensiones en `float32` ocupa
50 × 32 = **1600 bits**. Los datos originales son 500 bits. Es decir: el
"comprimido" era **más grande que el original**, y reportar "compresión 10×"
habría sido falso.

**Qué cambiamos.** Latente **binario** con estimador straight-through
(Bengio et al., 2013). Así la tasa se cuenta en bits reales por construcción:

$$R = \frac{\text{bits del latente}}{500}$$

**Lo aprendido.** Cualquier afirmación de compresión tiene que ser contable en
bits. Un latente continuo no comprime, solo cambia de representación.

---

## Etapa 3 — Añadir la vara y el piso

Hasta aquí solo teníamos BER. Faltaban las dos referencias que hacen que "sirve"
sea falsable.

**El piso.** La cota inferior de Shannon para fuentes binarias con distorsión de
Hamming. Ningún método puede bajar de ahí; si un experimento la viola, tiene un
bug.

**La vara.** Baselines clásicos con el **mismo número de bits**: PCA con
cuantización, decimación, y oráculos que conocen la estructura. Si el
autoencoder no le gana a PCA, no aporta nada.

**Se añadieron fuentes.** Ya no una o dos, sino cuatro tipos de estructura:
aleatoria, geométrica, algebraica (código de bloque) y correlacional (Markov).

**Lo aprendido.** Sin baselines, un BER de 0.05 no significa nada: puede ser
excelente o ridículo según lo que consiga un método trivial a la misma tasa.

---

## Etapa 4 — Correcciones al arnés

Al revisar el código aparecieron cinco problemas concretos:

| Problema | Por qué importaba |
|---|---|
| PCA recalculaba la descomposición hasta 18 veces por fuente | decenas de minutos desperdiciados en el modo completo |
| Decimación solo por vecino más cercano | sobre señales sobremuestreadas **subestimaba la vara** |
| RNG del canal sin semilla | rompía la reproducibilidad |
| Cota i.i.d. dibujada sobre fuentes estructuradas | es referencia, no cota operativa: confunde |
| El "AE lineal" no era PCA | tenía latente binarizado; el nombre inducía a error |

**El de la decimación fue el grave.** Sobre una señal repetida 4 veces, un
*sample-and-hold* trivial da BER **0.0000** a 125 bits, mientras que reconstruir
por vecino más cercano da 0.1241. Usar solo vecino habría hecho parecer ganador
a un autoencoder que en realidad empata o pierde.

**Lo aprendido.** La vara tiene que ser el **mejor** método clásico, no un
método clásico cualquiera. Una vara mal puesta produce victorias falsas.

---

## Etapa 5 — La escalera anidada

Se decidió comprimir por pasos: 250 → 125 → 70 → 35 bits.

**Descubrimiento útil.** Esos escalones **cortan las entropías** de las fuentes.
A 250 bits la compresión sin pérdida es teóricamente posible para cuatro de las
cinco fuentes; a 125 y por debajo, deja de serlo. La escalera no era arbitraria.

**Dos interpretaciones distintas del "por pasos".** Conviene no confundirlas:

- **Apilado** (`stacked`) — es una *técnica de entrenamiento*: entrenar
  500→250, congelar, luego 250→125, etc. Es la receta de Hinton y Salakhutdinov
  (2006).
- **Anidado** (`nested`) — es una *arquitectura*: un solo modelo donde los
  primeros 35 bits ya son decodificables, los primeros 70 refinan, etc. Esto es
  refinamiento sucesivo (Equitz y Cover, 1991) y en telecomunicaciones equivale
  a un códec compatible en tasa.

Se implementaron ambas, más `direct` (un modelo por escalón) como control.

**Advertencia registrada de antemano.** La teoría predice que anidar puede
costar rendimiento frente a un modelo dedicado. Por eso se compara `nested` con
`direct` en lugar de suponer que es gratis.

---

## Etapa 6 — La trampa del latente grande

**Qué se detectó.** "Reconstruir lo mejor posible" tiene una solución trivial:
con latente de 500 bits el autoencoder aprende la identidad y da BER 0. No
comprimió nada.

**Qué cambiamos.** Nada en el código: cambió el criterio de éxito. La afirmación
de viabilidad es siempre un **par (tasa, BER) contra la vara**, nunca un BER
suelto.

---

## Etapa 7 — Primera corrida completa

Se lanzó el barrido en GPU: 5 fuentes × 3 modos × 4 escalones = 60
configuraciones.

**Dos cosas salieron mal.**

**(a) El verificador reportó 120 filas en vez de 60.** No se corrió dos veces:
ambos shards escribían los baselines de las mismas fuentes, porque cada proceso
cargaba la lista de "ya hecho" una sola vez al arrancar y ambos la veían vacía.
Al cruzar resultados con baselines duplicados, el `merge` producía el doble de
filas. Los BER eran correctos; solo se mostraban duplicados.

**(b) Un valor imposible.** `oversamp`, modo `direct`, latente de 250 bits →
**BER 0.5000**.

Esa fuente son 125 símbolos repetidos 4 veces. Con 250 bits el autoencoder tiene
el **doble** de los bits necesarios; la respuesta correcta es BER ≈ 0. Obtuvo
azar puro, y peor que el mismo modo con la mitad de bits. Un 0.5000 exacto
significa salida constante: el latente se congeló.

**Lo aprendido, y es lo más importante del proyecto.** Mientras el arnés no
logre copiar una señal repetida teniendo bits de sobra, **no se puede
distinguir** entre "el autoencoder no encuentra la estructura" y "nuestra
implementación no optimiza". Los 17 de 20 casos donde el autoencoder perdía
contra PCA no medían autoencoders: medían un bug.

---

## Etapa 8 — Calibrar antes de medir

**La analogía que ordena todo.** Si una balanza indica 0.5 kg para una pesa
patrón de 1 kg, la conclusión no es que la gravedad cambió. Se arregla la
balanza.

**El test de calibración.** Entrenar `oversamp` con latente de 250 bits, donde
ya sabemos que la respuesta correcta es BER ≈ 0, y comparar dos variantes del
encoder.

**Hipótesis del fallo.** El encoder aplicaba `tanh` a preactivaciones sin
normalizar antes de binarizar. Con capas de 1536 unidades esas preactivaciones
alcanzan magnitudes donde la tangente hiperbólica se satura, su derivada se
anula, y el gradiente muere antes de llegar al encoder. Encaja con el patrón: el
fallo aparece **específicamente en el latente más grande**, que es donde las
preactivaciones crecen más.

**Corrección propuesta.** Normalizar antes de binarizar (Ba et al., 2016). El
recorte del straight-through ya acota el gradiente sin necesidad de la
tangente hiperbólica.

**Desenlace.** La hipótesis de la saturación resultó **falsa**, pero el test
destapó el modo de fallo real: la inestabilidad de escala del latente binario,
con su mecanismo confirmado por medición. El detalle está en el registro
fechado, más abajo.

---

## Qué está firme y qué no

**Firme** (NumPy puro, semillas fijas, sin redes neuronales):

- Las cotas teóricas de las cinco fuentes.
- Los baselines clásicos en los cuatro escalones.
- Que PCA es **ciego a la redundancia algebraica**: su BER sobre un código de
  bloque 2× comprimible es indistinguible de su BER sobre ruido puro.

**Firme tras el cierre:** el barrido de autoencoders, con calibración aprobada
y cuatro semillas; la correspondencia arquitectura–estructura; y la
inestabilidad de escala con su mecanismo medido.

**Abierto:** las extensiones de [Hacia un paper](07-hacia-paper.md), encabezadas
por un objetivo auxiliar que supervise la estructura algebraica.

---

## Hilo conductor

Si hubiera que resumir el proyecto en una frase por etapa:

1. La compresibilidad depende de la señal, no del autoencoder.
2. Un latente continuo no comprime; hay que contar bits.
3. Sin cota y sin vara, un BER no significa nada.
4. La vara tiene que ser el mejor método clásico, no cualquiera.
5. Anidar da un códec escalable, pero puede costar.
6. Bajar el BER subiendo el latente es trivial y no prueba nada.
7. Un resultado imposible invalida toda la corrida.
8. Calibrar el instrumento antes de creerle la medición.
9. Con `sign()` en el forward, la pérdida es ciega a la escala: los pesos crecen
   sin castigo mientras matan su propio gradiente.
10. Sin barras de error no se distingue una victoria de un accidente — pero
    tampoco se puede declarar ruido lo que no se ha medido.
11. La redundancia del latente es capacidad excedente, no codificación.
12. La estructura algebraica es aprendible con supervisión; lo que falla es que
    la reconstrucción no lleve hasta ella.
13. Un resultado negativo con presupuesto corto es indistinguible de uno real:
    hay que volver a medirlo con margen antes de creerlo.
14. Una conclusión sobre «los autoencoders» puede ser sobre una arquitectura.
    La frontera en R ≈ 0.25 lo era.
15. Etiquetar un experimento con el nombre de un paper obliga a reproducirlo.
    Una cascada de compresores no es Hinton 2006.

---

## Anexo: las versiones previas (v1, v2 y v3)

Antes del arnés definitivo hubo tres versiones, todas en CPU y a escala
pequeña. Se conservan porque **dos de sus conclusiones fueron revertidas
después**, y en ambos casos se puede identificar la causa. Un piloto que acierta
no enseña nada; uno que se equivoca de forma diagnosticable, sí.

### v1 — el primer arnés, con latente continuo

Dos fuentes (`random`, `lowdim`), dos arquitecturas, latente **continuo** en
`float32`, pérdida MSE, 4 000 muestras y 100 épocas en CPU.

**Lo que estableció y orientó todo el proyecto:** que la compresibilidad es
propiedad de la señal. Sobre `random` el BER de test sube hacia 0.5 al comprimir
y aparece una brecha enorme entre entrenamiento y prueba (L=200: train 0.0663,
test 0.2639) — la red memoriza pero no generaliza, que es la firma de una fuente
sin estructura explotable. Sobre `lowdim` el BER colapsa por debajo de L≈32, la
dimensión intrínseca del generador.

**Error de encuadre, corregido en la misma sesión.** El primer resumen decía
«`lowdim` 500 → 50, compresión 10× con ~5 % de BER». Es falso: 50 dimensiones en
`float32` son **1600 bits** contra 500 de fuente, un factor de 0.31× — una
*expansión*. El análisis de presupuesto de bits lo dejó claro cuantizando el
latente:

| bits/dim | bits totales | factor vs 500 | BER test |
|---|---|---|---|
| 32 (float) | 1600 | 0.31× (expansión) | 0.054 |
| 8 | 400 | 1.25× | 0.055 |
| 4 | 200 | 2.50× | 0.061 |
| 2 | 100 | 5.00× | 0.149 |
| 1 | 50 | 10.00× | 0.181 |

De aquí sale la decisión de usar latente binario en todo lo que vino después.

**Primera aparición de una conclusión que resultó falsa dos veces.** En v1 el
autoencoder lineal gana al profundo en **10 de 10** configuraciones. Se leyó
como confirmación de Baldi y Hornik. Lo notable es que aquí el latente era
*continuo*, sin estimador straight-through, así que **la causa no puede ser la
inestabilidad de escala** que explicó el mismo fenómeno en v2. Son dos orígenes
distintos para la misma conclusión equivocada, y ninguno sobrevivió al arnés
corregido.

En v1 la explicación probable es más simple: 4 000 muestras y un modelo profundo
que no llega a ajustar una fuente casi lineal.

**Interpretación corregida de la meseta.** Se dijo que el autoencoder «encuentra
la dimensión intrínseca» porque su BER quedaba plano entre L=200 y L=50
(0.0524, 0.0523, 0.0520). Pero el modelo lineal **no** hace meseta en ese rango:
sigue bajando hasta 0.0186. Si la meseta fuera la dimensión intrínseca, ambos la
tendrían. Era un piso de optimización del modelo profundo. Lo que sí es real es
el colapso por debajo de L≈32, donde la información se pierde de verdad.

### v2 — ablación, barrido binario y primer intento de JSCC

**Lo que estableció y sobrevivió:** el coste real del latente continuo. Con 128
dimensiones en `float32` el latente ocupa **4096 bits** para 500 bits de fuente
— una *expansión* de 0.12×, no una compresión. El BER era mejor (0.039 frente a
0.111 en `lowdim`) precisamente porque estaba gastando 32 veces más bits. Esa
tabla es el origen de la decisión de usar latente binario en todo el proyecto.

| fuente | latente | bits reales | compresión | BER test |
|---|---|---|---|---|
| lowdim | continuo (128 dim) | 4096 | 0.12× | 0.0390 |
| lowdim | binario (128 bits) | 128 | 3.91× | 0.1107 |
| code | continuo (128 dim) | 4096 | 0.12× | 0.3003 |
| code | binario (128 bits) | 128 | 3.91× | 0.3660 |

**Conclusión revertida: «el autoencoder lineal le gana al profundo».** En v2 el
lineal ganaba en **15 de 16** configuraciones, a veces por mucho (`markov`
L=250: 0.0457 frente a 0.1035). Se interpretó como confirmación de Baldi y
Hornik.

Era un **artefacto de la inestabilidad de escala**, diagnosticada mucho después.
El modelo profundo de v2 usaba `tanh` antes de binarizar, exactamente la
configuración que en `gate_test3` diverge; el lineal, con muchos menos
parámetros, sufría menos. Con el encoder corregido el profundo gana a ambos:

| caso | v2 profundo | v2 lineal | v4 corregido |
|---|---|---|---|
| markov L=250 | 0.1035 | 0.0457 | **0.0344** |
| markov L=125 | 0.1224 | 0.0797 | **0.0503** |
| lowdim L=250 | 0.0840 | 0.0718 | **0.0510** |
| lowdim L=125 | 0.1116 | 0.1047 | **0.0618** |

La comparación está confundida —v4 tiene además más datos, más capas y más
épocas—, pero el control limpio existe: en `gate_test3`, a igual escala, `tanh`
diverge y `BatchNorm1d(affine=False)` no.

**Primer JSCC, fallido.** Las curvas de BER frente a Eb/N0 salieron planas
(`markov` L=128 va de 0.1371 a 0.1234 entre −2 y 10 dB). El piso de distorsión
de la compresión tapaba por completo el efecto del canal. De ahí salió la
corrección de reportar el **exceso** sobre el piso sin ruido en lugar del BER
total.

### v3 — baselines clásicos, cotas y diagnóstico de paridad

**Lo que estableció y sobrevivió:** que hacen falta una vara y un piso. Aquí
aparecieron por primera vez PCA con cuantización, la decimación, los oráculos y
la cota inferior de Shannon. También el oráculo de `code` alcanzando **BER
0.0000 a tasa 0.500**, que es el pilar del resultado negativo.

**Conclusión revertida: «un MLP no aprende paridad de grado ≥ 3».**

| grado | v3 piloto (acc_test) | cierre final (acc_test) |
|---|---|---|
| 1 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 |
| **3** | **0.5069 — azar** | **1.0000 — aprende** |
| 4 | 0.4931 — azar | 0.4995 — azar |

Grado 3 es exactamente el de los checks del código, así que este resultado se
tomó como la explicación del fracaso sobre `code`. La figura de v3 lo dice en su
propio título: *«Un MLP no aprende paridad de grado alto (por eso el AE no ve la
redundancia algebraica)»*.

La diferencia entre las dos corridas es **presupuesto de entrenamiento**:

| | v3 piloto | cierre final |
|---|---|---|
| muestras | 60 000 | 400 000 |
| épocas | 25 | 60 |
| pasos aproximados | 2 925 | 23 400 |

Con ocho veces más pasos, grado 3 pasa de azar a exactitud perfecta. **El
"fracaso" era subentrenamiento, no una barrera de aprendibilidad.** Y la
consecuencia va más allá de este punto: un resultado negativo obtenido con
presupuesto insuficiente es indistinguible de uno real, y solo se detecta
volviéndolo a medir con margen.

### Qué se llevó de aquí al arnés definitivo

- Latente binario con estimador straight-through (del presupuesto de bits de
  v1, confirmado por la ablación de v2).
- Baselines clásicos y cotas de Shannon como referencias obligatorias (de v3).
- Oráculos como demostración de que la redundancia es explotable (de v3).
- Reportar el exceso sobre el piso en JSCC, no el BER total (del fallo de v2).
- Y, con el diario, dos recordatorios: que una conclusión puede ser artefacto de
  una inestabilidad no diagnosticada, y que un resultado negativo puede ser
  artefacto de un presupuesto corto.

---

## Registro de corridas

A partir de aquí, cada corrida se anota con fecha. Sin cronología no se puede
reconstruir el razonamiento después, y es lo que alimenta la sección de
limitaciones si esto llega a paper.

**Formato:** fecha · qué se ejecutó · qué salió · qué se decidió.

### 2026-08-23

- **Ejecutado:** `verificar_robustez.py --seeds 10` sobre los baselines clásicos.
- **Resultado:** la diferencia PCA entre `code` y `random` queda dentro de
  1 sigma en los cuatro escalones (máx. |dif| = 0.0003, sigma ≈ 0.0007).
- **Decidido:** el hallazgo pasa de observación a resultado con barras de error.
  Se adopta procedencia automática en `data/procedencia.json`.

### 2026-08-25 · Calibración del arnés

- **Ejecutado:** cuatro rondas de diagnóstico sobre `oversamp` L=250, donde la
  respuesta correcta es BER ≈ 0 porque al autoencoder le sobra el doble de bits.
- **Hipótesis 1 (saturación de la tangente hiperbólica): REFUTADA.** `|pre|>3`
  nunca superó el 5 % y estaba en 0 % al inicio, cuando el aprendizaje era más
  lento: la relación es inversa a la predicha. Cero bits muertos.
- **Descubierto un modo de fallo real:** el modelo alcanza BER 0.0003 en la
  época 60 y luego destruye su propia solución (época 150: BER 0.4725, 47 % de
  bits muertos). Mecanismo: `sign()` hace el paso hacia adelante invariante a la
  escala de las preactivaciones, así que la pérdida no penaliza que los pesos
  crezcan, pero el gradiente sí muere. Dirección degenerada.
- **Confirmado por medición:** `pre_max` final correlaciona monótonamente con el
  BER final — 3.8 → 0.0000; 13.3 → 0.0640; 16.9 → 0.1706; 28.6 → 0.1167.
- **Solución:** `BatchNorm1d(affine=False)` + `lr=1e-3`. Normaliza por dimensión
  sobre el batch, aplicando realimentación negativa a cada bit por separado;
  LayerNorm normaliza contra las otras dimensiones y no controla ninguna.
- **Decidido:** relanzar el barrido con esa configuración. Añadido control de
  monotonía al verificador (más bits siempre debe dar menos BER).

### 2026-08-25 · Barrido con cinco semillas

- **Calibración in situ aprobada:** `oversamp direct` L=125 da 3.2 × 10⁻⁷.
- **Reproducibilidad mucho mayor de lo previsto:** mediana de |Δ| entre semillas
  de 0.0005 en `direct` y 0.0007 en `nested`.
- **Seis configuraciones ganan en 4/4 semillas**, con márgenes de 4σ a 216σ.
- **Corrección metodológica:** en un análisis previo elegí el mejor modo por
  semilla antes de promediar, lo cual es selección posterior. Al fijar el modo,
  `markov nested L=70` pasa de aparente victoria a 0/4.
- **Inestabilidad residual:** 1 de 4 semillas divergió en `oversamp direct`
  L=250 (0.185 en vez de ~2 × 10⁻⁶). El mejor checkpoint por validación pasa a
  ser obligatorio.

### 2026-08-25 · Entropía del latente

- **Primer estimador refutado por cota de cordura.** La corrección de segundo
  orden `H ≈ H_marg − Σ I(i;j)` dio 0.0 bits para un latente que reconstruye una
  fuente de 125 bits: imposible. Solo vale para dependencias en árbol; con
  31 125 pares el sobreconteo crece como k² (verificado: 50 bits idénticos →
  estimación −1175 en lugar de 1).
- **Estimador correcto:** modelo autoregresivo sobre el latente, con la entropía
  cruzada de test como cota superior, verificada contra la cota de Fano.
- **Resultado:** la redundancia del latente es **capacidad excedente**, no
  codificación inteligente. Relación monótona entre `L/H_fuente` y redundancia:
  2.00 → 36.1 %, 1.74 → 33.8 %, 0.49 → 16.3 %, 0.42 → 3.6 %, 0.21 → 1.0 %.
- **Consecuencia:** a tasas bajas, donde el autoencoder gana, el latente es casi
  incompresible. La codificación entrópica no regrafica el mapa de viabilidad.

### 2026-08-25 · Cierre: tres experimentos finales

Los tres refutaron afirmaciones previas. Es lo que se esperaba de ellos.

- **Paridad: hipótesis principal REFUTADA.** Con 500 bits de entrada y grado 3
  —exactamente el de los checks del código— el MLP alcanza acc_test = 1.0000.
  El descenso de gradiente **sí** aprende XOR de grado 3. Solo falla en grado 4
  (acc_test 0.4995, con acc_train 1.0: memoriza sin generalizar).
- **Nueva explicación del fracaso en `code`:** la diferencia está en la
  supervisión. El diagnóstico tiene una etiqueta que *es* la paridad; el
  autoencoder solo tiene pérdida de reconstrucción y debe descubrir 250
  funciones de paridad simultáneas como subproducto de comprimir. El obstáculo
  es el objetivo, no la capacidad ni la aprendibilidad.
- **Cuantificación:** copiar 250 bits y adivinar el resto daría BER 0.2500; el
  autoencoder obtiene 0.1982; el oráculo 0.0000. Recorre el 21 % del camino.
- **Convolución: mi criterio de control estaba mal formulado.** Predije ~0 % de
  cambio en `lowdim`; salió −21.4 %. El error es mío: una convolución no es un
  superconjunto del MLP sino una arquitectura restringida, y sin localidad que
  explotar pierde la mezcla global. Que empeore es lo correcto.
- **Patrón limpio en las dos direcciones:** oversamp +10.3 %, markov +3.5 %,
  code +2.4 %, lowdim −21.4 %. La convolución ayuda donde hay estructura local y
  estorba donde es global. **Victoria nueva:** `oversamp` L=70 con conv (0.2271
  contra vara 0.2330).
- **Datos: saturado, y mi lectura previa era confusión de pasos.** Con pasos
  fijos (30 000 en todas las corridas), el BER va de 0.1106 a 0.1040 y se aplana
  desde 400k. El 14.5 % que atribuí a "más datos" venía de **más pasos**: con
  épocas fijas, duplicar los datos duplicaba las actualizaciones.

### 2026-09-09 · Preentrenamiento fiel y el confundido de arquitectura

Dos experimentos, dos predicciones refutadas, una conclusión publicada revisada.

- **`rbm_stack.py`** (RBM con CD-1 y AE por capa; escalera estrecha; 4 semillas;
  mejor checkpoint). Predicción registrada: no ayudaría. **Refutada:** gana en
  16/16 celdas de `lowdim` y `markov`, hasta 30 % a L=35. El `stacked` de v4,
  etiquetado como «la receta de Hinton», era una cascada de compresores
  binarios. Corregidas bibliografía y resultados.
- **RBM y AE no son equivalentes:** el AE gana a tasas bajas, la RBM a altas.
  Los preentrenados alcanzan su pico entre los pasos 750 y 6 000 y luego se
  degradan: el ajuste a `lr = 1e-3` daña la inicialización.
- **Hallazgo lateral:** la escalera **con inicialización aleatoria** ya daba
  0.0135 en `markov` L=250, contra 0.0345 de v4. Confundido: checkpoint o
  arquitectura.
- **`confundido.py`** (2×2, protocolo idéntico importado de `rbm_stack`).
  **Veredicto: arquitectura, 63–100 % de la brecha.** Checkpoint ~0.006;
  arquitectura 0.015.
- **Interacción arquitectura–tasa:** el ancho gana en L=35 por 0.04; la escalera
  en L ≥ 125 por ~0.015. La escalera es perfectamente estable (final − mejor =
  0.0000 en 16/16); el ancho se degrada hasta 0.009.
- **Conclusión publicada revisada:** «el AE no gana en R ≥ 0.25» era artefacto
  de una sola arquitectura. Con la escalera gana en `markov` L=125, L=250 y
  `lowdim` L=250.
- **Victoria rebajada:** `markov` L=35 con el ancho (4σ en v4) pierde con este
  protocolo. Con preentrenamiento es robusta.
- **`markov` L=250 con RBM: BER 0.0107.** A siete diezmilésimas del umbral FEC.

Ver [Hacia un paper](07-hacia-paper.md) para lo que queda abierto.

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
