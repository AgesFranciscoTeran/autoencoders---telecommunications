---
layout: default
title: Glosario
---

# Glosario

Todos los términos que aparecen en el proyecto, con la definición que se usa
aquí y dónde importa.

---

## Métricas y magnitudes

**BER** (*bit error rate*) — Fracción de posiciones donde el signo reconstruido
no coincide con el original: $$\mathrm{BER} = \frac{1}{n}\sum \mathbb{1}[\mathrm{sign}(\hat{x}_i) \neq x_i]$$.
Es la métrica central. BER = 0.5 equivale a adivinar al azar; BER = 0 es
reconstrucción perfecta.

**Tasa** ($$R$$) — Bits del latente dividido entre 500. Es el eje horizontal de
todas las figuras. $$R = 1$$ significa no comprimir; $$R = 0.07$$ (35 bits) es
compresión 14×.

**Compresión** — El inverso: $$500 / L$$. Un latente de 125 bits es compresión 4×.

**Entropía** ($$H$$) — Bits mínimos necesarios para representar la fuente sin
pérdida. Se calcula analíticamente para cada fuente, no se estima.

**Eb/N0** — Energía por bit dividida entre densidad de ruido. La medida estándar
de calidad de canal en telecomunicaciones. Aparece en la etapa JSCC.

---

## Conceptos de teoría de la información

**Cota inferior de Shannon (SLB)** — El BER mínimo físicamente alcanzable a una
tasa dada: $$D \geq H_b^{-1}(H/n - R)$$. **Ningún algoritmo puede bajar de ahí.**
Si un experimento la viola, tiene un bug — típicamente fuga entre entrenamiento
y prueba. Es la línea punteada negra de las figuras.

**Entropía binaria** ($$H_b$$) — $$H_b(p) = -p\log_2 p - (1-p)\log_2(1-p)$$. Se
invierte por bisección para despejar el BER de la cota.

**i.i.d.** — Independiente e idénticamente distribuido. Bits i.i.d. uniformes
son incompresibles: es el control negativo del experimento.

**Tasa-distorsión** — El marco donde la pregunta se vuelve precisa: no "¿cuánto
comprime?" sino "¿qué par (tasa, distorsión) alcanza?".

**Refinamiento sucesivo** — Cuándo una descripción gruesa puede refinarse hasta
el óptimo sin penalización. Es el marco teórico de la escalera anidada.

---

## Elementos del experimento

**Latente** — La representación comprimida que produce el encoder. Aquí es
siempre **binaria** (±1), de modo que su tamaño se cuenta en bits reales.

**Escalón** — Cada tamaño de latente probado: 35, 70, 125 y 250 bits. Los cuatro
juntos forman la *escalera*.

**Vara** (*baseline*) — Lo que consigue un método clásico usando el **mismo
número de bits**. Si el autoencoder no le gana, no aporta nada.

**Piso** — La cota de Shannon. Lo que nadie puede superar.

**Zona de viabilidad** — El margen entre la vara y el piso. Un autoencoder solo
es interesante si cae dentro.

**Oráculo** — Un método que **conoce la estructura de antemano** (la matriz de
paridad, la matriz de proyección). No es un competidor realista: es una cota
superior de lo alcanzable, que demuestra que la redundancia existe.

---

## Las cinco fuentes

**`random`** — Bits i.i.d. uniformes. $$H = 500$$ bits. Control negativo:
incompresible por Shannon.

**`oversamp`** — 125 símbolos, cada uno repetido 4 veces. $$H = 125$$ bits. Modela
una señal sobremuestreada antes de diezmar. Es el caso fácil, usado para
calibrar.

**`markov`** — Cadena de Markov con probabilidad de transición 0.05. $$H = 143.9$$
bits. Redundancia **correlacional**.

**`lowdim`** — $$\mathrm{sign}(Wz)$$ con $$z \in \mathbb{R}^{32}$$. $$H = 164.9$$ bits
(no 32 — ver *conteo de Cover*). Redundancia **geométrica**.

Es el modelo de medición de un bit `y = sign(Ax)` (Boufounos y Baraniuk, 2008),
no una construcción ad hoc.

**`code`** — Código lineal sistemático de tasa 1/2 sobre GF(2) con checks de
grado 3. $$H = 250$$ bits. Redundancia **algebraica**. El caso crítico.

**GF(2)** — El cuerpo de dos elementos. Aritmética módulo 2, donde la suma es el
XOR.

**Check de grado 3** — Cada bit de paridad es el XOR de 3 bits sistemáticos
(peso de columna 3 en `P`; peso de fila 4 en `H = [Pᵀ | I]`). La dependencia es
invisible no solo para métodos lineales sino también para los de **segundo
orden**: cualquier grado ≥ 2 anula las correlaciones por pares y deja la
covarianza en la identidad, así que PCA es ciego desde el grado 2. Lo que
distingue al 3 es que además queda fuera del alcance de una vara cuadrática, sin
salirse de lo que el gradiente aprende con supervisión. Ver
[Metodología](02-metodologia.md).

**Peso de columna / peso de fila** — Peso de columna 3 en `P`: cada paridad
depende de 3 sistemáticos. Peso de fila 4 en `H`: cada nodo de chequeo toca 3
sistemáticos más su paridad. Es la misma cosa dicha desde los dos lados; la
segunda es la convención en codificación.

**Conteo de Cover** — $$C(n,k) = 2\sum_{i<k}\binom{n-1}{i}$$, el número de
dicotomías linealmente separables. Da la entropía real de `lowdim`: 164.9 bits,
no los 32 que sugiere la dimensión del manifold.

**Manifold** — La superficie de baja dimensión donde vive la fuente `lowdim`.

---

## Modos de entrenamiento

**`direct`** — Un autoencoder independiente por escalón. Control: el mejor
rendimiento posible a cada tasa sin restricciones.

**`nested`** (anidado) — Un solo modelo donde **cualquier prefijo** del latente
es decodificable. Produce un códec compatible en tasa: si el canal se degrada,
transmites menos bits y el decoder sigue funcionando.

**`stacked`** (apilado) — Preentrenamiento voraz por capas: 500→250, congelar,
250→125, etc., con ajuste fino posterior. Receta de Hinton y Salakhutdinov.

---

## Técnicas

**STE** (*straight-through estimator*) — Truco para retropropagar a través de la
función signo, que no es diferenciable: hacia adelante aplica el signo, hacia
atrás deja pasar el gradiente. **Es lo que hace honesta la medición de
compresión.**

**LayerNorm** — Normalización que mantiene las preactivaciones en escala
unitaria. La corrección propuesta para el colapso: evita que la tangente
hiperbólica se sature y mate el gradiente.

**Saturación** — Cuando una función como `tanh` recibe entradas muy grandes, su
salida se pega a ±1 y su derivada se va a cero. El gradiente deja de fluir y la
red no aprende.

**PCA + cuantización** — El competidor justo: proyecta a $$d$$ componentes
principales y cuantiza cada una a $$b$$ bits, con $$d \cdot b = L$$. El cuantizador
es **Lloyd-Max**, el cuantizador escalar óptimo en MSE, ajustado por componente
sobre el conjunto de entrenamiento.

**Lloyd-Max** — Cuantizador escalar que minimiza el error cuadrático para una
distribución dada, alternando entre asignar cada valor a su centroide más cercano
y recolocar cada centroide en la media de los suyos. Reemplazó al rango min/max
uniforme, que dependía del tamaño de muestra.

**Decimación** — Enviar 1 de cada $$m$$ símbolos. Dos reconstrucciones:

- *hold* (sample-and-hold): repite el último valor transmitido.
- *vecino*: usa el transmitido más cercano.

La distinción importa: sobre `oversamp` a 125 bits, hold da 0.0000 y vecino
0.1241.

**JSCC** (*joint source-channel coding*) — Codificación conjunta de fuente y
canal: comprimir y proteger contra ruido en un solo paso.

**FEC** (*forward error correction*) — Código corrector de errores. Con LDPC o
turbo, un BER previo del orden de $$10^{-2}$$ se corrige hasta esencialmente cero.
Por eso $$10^{-2}$$ aparece como umbral en las figuras.

**LLR** (*log-likelihood ratio*) — Medida de confianza por bit. Los logits del
decoder funcionan como LLR: si los errores se concentran en bits de baja
confianza, un FEC de decisión blanda los limpia.

**BER en tapados** (`ber_tapados`) — BER medido **solo en las posiciones que se
corrompieron** durante una evaluación con enmascarado. Mide si el modelo infiere
los bits ocultos a partir de los visibles, que es el *mecanismo* que el denoising
afirma. Sobre `random` debe dar 0.5 exacto; un valor bajo significa que aprendió
estructura. Distingue «no ayuda» de «no aprende», cosa que el BER no hace.

**BER top90** — BER restringido al 90 % de los bits más confiables. Un cociente
BER/BER-top90 muy superior a 1 es el argumento de viabilidad con FEC.

---

## Mecánica de ejecución

**Shard** — Trozo del barrido asignado a una GPU. Con 2 GPUs se lanzan 2 shards
que se reparten las configuraciones y corren en paralelo.

**`./res 0,1`** — Los dos argumentos del lanzador: `./res` es la carpeta de
salida, `0,1` es la lista de GPUs **lógicas** (las que ve el proceso, que pueden
no coincidir con las físicas si hay remapeo).

**Smoke test** — Corrida mínima en CPU para verificar que el código arranca
antes de gastar GPU. Solo prueba que *corre*, no que esté correcto.

**Gate test** (test de calibración) — Corrida sobre un caso de **respuesta
conocida**. Prueba que el arnés *mide bien*. Es lo que el smoke test no hace.

**JSONL** — Un objeto JSON por línea. Formato de los resultados; permite añadir
filas sin releer el archivo, y hace el barrido reanudable.

**Reanudable** — Si un shard falla, al relanzar solo se ejecuta lo que falta,
porque cada configuración tiene una clave única ya registrada.

**bf16** — Formato de punto flotante de 16 bits nativo en GPUs Hopper. Acelera
el entrenamiento sin pérdida apreciable de precisión.

---

## Trampas encontradas

**Latente float32** — 50 dimensiones × 32 bits = 1600 bits > 500 originales.
Reportar "compresión 10×" habría sido falso.

**Vara subestimada** — Usar solo decimación por vecino hacía parecer ganador a
un autoencoder que empataba.

**Merge duplicado** — Ambos shards escribían los mismos baselines; el cruce
producía el doble de filas.

**BER 0.5000 exacto** — Salida constante, latente congelado. No es "mal
resultado": es **resultado imposible**, y por tanto un bug.

**Media contaminada** — El costo medio de anidar salía −0.037 por culpa de una
corrida colapsada; la mediana daba −0.0006. Con outliers, usar mediana.

**Artefacto obsoleto sobreviviendo a un crash** — `combinado.json` se escribe
*después* del resumen y `combinado.csv` *antes*. Cuando el resumen reventó, el
JSON quedó con las 78 filas del humo de un día antes, y así se subió al
repositorio junto a los datos buenos. Un fichero que no se reescribe no avisa de
que está obsoleto: comparar fechas de modificación entre artefactos de la misma
corrida lo detecta en un vistazo.

**Absolutos que el CSV refuta** — Escribir «azar exacto» cuando el dato es
0.4946 ± 0.0074 sobre 64 celdas, que está a 5.9 sd de 0.5. La afirmación
correcta («recupera ~2 % de la estructura disponible») es más débil y más útil.
Si publicas los datos, cada absoluto del texto es verificable con tres líneas de
pandas.

**Salvaguarda manual entre corridas** — La prueba de humo y la corrida real
escribían en el mismo directorio, con un `rm -rf` manual entre ambas como única
protección. Falló: 78 celdas de humo (600 pasos en vez de 30 000) entraron en el
conjunto real y movieron la hipótesis primaria de 0.0107 ± 0.0001 a
0.0203 ± 0.0191. La corrección es estructural, no disciplinaria: que el modo de
humo escriba en otro sitio por construcción.

**Comparar contra el parcial** — Lo anterior solo se detectó porque existía un
análisis intermedio con el que contrastar. Sin él, el valor contaminado habría
pasado como resultado del mejor punto del proyecto. Guardar y mirar parciales no
es curiosidad: es control de calidad.

**Vara dependiente del tamaño de muestra** — La más cara del proyecto. El
cuantizador de PCA usaba el rango min/max de las proyecciones de entrenamiento;
ese rango crece con *n*, así que la vara empeoraba cuantos más datos se le daban.
Produjo dos valores distintos para el mismo punto (0.1776 con 20 000 muestras,
0.1861 con 400 000) y, sobre todo, **infló los márgenes**: con el cuantizador
óptimo la vara de `lowdim` L=70 baja a 0.0977 y la victoria de 216 σ se convierte
en derrota. Si un baseline mejora al empeorar sus datos, no es un baseline.

**Etiqueta de paper sin reproducción** — Llamar «receta de Hinton» a una cascada
de compresores binarios. Al reproducirlo de verdad, el resultado se invirtió.

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
