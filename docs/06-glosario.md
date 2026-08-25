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
no coincide con el original: $\mathrm{BER} = \frac{1}{n}\sum \mathbb{1}[\mathrm{sign}(\hat{x}_i) \neq x_i]$.
Es la métrica central. BER = 0.5 equivale a adivinar al azar; BER = 0 es
reconstrucción perfecta.

**Tasa** ($R$) — Bits del latente dividido entre 500. Es el eje horizontal de
todas las figuras. $R = 1$ significa no comprimir; $R = 0.07$ (35 bits) es
compresión 14×.

**Compresión** — El inverso: $500 / L$. Un latente de 125 bits es compresión 4×.

**Entropía** ($H$) — Bits mínimos necesarios para representar la fuente sin
pérdida. Se calcula analíticamente para cada fuente, no se estima.

**Eb/N0** — Energía por bit dividida entre densidad de ruido. La medida estándar
de calidad de canal en telecomunicaciones. Aparece en la etapa JSCC.

---

## Conceptos de teoría de la información

**Cota inferior de Shannon (SLB)** — El BER mínimo físicamente alcanzable a una
tasa dada: $D \geq H_b^{-1}(H/n - R)$. **Ningún algoritmo puede bajar de ahí.**
Si un experimento la viola, tiene un bug — típicamente fuga entre entrenamiento
y prueba. Es la línea punteada negra de las figuras.

**Entropía binaria** ($H_b$) — $H_b(p) = -p\log_2 p - (1-p)\log_2(1-p)$. Se
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

**`random`** — Bits i.i.d. uniformes. $H = 500$ bits. Control negativo:
incompresible por Shannon.

**`oversamp`** — 125 símbolos, cada uno repetido 4 veces. $H = 125$ bits. Modela
una señal sobremuestreada antes de diezmar. Es el caso fácil, usado para
calibrar.

**`markov`** — Cadena de Markov con probabilidad de transición 0.05. $H = 143.9$
bits. Redundancia **correlacional**.

**`lowdim`** — $\mathrm{sign}(Wz)$ con $z \in \mathbb{R}^{32}$. $H = 164.9$ bits
(no 32 — ver *conteo de Cover*). Redundancia **geométrica**.

**`code`** — Código lineal sistemático de tasa 1/2 sobre GF(2) con checks de
grado 3. $H = 250$ bits. Redundancia **algebraica**. El caso crítico.

**GF(2)** — El cuerpo de dos elementos. Aritmética módulo 2, donde la suma es el
XOR.

**Check de grado 3** — Cada bit de paridad es el XOR de 3 bits de información.
La clave del proyecto: esa dependencia es invisible para métodos lineales.

**Conteo de Cover** — $C(n,k) = 2\sum_{i<k}\binom{n-1}{i}$, el número de
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

**PCA + cuantización** — El competidor justo: proyecta a $d$ componentes
principales y cuantiza cada una a $b$ bits, con $d \cdot b = L$.

**Decimación** — Enviar 1 de cada $m$ símbolos. Dos reconstrucciones:

- *hold* (sample-and-hold): repite el último valor transmitido.
- *vecino*: usa el transmitido más cercano.

La distinción importa: sobre `oversamp` a 125 bits, hold da 0.0000 y vecino
0.1241.

**JSCC** (*joint source-channel coding*) — Codificación conjunta de fuente y
canal: comprimir y proteger contra ruido en un solo paso.

**FEC** (*forward error correction*) — Código corrector de errores. Con LDPC o
turbo, un BER previo del orden de $10^{-2}$ se corrige hasta esencialmente cero.
Por eso $10^{-2}$ aparece como umbral en las figuras.

**LLR** (*log-likelihood ratio*) — Medida de confianza por bit. Los logits del
decoder funcionan como LLR: si los errores se concentran en bits de baja
confianza, un FEC de decisión blanda los limpia.

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

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
