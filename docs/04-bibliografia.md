---
layout: default
title: Bibliografía
---

# Bibliografía y dónde se usa cada referencia

Esta página no es una lista de lecturas: cada entrada indica **qué establece** y
**en qué parte del proyecto se usa**. Las referencias marcadas con ⚠ son las que
más condicionan el diseño experimental.

> **Verificar antes de citar formalmente.** Los datos bibliográficos (volumen,
> páginas, año de la versión publicada frente a la de preprint) deben
> confirmarse contra la fuente original antes de usarse en un documento
> académico. Varias de estas obras tienen versiones de preprint y de conferencia
> con paginación distinta.

---

## Fundamentos de autoencoders

**Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning
representations by back-propagating errors.* Nature, 323, 533–536.

El algoritmo que hace posible entrenar un encoder y un decoder conjuntamente.
Se usa implícitamente en todo el proyecto; su relevancia explícita aparece en el
diagnóstico de por qué el descenso de gradiente falla sobre paridad de grado
alto: el problema no es la representabilidad sino la optimización.

**⚠ Baldi, P., & Hornik, K. (1989).** *Neural networks and principal component
analysis: Learning from examples without local minima.* Neural Networks, 2(1),
53–58.

Establece que un autoencoder lineal con error cuadrático converge al subespacio
de las componentes principales. **Es la justificación teórica de usar PCA como
baseline**: si el autoencoder no le gana a PCA, no está aportando nada que la
teoría no garantizara ya para un modelo lineal. Fundamenta toda la sección de
"vara" del proyecto.

**⚠ Hinton, G. E., & Salakhutdinov, R. R. (2006).** *Reducing the dimensionality
of data with neural networks.* Science, 313(5786), 504–507.

Muestra que un autoencoder profundo supera a PCA **en datos con estructura**, y
propone el preentrenamiento voraz por capas con RBM. Dos usos directos: (a) es
la hipótesis que el proyecto pone a prueba fuente por fuente, y (b) su receta
se reproduce fielmente en `rbm_stack.py` (RBM con divergencia contrastiva,
decoder `Wᵀ`, ajuste fino posterior).

**Corrección de una etiqueta anterior.** El modo `stacked` de v4 se documentó
como «la receta de Hinton». No lo era: era una cascada de autoencoders profundos
comprimiendo códigos binarios, y su fracaso no decía nada sobre el paper. La
reproducción fiel **sí ayuda**, en 16 de 16 celdas, contra la predicción de que
la optimización moderna la había vuelto innecesaria.

**Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008).**
*Extracting and composing robust features with denoising autoencoders.*
Proceedings of ICML.

Entrenar con entradas corrompidas produce representaciones robustas. En el
proyecto es el puente conceptual hacia el ruido de canal: el modo
*canal-consciente* inyecta ruido AWGN sobre el latente durante el entrenamiento,
que es un autoencoder denoising cuya corrupción tiene significado físico.

**Kingma, D. P., & Welling, M. (2013).** *Auto-encoding variational Bayes.*
arXiv:1312.6114. Publicado en ICLR 2014.

Introduce el marco variacional y el truco de reparametrización. En este proyecto
se usa como referencia comparativa más que como método: el VAE regulariza el
latente con una divergencia KL hacia una previa continua, mientras que aquí se
necesita un latente **discreto** para que la tasa sea contable en bits. Se
menciona en la discusión de alternativas al estimador straight-through.

---

## Autoencoders en la capa física

**⚠ O'Shea, T., & Hoydis, J. (2017).** *An introduction to deep learning for the
physical layer.* IEEE Transactions on Cognitive Communications and Networking,
3(4), 563–575.

El trabajo que popularizó el "autoencoder de comunicaciones". **Distinción
crítica para no malinterpretar este proyecto**: el autoencoder de O'Shea y
Hoydis *expande* la señal y aprende un **código de canal** robusto al ruido,
entrenado extremo a extremo a través del canal. Este proyecto hace lo opuesto —
*comprime* la fuente. Son problemas distintos (codificación de canal frente a
codificación de fuente) y confundirlos invalida cualquier comparación de
resultados. La etapa JSCC del arnés es donde ambos enfoques se encuentran.

**Aoudia, F. A., & Hoydis, J. (2018).** *End-to-end learning of communications
systems without a channel model.* Asilomar Conference on Signals, Systems, and
Computers.

Elimina la necesidad de un modelo diferenciable del canal mediante aproximación
de gradiente por política. Relevante como extensión futura: el arnés actual
asume un canal AWGN diferenciable, lo cual es una simplificación.

**O'Shea, T. J., Corgan, J., & Clancy, T. C. (2016).** *Convolutional radio
modulation recognition networks.* Engineering Applications of Neural Networks
(EANN).

Aprendizaje profundo aplicado a señales de radio crudas para clasificación de
modulación. Se usa como evidencia de que las arquitecturas convolucionales
explotan la estructura temporal de las señales de RF, lo que motiva la fuente
`oversamp` y sugiere que un encoder convolucional podría superar al MLP actual.

**⚠ Wen, C.-K., Shih, W.-T., & Jin, S. (2018).** *Deep learning for massive MIMO
CSI feedback.* IEEE Wireless Communications Letters, 7(5), 748–751.

CsiNet: el caso de éxito real y bien citado de autoencoders para **compresión**
en telecomunicaciones. Es importante porque delimita el alcance de este
proyecto: allí lo que se comprime es la matriz de canal (altamente correlacionada
y dispersa en el dominio angular), no un flujo de bits. Es la evidencia externa
de que la conclusión correcta no es "los autoencoders no sirven en telecom" sino
"sirven sobre señales con el tipo adecuado de redundancia" — exactamente la tesis
del mapa de viabilidad.

---

## Teoría de la información

**⚠ Shannon, C. E. (1948).** *A mathematical theory of communication.* Bell
System Technical Journal, 27, 379–423 y 623–656.

El teorema de codificación de fuente fija el límite: una fuente de entropía H no
puede comprimirse sin pérdida por debajo de H bits. Es la razón por la que
`random` es el control negativo del experimento y por la que cualquier método
que parezca comprimirlo tiene un error metodológico.

**⚠ Shannon, C. E. (1959).** *Coding theorems for a discrete source with a
fidelity criterion.* IRE National Convention Record.

Introduce la teoría de tasa-distorsión. Es el marco donde la pregunta del
proyecto se vuelve precisa: no "¿cuánto comprime?" sino "¿qué par (tasa,
distorsión) alcanza, y a qué distancia del óptimo?".

**⚠ Cover, T. M., & Thomas, J. A. (2006).** *Elements of Information Theory*
(2ª ed.). Wiley.

Fuente de la cota inferior de Shannon usada en todo el proyecto: para una fuente
binaria estacionaria con distorsión de Hamming,
R(D) ≥ H(X)/n − H_b(D), de donde D ≥ H_b⁻¹(H/n − R). Se implementa en
`slb()` con inversión por bisección, y es la línea punteada negra de todas las
figuras.

**Cover, T. M. (1965).** *Geometrical and statistical properties of systems of
linear inequalities with applications in pattern recognition.* IEEE Transactions
on Electronic Computers, EC-14(3), 326–334.

El conteo de dicotomías linealmente separables, C(n,k) = 2·Σ binom(n−1,i). Se
usa para calcular exactamente la entropía de la fuente `lowdim`, que resulta ser
164.9 bits y no los 32 que sugiere ingenuamente la dimensión del manifold. Sin
este resultado la cota de esa fuente estaría mal puesta.

**Alcance de esta entrada.** Cover (1965) justifica el **conteo**, no el
**diseño** de la fuente. La procedencia del generador `sign(Wz)` está en la
sección siguiente; separarlas importa porque son dos objeciones distintas y se
responden por separado.

**⚠ Equitz, W. H. R., & Cover, T. M. (1991).** *Successive refinement of
information.* IEEE Transactions on Information Theory, 37(2), 269–275.

Caracteriza cuándo una descripción gruesa puede refinarse hasta el óptimo sin
penalización. Es el marco teórico de la **escalera anidada** del proyecto
(35 → 70 → 125 → 250 bits): predice que puede existir una brecha entre el
rendimiento anidado y el dedicado, y por eso el arnés compara explícitamente los
modos `nested` y `direct` en lugar de asumir que anidar es gratis.

---

## Medición de un bit y proyecciones por signo

Esta sección da procedencia al diseño de la fuente `lowdim`. Es la respuesta a
la objeción previsible —«¿por qué tomar un punto y hacerle 500 preguntas
binarias, y no otra construcción geométrica?»— y conviene tenerla explícita:
`sign(Wz)` no es una fuente inventada para el experimento, es un modelo de
medición establecido al que llegan de forma independiente tres líneas de
trabajo.

**Goemans, M. X., & Williamson, D. P. (1995).** *Improved approximation
algorithms for maximum cut and satisfiability problems using semidefinite
programming.* Journal of the ACM, 42(6), 1115–1145.

Origen del redondeo por hiperplano aleatorio: tomar un hiperplano al azar y
asignar según el lado en que cae cada vector. Se cita como raíz genealógica de
las dos entradas siguientes, que llegan al mismo objeto desde la búsqueda por
similitud y desde la adquisición de señales.

**⚠ Charikar, M. (2002).** *Similarity estimation techniques from rounding
algorithms.* Proceedings of the 34th Annual ACM Symposium on Theory of Computing
(STOC), 380–388.

Introduce la función hash sensible a la localidad basada en hiperplanos
aleatorios, y establece que la probabilidad de que dos vectores caigan en lados
opuestos es θ/π, con θ el ángulo entre ellos. **Es lo que justifica el BER como
métrica sobre `lowdim`**: la distancia de Hamming en la salida resulta ser una
función monótona del ángulo en el latente, de modo que el BER sobre esta fuente
mide error angular en ℝ³² salvo un cambio de escala. Sin esta identidad, el BER
sería solo la métrica del campo; con ella, es además la métrica natural de la
fuente.

**⚠ Boufounos, P. T., & Baraniuk, R. G. (2008).** *1-bit compressive sensing.*
42nd Annual Conference on Information Sciences and Systems (CISS), 16–21.

Establece el modelo de medición y = sign(Ax), donde cada medición conserva
únicamente el signo de una proyección lineal. **Es literalmente el generador de
`lowdim`.** Su función en el proyecto es quitarle a esa fuente el carácter de
artificio: la construcción «un punto latente, muchas preguntas binarias sobre
él» es el modelo de adquisición estándar de un cuantizador de un bit, no una
elección de conveniencia.

**Jacques, L., Laska, J. N., Boufounos, P. T., & Baraniuk, R. G. (2013).**
*Robust 1-bit compressive sensing via binary stable embeddings of sparse
vectors.* IEEE Transactions on Information Theory, 59(4), 2082–2102.

El lado de la recuperación: cuántas mediciones de un bit hacen falta para
reconstruir el vector latente y con qué robustez. Es el marco del **oráculo de
`lowdim`** —el método que conoce `W` y transmite `z` cuantizado— y sirve de
verificación externa de que ese oráculo está bien planteado como cota superior
de lo alcanzable.

**⚠ Li, Y., Tao, C., Seco-Granados, G., Mezghani, A., Swindlehurst, A. L., &
Liu, L. (2017).** *Channel estimation and performance analysis of one-bit
massive MIMO systems.* IEEE Transactions on Signal Processing, 65(15),
4075–4089.

Estimación de canal y desempeño del enlace ascendente en massive MIMO con ADCs
de un bit en la estación base, vía descomposición de Bussgang. Es la
**relevancia aplicada** de la fuente: los convertidores de un bit son un régimen
real, motivado por el consumo de potencia de los ADCs, y `lowdim` es su modelo
de observación.

**Límite que conviene declarar.** En `lowdim` la matriz `W` es gaussiana i.i.d.
—hiperplanos en posición general, que es la hipótesis exacta bajo la cual el
conteo de Cover es una igualdad y no una cota. Una matriz de canal real no es
i.i.d.: es correlacionada y dispersa en el dominio angular, como señala Wen et
al. (2018). La fuente modela la **geometría de la medición de un bit**, no la
estadística de un canal real, y no necesita hacerlo: su trabajo es aislar un
tipo de redundancia con entropía exacta. Sustituirla por un canal realista es la
extensión de señales reales listada en *Hacia un paper*.

**Mo, J., & Heath, R. W. (2015).** *Capacity analysis of one-bit quantized MIMO
systems with transmitter channel state information.* IEEE Transactions on Signal
Processing, 63(20), 5498–5512.

Análisis de capacidad del mismo régimen. Complementa a Li et al. por el lado
teórico: cuánta información sobrevive a la cuantización de un bit, que es la
pregunta de la que `lowdim` es un caso particular con `W` conocida.

---

## Optimización y latentes discretos

**⚠ Bengio, Y., Léonard, N., & Courville, A. (2013).** *Estimating or
propagating gradients through stochastic neurons for conditional computation.*
arXiv:1308.3432.

El estimador straight-through, que permite retropropagar a través de una función
signo no diferenciable. **Es lo que hace honesta la medición de compresión**: sin
latente binario, un latente de 50 dimensiones en float32 ocupa 1600 bits — más
que los 500 originales — y reportar "compresión 10×" sería falso. Con STE la
tasa es contable en bits por construcción.

**⚠ Shalev-Shwartz, S., Shamir, O., & Shammah, S. (2017).** *Failures of
gradient-based deep learning.* Proceedings of ICML.

Demuestra clases de problemas donde el descenso de gradiente falla pese a que la
red tiene capacidad de sobra, con la paridad como caso central. Fue el sustento
teórico de la hipótesis inicial del proyecto: que un MLP no podría descubrir la
redundancia de un código de bloque cuyos checks son de grado 3.

**El diagnóstico refutó esa aplicación.** Con 500 bits de entrada y grado 3, un
MLP alcanza acc_test = 1.0000; solo falla en grado 4. La obra sigue siendo
correcta —hay problemas donde el gradiente falla— pero el caso concreto de este
proyecto no era uno de ellos, y la explicación del fracaso sobre `code` tuvo que
reformularse en términos del objetivo de entrenamiento, no de la aprendibilidad.

**van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).** *Neural discrete
representation learning.* NeurIPS. (VQ-VAE)

Cuantización vectorial como alternativa al STE para obtener latentes discretos.
Es la extensión natural del proyecto si el straight-through resulta ser el cuello
de botella de optimización.

**Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).** *Layer normalization.*
arXiv:1607.06450.

Usado como corrección concreta: normalizar las preactivaciones antes de
binarizar evita que la saturación de la tangente hiperbólica anule el gradiente
del encoder. Es la hipótesis que explica el colapso observado en la primera
corrida completa.

---

## Cómo se relacionan

La lógica del proyecto encadena estas referencias así:

1. Shannon fija **qué es posible** (cotas).
2. Baldi–Hornik fija **qué es fácil** (PCA como vara).
3. Hinton–Salakhutdinov afirma que las redes profundas **superan** esa vara
   cuando hay estructura — la hipótesis a poner a prueba.
4. Bengio et al. hacen que la comparación sea **honesta** (bits reales).
5. Cover (1965) hace que la cota de `lowdim` esté **bien puesta** (164.9 bits,
   no 32).
6. Boufounos–Baraniuk y Charikar dan **procedencia al diseño de `lowdim`**: el
   generador es un modelo de medición establecido, y el BER es la métrica
   natural sobre él, no solo la del campo.
7. Shalev-Shwartz et al. predicen **dónde debería fallar** (redundancia
   algebraica).
8. Wen et al. muestran **dónde sí funciona** en la práctica (CSI, no bits).
9. Li et al. y Mo–Heath sitúan la fuente `lowdim` en un **régimen real** (ADCs
   de un bit en massive MIMO).
10. Equitz–Cover enmarcan la **contribución de ingeniería** (códec escalable).
11. O'Shea–Hoydis delimitan **qué no es este proyecto** (codificación de canal).