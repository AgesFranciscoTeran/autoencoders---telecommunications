# Artefactos de la versión 1

Primer arnés del proyecto: dos fuentes (`random`, `lowdim`), dos arquitecturas,
**latente continuo** en `float32`, pérdida MSE, 4 000 muestras y 100 épocas en
CPU. Análisis en [`docs/00-bitacora.md`](../../docs/00-bitacora.md), sección
«Anexo: las versiones previas».

## Advertencia sobre `figs/v1/lineal_vs_profundo.png`

Muestra al autoencoder lineal ganando al profundo en **10 de 10**
configuraciones. La conclusión no sobrevivió al arnés corregido.

Es un caso instructivo: la misma conclusión apareció después en v2, pero **por
una causa distinta**. En v2 fue la inestabilidad de escala del latente binario;
aquí el latente es continuo, sin estimador straight-through, así que esa
explicación no aplica. La causa probable en v1 es el tamaño de los datos
(4 000 muestras) frente a un modelo profundo sobre una fuente casi lineal.

## Advertencia sobre la meseta de `lowdim`

El modelo profundo da BER 0.0524, 0.0523 y 0.0520 en L=200, 100 y 50. Se
interpretó como que el autoencoder «encuentra la dimensión intrínseca» (k=32).

**Esa lectura es incorrecta.** El modelo lineal no hace meseta en ese rango:
mejora de 0.0372 a 0.0186. Si la meseta fuera la dimensión intrínseca, ambos la
tendrían. Era un piso de optimización del modelo profundo.

Lo que sí es real es el colapso por debajo de L≈32 (0.2107 en L=20, 0.3569 en
L=8), donde la información se pierde de verdad.

## Lo que sí sobrevivió

- **La compresibilidad es propiedad de la señal, no del autoencoder.** Es el
  hallazgo que orientó todo el proyecto.
- **La brecha entrenamiento/prueba sobre `random`** como firma de memorización
  sin generalización (L=200: train 0.0663, test 0.2639).
- **El presupuesto honesto de bits.** 50 dimensiones en `float32` son 1600 bits
  contra 500 de fuente: un factor de 0.31×, una expansión. De aquí sale la
  decisión de usar latente binario en todo el proyecto.

| archivo | contenido |
|---|---|
| `v1_resultados.csv` | barrido completo: 2 fuentes × 2 arquitecturas × 5 latentes |
| `figs/v1/ber_vs_latente.png` | BER de test contra dimensión latente |
| `figs/v1/lineal_vs_profundo.png` | comparación de arquitecturas (ver advertencia) |
| `figs/v1/random_train_vs_test.png` | brecha de generalización sobre ruido |
| `code/historico/ae_telecom_v1.py` | el script |
