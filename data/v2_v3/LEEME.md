# Artefactos de las versiones intermedias (v2 y v3)

Corridas piloto en CPU: 16 000 muestras de entrenamiento, 40 épocas, capas
ocultas de 384. **No son los resultados finales del proyecto.** Se conservan
porque dos de sus conclusiones fueron revertidas después y la causa es
identificable. El análisis está en [`docs/00-bitacora.md`](../../docs/00-bitacora.md),
sección «Anexo: las versiones intermedias».

## Advertencia sobre `figs/v2_v3/v3_paridad.png`

El título de esa figura afirma:

> «Un MLP no aprende paridad de grado alto (por eso el AE no ve la redundancia
> algebraica)»

**Esa afirmación es falsa y fue refutada.** Con presupuesto de entrenamiento
suficiente (400 000 muestras, 60 épocas frente a las 60 000 y 25 del piloto), un
MLP aprende paridad de grado 3 con exactitud de test 1.0000. El grado 3 es
justamente el de los checks del código usado en la fuente `code`.

La figura se conserva como registro de la hipótesis descartada. No usarla como
resultado.

## Advertencia sobre `v2_barrido_binario.csv`

Muestra al autoencoder lineal ganando al profundo en 15 de 16 configuraciones.
Es un artefacto de la inestabilidad de escala del latente binario, no
diagnosticada en ese momento. Con el encoder corregido
(`BatchNorm1d(affine=False)`) el modelo profundo supera a ambos.

## Contenido

| archivo | qué contiene |
|---|---|
| `log_v2.txt`, `log_v3.txt` | salida completa de cada corrida, con su configuración |
| `v2_ablacion.csv` | latente continuo vs binario, con el coste en bits reales |
| `v2_barrido_binario.csv` | BER vs tasa, profundo vs lineal (ver advertencia) |
| `v2_jscc.csv` | primer intento de JSCC, curvas planas |
| `v3_baselines.csv` | PCA, decimación y oráculos: la primera vara del proyecto |
| `v3_ae_vs_baselines.csv` | autoencoder frente a esa vara |
| `v3_paridad.csv` | diagnóstico de paridad del piloto (ver advertencia) |

## Lo que sí sobrevivió de estos pilotos

- El coste real del latente continuo: 128 dimensiones en `float32` son 4096
  bits para 500 bits de fuente, una expansión de 0.12×. Origen de la decisión de
  usar latente binario.
- La necesidad de una vara clásica y un piso teórico en cada punto.
- El oráculo de `code` alcanzando BER 0.0000 a tasa 0.500, pilar del resultado
  negativo.
- Reportar el exceso sobre el piso de compresión en JSCC, no el BER total.
