# Preentrenamiento fiel y confundido de arquitectura

Dos experimentos del cierre extendido (septiembre 2026).

## `rbm_stack.py` — reproducción fiel de Hinton & Salakhutdinov (2006)

Tres brazos sobre la misma escalera estrecha (500 → 250 → 125 → 70 → 35,
sigmoides intermedias, `BatchNorm1d(affine=False)` + STE en el cuello):

- `ladder_random` — inicialización aleatoria, extremo a extremo (control)
- `stacked_ae` — voraz por capas con autoencoder superficial sobre activaciones
  continuas, luego ajuste fino
- `stacked_rbm` — voraz por capas con RBM binaria-binaria (CD-1), encoder `W`,
  decoder `Wᵀ`, luego ajuste fino

4 semillas, 30 000 pasos de ajuste fino, mejor checkpoint por validación.
El script vive en `code/rbm_stack.py` del clon de trabajo; copiar aquí.

### Resumen (BER de test, media sobre 4 semillas)

| fuente | L | aleatoria | pre. AE | pre. RBM | vara |
|---|---|---|---|---|---|
| lowdim | 35 | 0.2610 | **0.1833** | 0.1911 | 0.1954 |
| lowdim | 70 | 0.1376 | **0.1140** | 0.1146 | 0.1861 |
| lowdim | 125 | 0.0854 | 0.0720 | **0.0710** | 0.0610 |
| lowdim | 250 | 0.0386 | 0.0394 | **0.0369** | 0.0492 |
| markov | 35 | 0.1979 | **0.1418** | 0.1732 | 0.1531 |
| markov | 70 | 0.1010 | **0.0838** | 0.0891 | 0.0914 |
| markov | 125 | 0.0464 | 0.0456 | **0.0441** | 0.0488 |
| markov | 250 | 0.0135 | 0.0143 | **0.0107** | 0.0250 |
| code | 35–250 | ≈ ruido | ≈ ruido | ≈ ruido | — |

Predicción registrada antes de correr: el preentrenamiento no ayudaría.
**Refutada en 16/16 celdas.**

## `confundido.py` — ¿checkpoint o arquitectura?

2×2 con protocolo idéntico (importa `finetune` de `rbm_stack`): MLP ancho de
v4 (1536×4) contra escalera estrecha, ambos con mejor checkpoint.

| caso | v4 (sin ckpt) | ancho + ckpt | escalera | vara |
|---|---|---|---|---|
| markov 250 | 0.0345 | 0.0288 | **0.0135** | 0.0250 |
| markov 125 | 0.0501 | 0.0614 | **0.0464** | 0.0488 |
| lowdim 250 | 0.0653 | 0.0553 | **0.0386** | 0.0492 |
| markov 35 | 0.1498 | **0.1564** | 0.1979 | 0.1531 |

**Veredicto: arquitectura, 63–100 % de la brecha.** La escalera es perfectamente
estable (final − mejor = 0.0000 en 16/16 corridas); el MLP ancho se degrada
hasta 0.009. El signo se invierte con la tasa: el ancho gana a L=35, la escalera
a L ≥ 125.

## Advertencia sobre conclusiones anteriores

- El modo `stacked` de v4 se documentó como «la receta de Hinton». No lo era.
- «El AE no gana nunca en R ≥ 0.25» era un artefacto de una sola arquitectura.
- `markov` L=35 con el MLP ancho ganaba en v4 por 4σ; con el protocolo de
  `rbm_stack` pierde. Es sensible al protocolo.
