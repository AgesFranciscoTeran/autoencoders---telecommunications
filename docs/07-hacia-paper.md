---
layout: default
title: Hacia un paper — qué queda
---

# Hacia un paper: qué queda

Los tres bloqueantes que tenía esta página están **resueltos**. Lo que queda es
otra cosa: extensiones que ampliarían el alcance, no huecos que invaliden lo
hecho.

---

## Resuelto

| brecha original | estado | cómo se resolvió |
|---|---|---|
| Barrido de autoencoders sin validar | **cerrada** | calibración in situ: `oversamp direct` L=125 da 3.2 × 10⁻⁷ |
| Diagnóstico de paridad nunca ejecutado | **cerrada** | ejecutado, y **refutó** la hipótesis que iba a sostener |
| Una sola semilla | **cerrada** | 4 semillas completas; márgenes de 4σ a 216σ |
| Solo encoders MLP | **cerrada** | convolución 1D probada, con la localidad como variable y control en las dos direcciones |
| `lowdim L=70` sin saturar en datos | **cerrada** | con pasos fijos, saturado desde 400k |

El diagnóstico de paridad merece una nota. Se construyó para **demostrar** que el
descenso de gradiente no aprende XOR de grado alto, y demostró lo contrario:
grado 3 se aprende con acc_test = 1.0000. La conclusión sobre `code` sobrevive
—el autoencoder sí fracasa— pero con un mecanismo distinto y mejor sustentado.
Un diagnóstico que refuta la hipótesis de quien lo diseñó es evidencia de que el
diseño era honesto.

---

## Contribuciones, ordenadas por fuerza

### 1. Mapa de viabilidad por tipo de estructura

En lugar de un veredicto único sobre "los autoencoders en telecomunicaciones",
un mapa que dice para qué tipo de redundancia sirven, a qué tasa, y con qué
arquitectura. Cada punto con cota teórica, baseline clásico y barras de error
sobre cuatro semillas.

*Sustento:* completo.

### 2. Resultado negativo sobre redundancia algebraica

El autoencoder recorre el 21 % del camino entre la estrategia trivial y el
óptimo sobre un código de bloque, pese a que el oráculo demuestra compresión sin
pérdida al 50 % de la tasa. Con tres evidencias convergentes: PCA es igualmente
ciego, un FEC de decisión blanda no ayudaría, y la estructura **sí** es
aprendible con supervisión directa.

*Sustento:* completo. La explicación es "el objetivo de reconstrucción no genera
camino de gradiente", no "SGD no puede con paridad".

### 3. Correspondencia arquitectura–estructura

La convolución ayuda con estructura local (+10.3 % en `oversamp`) y **estorba**
con estructura global (−21.4 % en `lowdim`). Confirmación en las dos
direcciones, que es lo que la hace sólida.

*Sustento:* completo, aunque con dos escalones (L=70 y L=125) y una sola semilla
por configuración.

### 4. Inestabilidad de escala en latentes binarios

Modo de fallo con mecanismo identificado y confirmado por medición (`pre_max`
correlaciona monótonamente con el BER final). Aplica a cualquier autoencoder con
latente binario y straight-through, no solo a este proyecto.

*Sustento:* completo, y probablemente lo más transferible.

### 5. Códec escalable anidado

Anidar cuesta +0.0011 BER de mediana y produce cuatro puntos de operación con un
solo modelo. La teoría de refinamiento sucesivo advertía de una brecha; la
medición dice que es despreciable.

*Sustento:* completo.

### 6. PCA es ciego a la redundancia algebraica

Diferencia dentro de 1σ sobre 10 semillas.

*Sustento:* completo, pero **es resultado de apoyo, no titular**. A alguien con
formación en teoría de la información no le sorprende que un método de segundo
orden no vea XOR de grado 3. Su valor está en establecer el contraste contra el
cual el resultado del autoencoder cobra sentido.

---

## Lo que ampliaría el alcance

Ninguno de estos invalida lo hecho. Son extensiones.

| # | Extensión | Por qué |
|---|---|---|
| 1 | Un objetivo auxiliar que supervise la estructura algebraica | Es la vía directa que abre el resultado del diagnóstico de paridad: si el obstáculo es el objetivo, cámbiese el objetivo |
| 2 | Señales reales (LDPC estándar, conjuntos públicos de RF) | Todas las fuentes son sintéticas. Es fortaleza metodológica —permite cotas exactas— pero un paper aplicado necesita un caso real |
| 3 | Convolución en los cuatro escalones y con varias semillas | Probada en dos escalones con una semilla |
| 4 | Selección de checkpoint por validación en todo el barrido | 1 de 4 semillas divergió; `gate_test3.py` ya lo implementa |
| 5 | Mejor modelo de entropía del latente (MADE, transformer) | El modelo autoregresivo lineal da una cota probablemente floja; el GRU falló porque el latente no tiene orden natural |
| 6 | VQ-VAE | La variante de VAE que aplica: latentes discretos y codebook como modelo de entropía. Un VAE gaussiano iría en contra: el término KL acota superiormente `I(x;z)` y reduciría la información justo cuando se quiere maximizarla |

La extensión 1 es la más valiosa: convierte un resultado negativo en una
pregunta con respuesta posible.

---

## Correspondencia cuaderno → paper

| Sección del paper | De dónde sale |
|---|---|
| Introducción y motivación | `index.md` |
| Trabajo relacionado | `04-bibliografia.md` (ya mapea qué establece cada obra) |
| Formulación del problema | `01-teoria.md` (cotas, tasa, métrica) |
| Metodología | `02-metodologia.md` (fuentes, modos, baselines, controles) |
| Configuración experimental | `05-reproducir.md` |
| Resultados | `03-resultados.md` |
| Limitaciones | esta página |
| Reproducibilidad | `code/` + `data/procedencia.json` |
| Apéndice: correcciones | `00-bitacora.md` |

La bitácora no va al paper, pero alimenta la sección de limitaciones y el
apéndice. Las siete correcciones de medición documentadas —incluidas las tres
hipótesis refutadas— son exactamente lo que un revisor pregunta.

---

## Prácticas que valieron la pena

**Prueba de calibración con respuesta conocida.** `oversamp` con el doble de
bits necesarios debe dar BER ≈ 0. Nueve rondas de depuración se justificaron
porque sin ella los 60 números medían bugs, no autoencoders.

**Controles con predicción explícita en ambas direcciones.** El control de la
convolución funcionó porque predecía dónde *no* debía ayudar. El de `random`
funcionó porque una victoria consistente ahí habría invalidado todo.

**Cotas de cordura automáticas.** La cota de Fano detectó que el primer
estimador de entropía daba resultados imposibles. El control de monotonía
detecta divergencias que un BER aislado no delata.

**Separar variables confundidas.** Pasos fijos frente a épocas fijas cambió la
conclusión sobre los datos. Variar `n_in` en el diagnóstico de paridad separó
"aprender XOR" de "encontrar el subconjunto".

**Procedencia automática.** `data/procedencia.json` con fecha, semillas,
versiones y hash de cada script.

---

## Preguntas abiertas

1. ¿Un objetivo auxiliar que supervise la estructura permitiría al autoencoder
   explotar la redundancia algebraica que sí es aprendible con supervisión?
2. ¿Se mantiene el resultado sobre `code` con un LDPC estándar en lugar de un
   código sintético?
3. ¿Cuál es la entropía real del latente con un modelo autoregresivo mejor que
   el lineal?
4. ¿Existe alguna combinación de tasa y estructura donde el autoencoder cruce el
   umbral de 10⁻² sobre una fuente no trivial?
5. ¿Por qué el preentrenamiento voraz por capas perjudica sistemáticamente en
   este problema, cuando en Hinton y Salakhutdinov (2006) ayudaba?

---

## Antes de publicar

- Verificar todos los datos bibliográficos contra las fuentes originales.
- Decidir la atribución institucional y consultarla con quien corresponda.
- Revisar que no quede material que no deba hacerse público.
- Ejecutar el nivel 1 de reproducción en una máquina limpia y confirmar que los
  números coinciden.
