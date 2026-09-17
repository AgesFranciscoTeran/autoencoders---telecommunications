---
layout: default
title: Reproducir los experimentos
---

# Reproducir los experimentos

El proyecto se reproduce en **tres niveles independientes**. El nivel 1 no
necesita GPU ni PyTorch y valida la parte teórica y los baselines clásicos; el
nivel 2 necesita GPU y reproduce el barrido principal de autoencoders; el nivel
3 son los experimentos posteriores al barrido, cada uno con su propio script.

---

## Nivel 1 — sin GPU, en segundos

Regenera las cotas de Shannon, los baselines clásicos y el hallazgo de que PCA
es ciego a la redundancia algebraica. Es lo que respalda todos los resultados
marcados como verificados.

```bash
cd ruta/al/repositorio
pip install -r requirements.txt

python3 code/generar_tablas.py     # cotas + baselines históricos
python3 code/vara_definitiva.py    # la vara vigente (Lloyd-Max)
python3 code/consolidar_varas.py   # -> data/baselines_clasicos.csv
python3 code/generar_figuras.py    # escribe figs/*.png
```

**Tres scripts y no uno, a propósito.** `generar_tablas.py` usa el cuantizador
uniforme min/max, que es la vara **retractada**: su rango crece con el número de
muestras, así que el baseline empeoraba cuantos más datos se le daban. Escribía
`baselines_clasicos.csv`, el mismo archivo que hoy guarda la vara buena, de modo
que ejecutar el nivel 1 deshacía la corrección en silencio y arrastraba a las
figuras y a la presentación, que leen de ahí.

Ahora escribe `baselines_minmax_historico.csv` y no toca la vara vigente. Lo que
sigue produciendo es válido: las cotas de Shannon y la evidencia de que PCA es
ciego al código.

Salida esperada al final de `generar_tablas.py`:

```
PCA sobre 'code' vs 'random' (si son iguales, PCA es ciego al codigo):
      L     code   random      dif
     35   0.4153   0.4161  -0.0008
     70   0.3794   0.3789  +0.0005
    125   0.3350   0.3354  -0.0004
    250   0.2529   0.2536  -0.0007
```

Si esos cuatro números se reproducen, el hallazgo de que PCA es ciego al
código queda confirmado en tu máquina. Las semillas están fijadas (`SEED = 0`,
`numpy.random.default_rng`), así que los valores deben coincidir dígito a
dígito. El hallazgo no depende del cuantizador, porque se aplica el mismo a
`code` y a `random`; la **vara** sí depende, y esa se regenera aparte.

### La vara vigente, en dos pasos

Un script **calcula** y otro **formatea**. La separación es deliberada: el
segundo no tiene ninguna decisión científica dentro, así que se puede releer en
un minuto y no hay dos sitios donde el mismo número pueda divergir.

```bash
python3 code/vara_definitiva.py     # -> data/varas_definitivas.json
python3 code/consolidar_varas.py    # -> data/baselines_clasicos.csv
```

`vara_definitiva.py` prueba, para cada fuente y escalón, cuantización uniforme
frente a Lloyd-Max, síntesis por transpuesta frente a mínimos cuadrados,
decimación donde aplica y el oráculo de paridad en `code`, y se queda con la
mejor. Tarda varios minutos y no necesita GPU.

`consolidar_varas.py` escribe una fila por (fuente, escalón) con la vara
vigente y, al lado, las alternativas que perdieron —incluida la columna
`ber_minmax_historico`, la retractada— para que el recálculo quede auditable
desde el propio CSV. Comprueba además que no falte ninguna fuente: el CSV
anterior había perdido las cuatro filas de `code` sin que nadie lo notara, y los
consumidores se quedaban con la vara en blanco.

```bash
python3 code/consolidar_varas.py --verificar   # no escribe; compara CSV y JSON
```

Ese modo es el que conviene dejar en cualquier comprobación previa a publicar:
falla si el CSV en disco no es exactamente lo que el JSON implica.

### La vara de `ber_tapados`

`ber_tapados` —el BER medido solo en las posiciones enmascaradas— se reportó
durante dos semanas contra 0.5 y contra nada más. `vara_tapados.py` calcula el
techo: el mejor valor alcanzable por alguien que conoce la estructura de la
fuente.

```bash
python3 code/vara_tapados.py                 # -> data/varas_tapados.json
python3 code/vara_tapados.py --sin-comparar  # solo la vara, sin leer el nivel 3
```

Tarda unos 25 segundos y no necesita GPU. Cuatro de las cinco fuentes admiten el
óptimo de Bayes en forma cerrada —`random` analítico, `oversamp` por
combinatoria de bloques, `markov` por forward-backward y `code` por rango sobre
GF(2)— y son **pisos exactos**. `lowdim` no: ahí la posterior es una gaussiana
restringida a un cono, así que se usa el estimador de margen máximo conociendo
*W*, que es una **vara alcanzable**, no un óptimo demostrado. La salida etiqueta
cada fila con su tipo, y conviene conservar la distinción al citarla.

Si existen `data/denoising/denoising_resumen.csv` o
`data/combinado/combinado.csv`, imprime además medido contra vara y la fracción
del camino realmente alcanzable. Las dos corridas salen etiquetadas por origen y
no se fusionan: son experimentos distintos, y mezclarlos en una misma fila ya
costó una corrección.

### Qué comprobar

| Comprobación | Criterio |
|---|---|
| `random` alcanza el BER mínimo | ninguna fila con `ber_minimo` < cota |
| `code` a 250 bits | `oraculo 250/250` con BER 0.0000 |
| `oversamp` a 125 bits | `decimacion m=4 (hold)` con BER 0.0000 |
| PCA sobre `code` ≈ PCA sobre `random` | diferencia < 0.001 en los 4 escalones |
| vara de `ber_tapados` sobre `random` | 0.5000 con sd 0.0000 en las tres *p* |

---

## Nivel 2 — barrido de autoencoders (necesita GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Test de calibración primero

**No lances el barrido completo sin pasar este test.** Entrena la fuente
`oversamp` con un latente de 250 bits para 125 bits de entropía: al autoencoder
le sobra el doble de capacidad, así que la respuesta correcta es BER ≈ 0. Si
falla, cualquier resultado posterior mide bugs y no autoencoders.

```bash
python3 code/gate_test.py --device cuda
```

El test reporta además dos diagnósticos de colapso: porcentaje de
preactivaciones saturadas y porcentaje de bits latentes que efectivamente
varían.

### Barrido completo

```bash
bash code/run_h200.sh ./res 0,1     # dos GPUs, shards independientes
python3 code/check_v4.py ./res      # verificación y tabla final
```

Argumentos de `run_h200.sh`: el primero es el directorio de salida, el segundo
es la lista de GPUs lógicas separadas por comas. El script hace un smoke test en
CPU antes de tocar la GPU, lanza un shard por GPU y consolida al terminar.

El barrido es **reanudable**: los resultados se acumulan en JSONL con una clave
por configuración, de modo que si un shard falla puedes relanzar y solo se
ejecuta lo que falta.

### Modos disponibles

```bash
python3 code/ae_telecom_v4.py --mode smoke   # minutos, CPU, valida que corre
python3 code/ae_telecom_v4.py --mode quick   # ~1 h en una GPU
python3 code/ae_telecom_v4.py --mode full    # barrido completo
```

---

## Nivel 3 — experimentos posteriores al barrido

Cada uno es autocontenido, se lanza desde la raíz del repositorio y necesita
GPU. Todos escriben procedencia con fecha, semillas, versiones y hash de git.

| script | qué responde | escribe | coste |
|---|---|---|---|
| `cierre.py --stage all` | paridad, convolución y saturación de datos | `./cierre/*.json` | horas |
| `confundido.py` | separa «mejor checkpoint» de «arquitectura» | `./confundido/confundido.json` | ~1 h |
| `rbm_stack.py` | preentrenamiento fiel a Hinton frente a escalera aleatoria | `./rbm_stack/rbm_stack.csv` | horas |
| `umbral.py` | si `markov` L=250 cruza 10⁻² | `./umbral/umbral.json` | ~1 h |
| `denoising.py` | si el enmascarado abre el camino de gradiente | `data/denoising/` | ~4 h |
| `combinado.py` | factorial inicialización × ruido | `data/combinado/` | ~5 h |
| `analizar_combinado.py` | rehace la estadística del factorial sin volver a entrenar | stdout | segundos |

```bash
python3 code/denoising.py --quick          # humo: 1 semilla, pasos cortos
python3 code/denoising.py                  # rejilla completa, 4 semillas
python3 code/combinado.py --solo-primaria  # solo markov L=250, ~15 min
python3 code/combinado.py                  # factorial completo
python3 code/combinado.py --quick          # humo, escribe en data/combinado_smoke/
python3 code/analizar_combinado.py --parcial
```

`denoising.py` y `combinado.py` son **reanudables**: acumulan una línea por
celda en un JSONL y al relanzar solo ejecutan lo que falta. El modo `--quick`
escribe en un directorio aparte por construcción, para que una prueba de humo no
pueda mezclarse con una corrida real. Separar el análisis de la ejecución
(`analizar_combinado.py`) permite corregir el criterio estadístico sin repetir
cinco horas de GPU.

Dos avisos sobre los datos ya publicados de este nivel:

- La columna `vara` de `data/denoising/denoising_resumen.csv` conserva los
  valores **anteriores** al recálculo con Lloyd-Max. Comparar contra ella da
  veredictos equivocados en `lowdim`: contra 0.1776 el mejor punto parece
  victoria, contra 0.0977 es derrota. Las varas vigentes están en
  `data/baselines_clasicos.csv`.
- El diccionario `VARAS` codificado en `rbm_stack.py` también es el antiguo.
  `denoising.py` lo sobrescribe leyendo el CSV; `rbm_stack.py` ejecutado
  directamente, no.

---

## Estructura del repositorio

```
.
├── index.md                   página principal
├── docs/                      teoría, metodología, resultados, bibliografía
├── code/
│   ├── generar_tablas.py      nivel 1: cotas y baselines (sin GPU)
│   ├── generar_figuras.py     nivel 1: figuras
│   ├── vara_definitiva.py     nivel 1: calcula la vara vigente (Lloyd-Max)
│   ├── consolidar_varas.py    nivel 1: JSON de varas -> baselines_clasicos.csv
│   ├── vara_tapados.py        nivel 1: vara de ber_tapados (Bayes + margen máximo)
│   ├── ae_telecom_v4.py       nivel 2: arnés principal de autoencoders
│   ├── gate_test.py           nivel 2: test de calibración
│   ├── check_v4.py            nivel 2: verificación post-corrida
│   ├── run_h200.sh            nivel 2: lanzador multi-GPU
│   ├── cierre.py              nivel 3: paridad, convolución, datos
│   ├── confundido.py          nivel 3: checkpoint frente a arquitectura
│   ├── rbm_stack.py           nivel 3: preentrenamiento fiel a Hinton
│   ├── umbral.py              nivel 3: cruzar 10⁻² con decisión blanda
│   ├── denoising.py           nivel 3: enmascarado de entrada
│   ├── combinado.py           nivel 3: factorial inicialización × ruido
│   ├── analizar_combinado.py  nivel 3: análisis separado de la corrida
│   ├── entropia_latente2.py   entropía del latente (estimador autoregresivo)
│   ├── build_presentacion.py  regenera el PDF de la presentación
│   └── historico/             arneses v1–v3, solo referencia
├── data/                      CSV y JSON de resultados
├── figs/                      figuras regenerables (nivel 1)
└── presentacion/              PDF de 21 páginas y su página de incrustación
```

## Entorno de referencia

El barrido se ejecuta en una GPU con soporte bf16 (probado con dos GPUs
lógicas en paralelo). El nivel 1 se ejecutó con
NumPy 2.4 y pandas 2.x sobre CPU, y no depende del hardware.

## Publicar este sitio

El repositorio está listo para GitHub Pages con Jekyll:

1. Sube el contenido a un repositorio público.
2. Settings → Pages → Source: *Deploy from a branch*, rama `main`, carpeta `/`.
3. El sitio queda en `https://USUARIO.github.io/REPO/`.

Si el repositorio es privado, GitHub Pages requiere plan de pago; en ese caso
los archivos markdown se leen igual directamente en GitHub sin activar Pages.
