---
layout: default
title: Reproducir los experimentos
---

# Reproducir los experimentos

El proyecto está diseñado para reproducirse en **dos niveles independientes**.
El nivel 1 no necesita GPU ni PyTorch y valida toda la parte teórica y los
baselines clásicos; el nivel 2 necesita GPU y reproduce el barrido de
autoencoders.

---

## Nivel 1 — sin GPU, en segundos

Regenera las cotas de Shannon, los baselines clásicos y el hallazgo de que PCA
es ciego a la redundancia algebraica. Es lo que respalda todos los resultados
marcados como verificados.

```bash
cd ruta/al/repositorio
pip install -r requirements.txt

python3 code/generar_tablas.py     # escribe data/*.csv
python3 code/generar_figuras.py    # escribe figs/*.png
```

Salida esperada al final de `generar_tablas.py`:

```
PCA sobre 'code' vs 'random' (si son iguales, PCA es ciego al codigo):
      L     code   random      dif
     35   0.4153   0.4161  -0.0008
     70   0.3794   0.3789  +0.0005
    125   0.3350   0.3354  -0.0004
    250   0.2529   0.2536  -0.0007
```

Si esos cuatro números se reproducen, el resultado central verificado del
proyecto queda confirmado en tu máquina. Las semillas están fijadas
(`SEED = 0`, `numpy.random.default_rng`), así que los valores deben coincidir
dígito a dígito.

### Qué comprobar

| Comprobación | Criterio |
|---|---|
| `random` alcanza el BER mínimo | ninguna fila con `ber_minimo` < cota |
| `code` a 250 bits | `oraculo 250/250` con BER 0.0000 |
| `oversamp` a 125 bits | `decimacion m=4 (hold)` con BER 0.0000 |
| PCA sobre `code` ≈ PCA sobre `random` | diferencia < 0.001 en los 4 escalones |

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

## Estructura del repositorio

```
.
├── index.md                   página principal
├── docs/                      teoría, metodología, resultados, bibliografía
├── code/
│   ├── generar_tablas.py      nivel 1: cotas y baselines (sin GPU)
│   ├── generar_figuras.py     nivel 1: figuras
│   ├── ae_telecom_v4.py       arnés principal de autoencoders
│   ├── gate_test.py           test de calibración
│   ├── check_v4.py            verificación post-corrida
│   └── run_h200.sh            lanzador multi-GPU
├── data/                      CSV regenerables (nivel 1)
└── figs/                      figuras regenerables (nivel 1)
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
