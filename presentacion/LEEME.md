# Presentación

Resumen didáctico del proyecto en 21 páginas, formato 16:9, pensado para
exponerlo y para que alguien que llega sin contexto lo siga de principio a fin.

- `presentacion-autoencoders-bpsk.pdf` — el archivo que se proyecta. Vectorial,
  con las fuentes incrustadas y el texto seleccionable. No depende de internet.
- `index.html` — página que lo incrusta, para que los enlaces del cuaderno y del
  README sigan funcionando en GitHub Pages.
- `fonts/` — IBM Plex Sans y Serif en TTF, licencia OFL (`LICENSE-IBMPlex.txt`).
  Están aquí para que el PDF se regenere idéntico en cualquier máquina.

## Regenerar

```bash
python3 code/build_presentacion.py     # desde la raíz del repositorio
```

Necesita `matplotlib`, `numpy` y `pandas`. Escribe el PDF en esta carpeta. Si
`fonts/` no estuviera, el script avisa y cae a la tipografía por defecto de
matplotlib: el PDF sale igual, solo cambia la letra.

## De dónde salen los números

Los gráficos de cotas, varas y canal se construyen leyendo los CSV del
repositorio, así que no hay cifras tecleadas dos veces:

| página | gráfico | fuente de datos |
|---|---|---|
| 6 | mapa de cotas de Shannon | `data/cotas_teoricas.csv` |
| 7 | zona de viabilidad, piso y vara | `data/cotas_teoricas.csv`, `data/baselines_clasicos.csv` |
| 8 | curvas planas del JSCC | `data/v2_v3/v2_jscc.csv` |

El resto de páginas lleva los resultados de autoencoder transcritos en el
script, con la misma cifra que aparece en `docs/03-resultados.md` y en
`data/rbm_stack/LEEME.md`. Si se corrige un número allí, hay que corregirlo
también en `code/build_presentacion.py`.

## Estructura

| páginas | bloque |
|---|---|
| 1–8 | encuadre: de dónde venimos, el cuantizador, por qué contar bits, las fuentes, el piso, la zona de viabilidad, el alcance sin canal |
| 9–10 | diseño: los tres modos y la calibración del arnés |
| 11–14 | resultados: dónde gana, la frontera de arquitectura, el preentrenamiento, el punto que roza el umbral |
| 15–16 | el resultado negativo sobre redundancia algebraica y su mecanismo |
| 17–18 | hallazgos de método y las predicciones registradas |
| 19–20 | veredicto y trabajo abierto |
| 21 | anexo de configuración, no se presenta |

## Historial

La versión anterior era una presentación interactiva en reveal.js de 25
diapositivas. Está en el historial de git; se sustituyó por el PDF para no
depender de CDN externas al proyectar y para cambiar tablas por gráficos.
