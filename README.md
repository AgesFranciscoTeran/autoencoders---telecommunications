# Autoencoders para compresión de señales BPSK

Bitácora personal de trabajo: experimentos, decisiones, errores y correcciones.

**Pregunta.** ¿Puede un autoencoder comprimir un vector de 500 símbolos BPSK
(±1) y reconstruirlo con fidelidad suficiente para un enlace real?

## Empieza por aquí

- **[Bitácora](docs/00-bitacora.md)** — el camino completo, etapa por etapa
- **[Glosario](docs/06-glosario.md)** — todos los términos
- **[Resultados](docs/03-resultados.md)** — qué está firme y qué no

Referencia: [Teoría](docs/01-teoria.md) · [Metodología](docs/02-metodologia.md) ·
[Bibliografía](docs/04-bibliografia.md) · [Reproducir](docs/05-reproducir.md) ·
[Hacia un paper](docs/07-hacia-paper.md)

## La idea en una línea

Un autoencoder solo sirve si le gana a PCA usando el mismo número de bits, y
solo puede juzgarse contra la cota de Shannon. Un BER suelto no significa nada.

## Estado

| Componente | Estado |
|---|---|
| Cotas teóricas | Firme |
| Baselines clásicos | Firme |
| PCA ciego a redundancia algebraica | Firme |
| Barrido de autoencoders | Firme (4 semillas) |
| Encoder convolucional | Firme |
| Diagnóstico de paridad | Ejecutado (refutó la hipótesis inicial) |

**Proyecto cerrado.** No son viables a tasas agresivas (18/20 puntos del barrido
principal por encima del umbral FEC), pero sí en una franja estrecha con decisión
blanda: `markov` L=250 con escalera y RBM llega a BER 0.0107 con el 90 % de los
bits en 0.0011. Y superan a lo clásico en casi todo el mapa de `markov` y
`lowdim` si la arquitectura se elige por tasa.

## Reproducir sin GPU, en segundos

```bash
pip install -r requirements.txt
python3 code/generar_tablas.py
```

Debe imprimir:

```
PCA sobre 'code' vs 'random' (si son iguales, PCA es ciego al codigo):
      L     code   random      dif
     35   0.4153   0.4161  -0.0008
     70   0.3794   0.3789  +0.0005
    125   0.3350   0.3354  -0.0004
    250   0.2529   0.2536  -0.0007
```

## Licencia

MIT.
