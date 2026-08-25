#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUSTEZ MULTI-SEMILLA + PROCEDENCIA

Un resultado con una sola semilla es una observacion, no un resultado.
Este script repite el calculo clasico sobre N semillas independientes y reporta
media +/- desviacion estandar, que es lo minimo publicable.

Ademas escribe data/procedencia.json con todo lo necesario para reconstruir
cualquier numero del cuaderno: fecha, hash del script, versiones, semillas.

Uso:
    python3 code/verificar_robustez.py --seeds 10
"""
import argparse, hashlib, json, os, platform, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generar_tablas as G

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RAIZ, "data")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def procedencia(semillas, extra=None):
    """Todo lo necesario para reconstruir un numero seis meses despues."""
    code = os.path.dirname(os.path.abspath(__file__))
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semillas": list(semillas),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "plataforma": platform.platform(),
        "scripts": {f: sha256(os.path.join(code, f))
                    for f in sorted(os.listdir(code)) if f.endswith(".py")},
        **(extra or {}),
    }


def corre_semilla(seed):
    """Devuelve BER de PCA por (fuente, escalon) para una semilla."""
    filas = []
    for reg in ["random", "oversamp", "markov", "lowdim", "code"]:
        Xtr, Xte, side, H = G.genera(reg, seed=seed)
        best = G.pca_curva(Xtr, Xte, G.LADDER)
        for L in G.LADDER:
            filas.append(dict(semilla=seed, fuente=reg, latente_bits=L,
                              ber_pca=best[L][0]))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    semillas = list(range(a.seeds))

    filas = []
    for s in semillas:
        filas += corre_semilla(s)
        print(f"  semilla {s} lista", flush=True)

    df = pd.DataFrame(filas)
    res = (df.groupby(["fuente", "latente_bits"]).ber_pca
             .agg(media="mean", desv="std", n="count").reset_index())
    res.to_csv(os.path.join(OUT, "pca_multisemilla.csv"), index=False)

    # --- el contraste que sostiene el hallazgo firme ---
    piv = res.set_index(["fuente", "latente_bits"])
    print(f"\n{'='*72}")
    print(f"PCA sobre 'code' vs 'random'  ({a.seeds} semillas, media +/- desv)")
    print(f"{'='*72}")
    print(f"{'L':>5s} {'code':>16s} {'random':>16s} {'diferencia':>18s}")
    comp = []
    for L in G.LADDER:
        c, r = piv.loc[("code", L)], piv.loc[("random", L)]
        d = c.media - r.media
        sd = float(np.hypot(c.desv, r.desv))          # desv de la diferencia
        comp.append(dict(latente_bits=L, code_media=c.media, code_desv=c.desv,
                         random_media=r.media, random_desv=r.desv,
                         diferencia=d, desv_diferencia=sd,
                         indistinguible=bool(abs(d) < 2 * sd)))
        print(f"{L:5d} {c.media:8.4f}+/-{c.desv:.4f} {r.media:8.4f}+/-{r.desv:.4f} "
              f" {d:+8.4f}+/-{sd:.4f}")
    pd.DataFrame(comp).to_csv(os.path.join(OUT, "pca_code_vs_random.csv"), index=False)

    todos = all(c["indistinguible"] for c in comp)
    print(f"\n  Indistinguible (|dif| < 2 sigma) en los 4 escalones: "
          f"{'SI' if todos else 'NO'}")
    if todos:
        print("  -> El hallazgo se sostiene con barras de error. Publicable.")

    prov = procedencia(semillas, {"resultado_pca_indistinguible": todos})
    with open(os.path.join(OUT, "procedencia.json"), "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"\n[guardado] {OUT}/pca_multisemilla.csv")
    print(f"[guardado] {OUT}/pca_code_vs_random.csv")
    print(f"[guardado] {OUT}/procedencia.json")


if __name__ == "__main__":
    main()
