#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figuras a partir de data/*.csv.  Uso: python3 code/generar_figuras.py"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, F = os.path.join(R, "data"), os.path.join(R, "figs")
os.makedirs(F, exist_ok=True)
cot = pd.read_csv(f"{D}/cotas_teoricas.csv"); bl = pd.read_csv(f"{D}/baselines_clasicos.csv")
pca = pd.read_csv(f"{D}/pca_vs_random.csv")
regs = ["random", "oversamp", "markov", "lowdim", "code"]

fig, axes = plt.subplots(1, 5, figsize=(19, 3.9), sharey=True)
for ax, reg in zip(axes, regs):
    c = cot[cot.fuente == reg].sort_values("latente_bits")
    b = bl[bl.fuente == reg].sort_values("latente_bits")
    ax.plot(b.tasa, np.maximum(b.ber_baseline, 5e-5), "s-", color="#BA7517",
            label="mejor clasico", zorder=3)
    ax.plot(c.tasa, np.maximum(c.ber_minimo, 5e-5), "k:", lw=1.8, label="cota Shannon")
    ax.fill_between(c.tasa, np.maximum(c.ber_minimo, 5e-5),
                    np.maximum(b.ber_baseline.values, 5e-5),
                    color="#5DCAA5", alpha=0.22, label="zona de viabilidad")
    ax.axhline(1e-2, ls="--", c="#1D9E75", lw=1, label="1e-2 (umbral FEC)")
    ax.set_yscale("log"); ax.set_ylim(4e-5, 0.6)
    h = c.H_sobre_n.iloc[0]
    ax.axvline(h, ls=":", c="grey", lw=1.2)
    ax.set_title(f"{reg}\nH/n = {h:.3f}", fontsize=11)
    ax.set_xlabel("tasa R = bits / 500"); ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("BER"); axes[-1].legend(fontsize=7.5, loc="lower left")
fig.suptitle("Zona de viabilidad por fuente: un autoencoder solo aporta si cae dentro del area verde",
             fontsize=12)
plt.tight_layout(); plt.savefig(f"{F}/zona_viabilidad.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.4))
w = 0.35; x = np.arange(4)
cc = pca[pca.fuente == "code"].sort_values("latente_bits")
rr = pca[pca.fuente == "random"].sort_values("latente_bits")
ax.bar(x - w/2, cc.ber_pca, w, label="code (2x comprimible)", color="#D85A30")
ax.bar(x + w/2, rr.ber_pca, w, label="random (incomprimible)", color="#888780")
ax.set_xticks(x); ax.set_xticklabels([f"L={v}" for v in cc.latente_bits])
ax.set_ylabel("BER de PCA"); ax.legend()
ax.set_title("PCA es ciego a la redundancia algebraica\nel codigo de bloque le resulta indistinguible del ruido")
ax.grid(alpha=0.3, axis="y"); plt.tight_layout()
plt.savefig(f"{F}/pca_ciego.png", dpi=140); plt.close()
print("[guardado]", f"{F}/zona_viabilidad.png"); print("[guardado]", f"{F}/pca_ciego.png")
