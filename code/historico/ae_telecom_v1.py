#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoders para señales de telecomunicaciones (vectores BPSK de +/-1, dim 500).
================================================================================
Objetivo: comprimir un vector de 500 bits (+1/-1) al latente mas pequeno posible
con un encoder, reconstruirlo con un decoder, y medir la calidad con BER
(Bit Error Rate), la metrica natural en telecom.

Idea central (importante para el tutor):
    La compresibilidad depende de la ESTRUCTURA de la senal, no del autoencoder.
    - Bits i.i.d. aleatorios  -> entropia = 500 bits (Shannon). No se pueden
      comprimir sin perdida por debajo de 500 bits. El AE NO puede generalizar.
    - Senal con redundancia    -> el AE aprende a comprimir hasta ~la dimension
      intrinseca del generador, con BER practicamente 0.

Se comparan: regimenes de datos (random vs low-dim), arquitecturas (lineal vs
profunda), un barrido de dimension latente, y analisis honesto de presupuesto de
bits (cuantizando el latente) + robustez a un "canal" ruidoso sobre el latente.

Uso:
    python ae_telecom.py --mode quick          # rapido (CPU)
    python ae_telecom.py --mode full           # barrido completo (GPU recomendado)
    python ae_telecom.py --mode full --device cuda --outdir ./results
"""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# 1. Generacion de datos
# --------------------------------------------------------------------------- #
def make_random_bits(n, dim, rng):
    """Bits +/-1 i.i.d. uniformes. Entropia = dim bits. Caso 'incompresible'."""
    b = rng.integers(0, 2, size=(n, dim)).astype(np.float32)
    return 2.0 * b - 1.0  # {0,1} -> {-1,+1}


def make_lowdim_bits(n, dim, k, W, rng):
    """
    Senal estructurada: bits = sign(W @ z), z ~ N(0, I_k), W fija (dim x k).
    La 'fuente' vive en un manifold de dimension ~k -> es COMPRIMIBLE.
    W debe ser la MISMA en train y test (misma fuente, simbolos frescos).
    """
    z = rng.standard_normal(size=(n, k)).astype(np.float32)
    x = np.sign(z @ W.T).astype(np.float32)
    x[x == 0.0] = 1.0
    return x


def build_dataset(regime, n_train, n_test, dim, k, seed):
    rng = np.random.default_rng(seed)
    if regime == "random":
        Xtr = make_random_bits(n_train, dim, rng)
        Xte = make_random_bits(n_test, dim, rng)  # test independiente: nada que memorizar sirve
        info = {"intrinsic_dim": dim}
    elif regime == "lowdim":
        W = rng.standard_normal(size=(dim, k)).astype(np.float32) / np.sqrt(k)
        Xtr = make_lowdim_bits(n_train, dim, k, W, rng)
        Xte = make_lowdim_bits(n_test, dim, k, W, rng)  # simbolos frescos, misma fuente
        info = {"intrinsic_dim": k}
    else:
        raise ValueError(regime)
    return torch.from_numpy(Xtr), torch.from_numpy(Xte), info


# --------------------------------------------------------------------------- #
# 2. Modelos
# --------------------------------------------------------------------------- #
class DeepAE(nn.Module):
    """Autoencoder no lineal. tanh a la salida -> rango [-1, 1] natural para +/-1."""
    def __init__(self, dim, latent, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, dim), nn.Tanh(),
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))


class LinearAE(nn.Module):
    """Autoencoder lineal (sin no linealidad interna) ~ PCA. Salida lineal."""
    def __init__(self, dim, latent):
        super().__init__()
        self.encoder = nn.Linear(dim, latent, bias=False)
        self.decoder = nn.Linear(latent, dim, bias=False)

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))


def build_model(arch, dim, latent, hidden):
    if arch == "deep":
        return DeepAE(dim, latent, hidden)
    if arch == "linear":
        return LinearAE(dim, latent)
    raise ValueError(arch)


# --------------------------------------------------------------------------- #
# 3. Metrica de telecom
# --------------------------------------------------------------------------- #
def ber(recon, x):
    """Bit Error Rate = fraccion de bits donde sign(recon) != bit original."""
    hard = torch.sign(recon)
    hard[hard == 0] = -1  # sign(0) cuenta como error (medida cero en la practica)
    return (hard != x).float().mean().item()


# --------------------------------------------------------------------------- #
# 4. Entrenamiento / evaluacion
# --------------------------------------------------------------------------- #
def train_eval(model, Xtr, Xte, epochs, batch, lr, device, log_every=0):
    model.to(device)
    Xtr, Xte = Xtr.to(device), Xte.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = Xtr.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        run = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.shape[0]
        if log_every and (ep + 1) % log_every == 0:
            print(f"      epoch {ep+1:3d}/{epochs}  train_mse={run/n:.4f}")

    model.eval()
    with torch.no_grad():
        rtr, rte = model(Xtr), model(Xte)
        out = {
            "train_mse": nn.MSELoss()(rtr, Xtr).item(),
            "test_mse":  nn.MSELoss()(rte, Xte).item(),
            "train_ber": ber(rtr, Xtr),
            "test_ber":  ber(rte, Xte),
        }
    return out


# --------------------------------------------------------------------------- #
# 5. Analisis honesto: presupuesto de bits (cuantizar el latente)
# --------------------------------------------------------------------------- #
def quantize_latent_ber(model, Xte, bits, device):
    """
    Cuantiza el latente a 'bits' por dimension (cuantizador uniforme sobre el
    rango observado) y recalcula el BER de test. Asi el latente ocupa
    latent_dim * bits bits REALES (no floats de 32 bits 'gratis').
    """
    model.eval()
    Xte = Xte.to(device)
    with torch.no_grad():
        z = model.encode(Xte)
        zmin, zmax = z.min(0).values, z.max(0).values
        levels = 2 ** bits
        span = (zmax - zmin).clamp_min(1e-8)
        zq = torch.round((z - zmin) / span * (levels - 1))
        zq = zq / (levels - 1) * span + zmin
        rec = model.decode(zq)
        return ber(rec, Xte)


# --------------------------------------------------------------------------- #
# 6. Robustez a canal ruidoso sobre el latente (sabor telecom)
# --------------------------------------------------------------------------- #
def channel_noise_ber(model, Xte, snr_db_list, device, rng):
    """
    Simula transmitir el latente por un canal AWGN: z_rx = z + n.
    Reporta BER vs SNR (dB). (Evaluacion; entrenar con ruido mejoraria esto.)
    """
    model.eval()
    Xte = Xte.to(device)
    results = []
    with torch.no_grad():
        z = model.encode(Xte)
        sig_power = z.pow(2).mean().item()
        for snr_db in snr_db_list:
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = torch.from_numpy(
                rng.standard_normal(size=tuple(z.shape)).astype(np.float32)
            ).to(device) * np.sqrt(noise_power)
            rec = model.decode(z + noise)
            results.append((snr_db, ber(rec, Xte)))
    return results


# --------------------------------------------------------------------------- #
# 7. Orquestacion
# --------------------------------------------------------------------------- #
def run_sweep(cfg):
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["outdir"], exist_ok=True)
    rows = []
    t0 = time.time()

    print(f"\n{'='*74}\nBARRIDO PRINCIPAL  (device={device})\n{'='*74}")
    for regime in cfg["regimes"]:
        Xtr, Xte, info = build_dataset(
            regime, cfg["n_train"], cfg["n_test"], cfg["dim"], cfg["k"], cfg["seed"])
        for arch in cfg["archs"]:
            for latent in cfg["latents"]:
                torch.manual_seed(cfg["seed"])  # misma init por config -> comparable
                model = build_model(arch, cfg["dim"], latent, cfg["hidden"])
                res = train_eval(model, Xtr, Xte, cfg["epochs"], cfg["batch"],
                                 cfg["lr"], device)
                row = {"regime": regime, "arch": arch, "latent": latent,
                       "ratio": latent / cfg["dim"],
                       "intrinsic_dim": info["intrinsic_dim"], **res}
                rows.append(row)
                print(f"  {regime:7s} | {arch:6s} | L={latent:4d} "
                      f"(x{latent/cfg['dim']:.3f}) | "
                      f"test_BER={res['test_ber']:.4f} | "
                      f"train_BER={res['train_ber']:.4f} | "
                      f"test_MSE={res['test_mse']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg["outdir"], "resultados_ae.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[guardado] {csv_path}")

    make_plots(df, cfg)
    bonus_analysis(cfg, df)
    print(f"\nTiempo total: {time.time()-t0:.1f}s")
    return df


def bonus_analysis(cfg, df):
    """Presupuesto de bits + canal ruidoso sobre el mejor modelo estructurado."""
    device = cfg["device"]
    sub = df[(df.regime == "lowdim") & (df.arch == "deep")]
    if sub.empty:
        return
    # elige el latente mas pequeno con test_BER bajo
    good = sub[sub.test_ber < 0.01].sort_values("latent")
    latent = int(good.latent.iloc[0]) if not good.empty else int(sub.latent.min())

    Xtr, Xte, info = build_dataset(
        "lowdim", cfg["n_train"], cfg["n_test"], cfg["dim"], cfg["k"], cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = build_model("deep", cfg["dim"], latent, cfg["hidden"])
    train_eval(model, Xtr, Xte, cfg["epochs"], cfg["batch"], cfg["lr"], device)

    print(f"\n{'='*74}\nANALISIS DE BITS  (lowdim, deep, latent={latent}, "
          f"dim intrinseca={info['intrinsic_dim']})\n{'='*74}")
    print("  Presupuesto honesto: latente cuantizado a b bits/dim")
    print(f"  {'bits/dim':>8} | {'bits totales':>12} | {'ratio vs 500b':>13} | {'test_BER':>9}")
    for b in [32, 8, 4, 2, 1]:
        if b == 32:
            be = float(df[(df.regime=='lowdim')&(df.arch=='deep')&
                          (df.latent==latent)].test_ber.iloc[0])
        else:
            be = quantize_latent_ber(model, Xte, b, device)
        total = latent * b
        print(f"  {b:>8} | {total:>12} | {500/total:>12.2f}x | {be:>9.4f}")

    print("\n  Canal AWGN sobre el latente (transmitir la representacion comprimida):")
    rng = np.random.default_rng(cfg["seed"] + 1)
    for snr_db, be in channel_noise_ber(model, Xte,
                                        [30, 20, 15, 10, 5, 0], device, rng):
        print(f"    SNR={snr_db:>3} dB -> test_BER={be:.4f}")


def make_plots(df, cfg):
    # Plot 1: test BER vs latente, por regimen (arch=deep)
    plt.figure(figsize=(7, 4.5))
    for regime in cfg["regimes"]:
        d = df[(df.regime == regime) & (df.arch == "deep")].sort_values("latent")
        if not d.empty:
            plt.plot(d.latent, d.test_ber, "o-", label=f"{regime} (deep)")
    plt.axhline(0.5, ls="--", c="grey", lw=1, label="0.5 (azar)")
    plt.axhline(1e-2, ls=":", c="green", lw=1, label="1e-2 (corregible con FEC)")
    plt.xlabel("Dimension latente")
    plt.ylabel("BER (test)")
    plt.title("BER de test vs compresion  (dim entrada = 500)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p1 = os.path.join(cfg["outdir"], "ber_vs_latente.png")
    plt.savefig(p1, dpi=130); plt.close()

    # Plot 2: train vs test BER en 'random' (brecha de generalizacion)
    d = df[(df.regime == "random") & (df.arch == "deep")].sort_values("latent")
    if not d.empty:
        plt.figure(figsize=(7, 4.5))
        plt.plot(d.latent, d.train_ber, "o-", label="train BER")
        plt.plot(d.latent, d.test_ber, "s-", label="test BER")
        plt.axhline(0.5, ls="--", c="grey", lw=1)
        plt.xlabel("Dimension latente"); plt.ylabel("BER")
        plt.title("Bits aleatorios: brecha train/test (limite de Shannon)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        p2 = os.path.join(cfg["outdir"], "random_train_vs_test.png")
        plt.savefig(p2, dpi=130); plt.close()

    # Plot 3: lineal vs profundo
    plt.figure(figsize=(7, 4.5))
    for regime in cfg["regimes"]:
        for arch in cfg["archs"]:
            d = df[(df.regime == regime) & (df.arch == arch)].sort_values("latent")
            if not d.empty:
                ls = "o-" if arch == "deep" else "s--"
                plt.plot(d.latent, d.test_ber, ls, label=f"{regime}/{arch}")
    plt.xlabel("Dimension latente"); plt.ylabel("BER (test)")
    plt.title("Lineal (~PCA) vs Profundo")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p3 = os.path.join(cfg["outdir"], "lineal_vs_profundo.png")
    plt.savefig(p3, dpi=130); plt.close()
    print(f"[guardado] {p1}\n[guardado] {p3}")


# --------------------------------------------------------------------------- #
# 8. Configuraciones
# --------------------------------------------------------------------------- #
def get_cfg(args):
    base = dict(
        dim=500, k=32, hidden=256, lr=2e-3,
        seed=0, device=args.device, outdir=args.outdir,
        regimes=["random", "lowdim"], archs=["deep", "linear"],
    )
    if args.mode == "quick":
        base.update(n_train=16000, n_test=4000, epochs=40, batch=512,
                    latents=[500, 200, 100, 50, 20, 8])
    else:  # full
        base.update(n_train=100000, n_test=20000, epochs=150, batch=1024,
                    latents=[500, 400, 300, 250, 200, 150, 100, 64, 50, 40,
                             32, 24, 16, 12, 8, 4, 2])
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="./ae_results")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = get_cfg(args)
    print("CONFIG:", json.dumps({k: v for k, v in cfg.items()
                                 if k not in ("device",)}, default=str))
    run_sweep(cfg)


if __name__ == "__main__":
    main()
