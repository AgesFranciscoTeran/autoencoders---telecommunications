#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AUTOENCODERS PARA SENALES BPSK (+/-1, dim 500)  --  HARNESS v2
================================================================================
Que cambia respecto a v1 (y por que importa para el tutor):

 (1) LATENTE BINARIO (straight-through estimator).
     En v1 el latente era float32: un latente de dim 50 ocupaba 1600 bits, mas
     que los 500 originales -> no era compresion. Aqui el cuello de botella
     emite bits reales, asi que la TASA es exacta por construccion:
         R = L / 500   [bits de canal por bit de fuente]

 (2) PERDIDA ALINEADA AL BER.
     v1 usaba MSE+tanh. La decision por bit es una clasificacion binaria:
     BCE-con-logits optimiza directamente la probabilidad de acertar el signo.
     Se incluye la ablacion MSE vs BCE para documentarlo.

 (3) COTAS TEORICAS EXPLICITAS (la vara de medir).
     - Fuente i.i.d. Bernoulli(1/2), distorsion de Hamming:
           R(D) = 1 - H_b(D)     =>   D_min(R) = H_b^{-1}(1 - R)
       Ningun compresor (ni el AE) puede quedar por debajo de esa curva.
     - Fuentes con estructura: entropia real de la fuente en bits (cota para
       compresion sin perdida).

 (4) CUATRO FUENTES, de "imposible" a "muy comprimible":
     random  : bits i.i.d.            H = 500 bits
     lowdim  : sign(W z), z en R^k    H = log2(#regiones del arreglo) ~ 170 bits
     code    : codigo lineal GF(2)    H = 250 bits exactos (tasa 1/2)
     markov  : fuente correlacionada  H = 1 + 499*H_b(p) bits

 (5) JSCC: el latente binario ES una senal BPSK -> se transmite por AWGN con
     presupuesto de energia JUSTO y se compara contra BPSK sin codificar.
     Con Eb = energia por bit de FUENTE y L simbolos de canal:
         Es = 500*Eb/L    =>    sigma^2 = (L/500) / (2*Eb/N0)
     En L=500 esto recupera exactamente BER = Q(sqrt(2 Eb/N0)).

Uso:
    python3 ae_telecom_v2.py --stage all   --mode quick
    python3 ae_telecom_v2.py --stage all   --mode full --device cuda
    python3 ae_telecom_v2.py --stage sweep --mode full --device cuda --outdir ./res
================================================================================
"""

import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================== #
# 0. TEORIA: entropias, cota tasa-distorsion, Q(x)
# =========================================================================== #
def Hb(p):
    """Entropia binaria en bits."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def Hb_inv(h):
    """Inversa de H_b en la rama [0, 0.5] (biseccion)."""
    h = np.clip(h, 0.0, 1.0)
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Hb(mid) < h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rd_bound_ber(rate):
    """
    BER minimo alcanzable por CUALQUIER esquema a tasa 'rate' (bits/bit fuente)
    para fuente Bernoulli(1/2) con distorsion de Hamming: 1 - H_b(D) = R.
    """
    rate = np.asarray(rate, dtype=np.float64)
    return np.array([Hb_inv(1.0 - r) if r < 1.0 else 0.0 for r in rate.ravel()]).reshape(rate.shape)


def qfunc(x):
    """Q(x) = P(N(0,1) > x)."""
    return 0.5 * torch.erfc(torch.as_tensor(x, dtype=torch.float64) / math.sqrt(2.0)).numpy()


def log2_regions_hyperplane_arrangement(n, k):
    """
    log2 del numero de celdas de un arreglo de n hiperplanos por el origen en
    R^k en posicion general:  C(n,k) = 2 * sum_{i=0}^{k-1} binom(n-1, i).
    Es la ENTROPIA de la fuente 'lowdim' (cuantos patrones de signo distintos
    existen). Ojo: NO es k. La dimension del manifold es k; la entropia es esto.
    """
    logs = []
    for i in range(k):
        logs.append(math.lgamma(n) - math.lgamma(i + 1) - math.lgamma(n - i))
    m = max(logs)
    tot = m + math.log(sum(math.exp(l - m) for l in logs))
    return (tot + math.log(2.0)) / math.log(2.0)


# =========================================================================== #
# 1. FUENTES
# =========================================================================== #
def src_random(n, dim, rng, _p=None):
    b = rng.integers(0, 2, size=(n, dim)).astype(np.float32)
    return 2.0 * b - 1.0


def src_lowdim(n, dim, k, W, rng):
    z = rng.standard_normal(size=(n, k)).astype(np.float32)
    x = np.sign(z @ W.T).astype(np.float32)
    x[x == 0.0] = 1.0
    return x


def src_code(n, dim, P, rng):
    """
    Codigo lineal binario sistematico de tasa 1/2: x = [u | u P] sobre GF(2),
    con exactamente 3 unos por columna de P (checks de grado 3 -> aprendibles).
    En el dominio +/-1 la paridad es un PRODUCTO de simbolos -> no lineal en R.
    """
    k = dim // 2
    u = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    par = (u.astype(np.int32) @ P) % 2
    bits = np.concatenate([u, par.astype(np.int8)], axis=1).astype(np.float32)
    return 2.0 * bits - 1.0


def src_markov(n, dim, p, rng):
    """Cadena de Markov binaria simetrica: flip con prob p. H = 1 + (dim-1)*H_b(p)."""
    flips = (rng.random(size=(n, dim)) < p).astype(np.int8)
    flips[:, 0] = rng.integers(0, 2, size=n).astype(np.int8)
    bits = np.cumsum(flips, axis=1) % 2
    return (2.0 * bits - 1.0).astype(np.float32)


def build_dataset(regime, n_train, n_test, dim, cfg, seed):
    """Devuelve (Xtr, Xte, meta). La 'fuente' (W, P) es la MISMA en train y test."""
    rng = np.random.default_rng(seed)
    if regime == "random":
        Xtr, Xte = src_random(n_train, dim, rng), src_random(n_test, dim, rng)
        meta = {"entropy_bits": float(dim), "manifold_dim": dim}
    elif regime == "lowdim":
        k = cfg["k"]
        W = rng.standard_normal(size=(dim, k)).astype(np.float32) / np.sqrt(k)
        Xtr = src_lowdim(n_train, dim, k, W, rng)
        Xte = src_lowdim(n_test, dim, k, W, rng)
        meta = {"entropy_bits": log2_regions_hyperplane_arrangement(dim, k),
                "manifold_dim": k}
    elif regime == "code":
        k = dim // 2
        P = np.zeros((k, k), dtype=np.int32)
        for j in range(k):
            P[rng.choice(k, size=3, replace=False), j] = 1
        Xtr, Xte = src_code(n_train, dim, P, rng), src_code(n_test, dim, P, rng)
        meta = {"entropy_bits": float(k), "manifold_dim": k}
    elif regime == "markov":
        p = cfg["markov_p"]
        Xtr = src_markov(n_train, dim, p, rng)
        Xte = src_markov(n_test, dim, p, rng)
        meta = {"entropy_bits": float(1 + (dim - 1) * Hb(p)), "manifold_dim": None}
    else:
        raise ValueError(regime)
    return torch.from_numpy(Xtr), torch.from_numpy(Xte), meta


# =========================================================================== #
# 2. MODELOS
# =========================================================================== #
class BinarizeSTE(torch.autograd.Function):
    """sign(u) hacia adelante, identidad hacia atras (straight-through)."""
    @staticmethod
    def forward(ctx, u):
        s = torch.sign(u)
        s[s == 0] = 1.0
        return s

    @staticmethod
    def backward(ctx, g):
        return g


class AE(nn.Module):
    """
    Encoder 500 -> h -> h -> L   (cuello de botella: continuo o binario)
    Decoder L   -> h -> h -> 500 (LOGITS; el signo del logit es la decision)
    """
    def __init__(self, dim, latent, hidden=512, arch="deep", latent_type="binary"):
        super().__init__()
        self.latent_type, self.arch, self.latent = latent_type, arch, latent
        if arch == "deep":
            self.enc = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, latent))
            self.dec = nn.Sequential(
                nn.Linear(latent, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, dim))
        elif arch == "linear":
            self.enc = nn.Linear(dim, latent, bias=False)
            self.dec = nn.Linear(latent, dim, bias=False)
        else:
            raise ValueError(arch)

    def encode(self, x):
        pre = self.enc(x)
        if self.latent_type == "binary":
            u = torch.tanh(pre)
            return BinarizeSTE.apply(u)      # +/-1 exactos (BPSK)
        return pre

    def decode(self, z):
        return self.dec(z)                   # logits

    def forward(self, x, noise_std=0.0):
        z = self.encode(x)
        if noise_std > 0:
            z = z + noise_std * torch.randn_like(z)
        return self.decode(z)


# =========================================================================== #
# 3. METRICA Y PERDIDAS
# =========================================================================== #
def ber_from_logits(logits, x):
    """BER = fraccion de bits donde sign(salida) != bit original."""
    hard = torch.sign(logits)
    hard[hard == 0] = -1.0
    return (hard != x).float().mean().item()


def loss_fn(kind, out, x):
    if kind == "bce":
        target = (x + 1.0) * 0.5                      # {-1,+1} -> {0,1}
        return F.binary_cross_entropy_with_logits(out, target)
    if kind == "mse":
        return F.mse_loss(torch.tanh(out), x)         # tanh para acotar a [-1,1]
    raise ValueError(kind)


# =========================================================================== #
# 4. ENTRENAMIENTO
# =========================================================================== #
def sigma_from_ebn0(ebn0_db, latent, dim):
    """
    Presupuesto de energia JUSTO: Eb = energia por bit de FUENTE.
    Es = dim*Eb/L  ->  sigma^2 = (L/dim) / (2*Eb/N0).
    """
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return math.sqrt((latent / dim) / (2.0 * ebn0))


def train_model(model, Xtr, Xte, cfg, loss_kind="bce", train_ebn0_db=None, verbose=False):
    dev = cfg["device"]
    model.to(dev)
    Xtr, Xte = Xtr.to(dev), Xte.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n, bs = Xtr.shape[0], cfg["batch"]

    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        run = 0.0
        for i in range(0, n, bs):
            xb = Xtr[perm[i:i + bs]]
            if train_ebn0_db is None:
                ns = 0.0
            else:  # entrenamiento consciente del canal: Eb/N0 aleatorio +/-3 dB
                ns = sigma_from_ebn0(train_ebn0_db + float(np.random.uniform(-3, 3)),
                                     model.latent, Xtr.shape[1])
            out = model(xb, noise_std=ns)
            loss = loss_fn(loss_kind, out, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * xb.shape[0]
        sch.step()
        if verbose and (ep + 1) % max(1, cfg["epochs"] // 5) == 0:
            print(f"      ep {ep+1:3d}/{cfg['epochs']}  loss={run/n:.4f}")

    model.eval()
    with torch.no_grad():
        ns = 0.0 if train_ebn0_db is None else sigma_from_ebn0(train_ebn0_db, model.latent, Xtr.shape[1])
        res = {"train_ber": ber_from_logits(model(Xtr, ns), Xtr),
               "test_ber":  ber_from_logits(model(Xte, ns), Xte)}
    return res


@torch.no_grad()
def eval_ebn0_curve(model, Xte, ebn0_list, dim, device, reps=3):
    """BER de test vs Eb/N0 transmitiendo el latente binario por AWGN."""
    model.eval(); Xte = Xte.to(device)
    out = []
    for db in ebn0_list:
        s = sigma_from_ebn0(db, model.latent, dim)
        vals = [ber_from_logits(model(Xte, noise_std=s), Xte) for _ in range(reps)]
        out.append((db, float(np.mean(vals))))
    return out


# =========================================================================== #
# 5. ETAPAS EXPERIMENTALES
# =========================================================================== #
def stage_ablation(cfg):
    """MSE vs BCE  x  latente continuo vs binario. Justifica las decisiones."""
    print(f"\n{'='*76}\nETAPA A - ABLACION: perdida y tipo de latente\n{'='*76}")
    rows = []
    for regime in ["lowdim", "code"]:
        Xtr, Xte, meta = build_dataset(regime, cfg["n_train"], cfg["n_test"],
                                       cfg["dim"], cfg, cfg["seed"])
        for lt in ["continuous", "binary"]:
            for lk in ["mse", "bce"]:
                torch.manual_seed(cfg["seed"])
                m = AE(cfg["dim"], cfg["abl_latent"], cfg["hidden"], "deep", lt)
                r = train_model(m, Xtr, Xte, cfg, loss_kind=lk)
                bits = cfg["abl_latent"] * (1 if lt == "binary" else 32)
                rows.append(dict(regime=regime, latent_type=lt, loss=lk,
                                 latent=cfg["abl_latent"], bits_reales=bits,
                                 compresion=cfg["dim"] / bits, **r))
                print(f"  {regime:7s} | {lt:10s} | {lk:3s} | bits={bits:5d} "
                      f"({cfg['dim']/bits:5.2f}x) | test_BER={r['test_ber']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v2_ablacion.csv"), index=False)
    return df


def stage_sweep(cfg):
    """Barrido principal con latente BINARIO: BER vs tasa real, 4 fuentes."""
    print(f"\n{'='*76}\nETAPA B - BARRIDO: BER vs TASA REAL (latente binario)\n{'='*76}")
    rows = []
    for regime in cfg["regimes"]:
        Xtr, Xte, meta = build_dataset(regime, cfg["n_train"], cfg["n_test"],
                                       cfg["dim"], cfg, cfg["seed"])
        print(f"\n  --- fuente '{regime}': entropia = {meta['entropy_bits']:.1f} bits "
              f"de {cfg['dim']} (tasa minima sin perdida = "
              f"{meta['entropy_bits']/cfg['dim']:.3f} bits/bit) ---")
        for arch in cfg["archs"]:
            for L in cfg["latents"]:
                torch.manual_seed(cfg["seed"])
                m = AE(cfg["dim"], L, cfg["hidden"], arch, "binary")
                r = train_model(m, Xtr, Xte, cfg, loss_kind="bce")
                rate = L / cfg["dim"]
                rows.append(dict(regime=regime, arch=arch, latent_bits=L, rate=rate,
                                 compresion=cfg["dim"] / L,
                                 entropy_bits=meta["entropy_bits"],
                                 rd_bound_ber=float(rd_bound_ber(np.array([rate]))[0]),
                                 **r))
                print(f"    {arch:6s} | L={L:4d} bits ({cfg['dim']/L:5.2f}x) "
                      f"| R={rate:.3f} | test_BER={r['test_ber']:.4f} "
                      f"| train_BER={r['train_ber']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v2_barrido_binario.csv"), index=False)
    return df


def stage_jscc(cfg):
    """
    JSCC: transmitir el latente binario por AWGN con energia justa y comparar
    contra BPSK sin codificar (que es el caso L=500, R=1).
    """
    print(f"\n{'='*76}\nETAPA C - JSCC: BER vs Eb/N0 sobre canal AWGN\n{'='*76}")
    grid = cfg["ebn0_grid"]
    rows = []
    for regime in cfg["jscc_regimes"]:
        Xtr, Xte, meta = build_dataset(regime, cfg["n_train"], cfg["n_test"],
                                       cfg["dim"], cfg, cfg["seed"])
        for L in cfg["jscc_latents"]:
            for aware in [False, True]:
                torch.manual_seed(cfg["seed"])
                m = AE(cfg["dim"], L, cfg["hidden"], "deep", "binary")
                train_model(m, Xtr, Xte, cfg, loss_kind="bce",
                            train_ebn0_db=(cfg["train_ebn0"] if aware else None))
                curve = eval_ebn0_curve(m, Xte, grid, cfg["dim"], cfg["device"])
                for db, be in curve:
                    rows.append(dict(regime=regime, latent_bits=L,
                                     rate=L / cfg["dim"],
                                     canal_en_entrenamiento=aware,
                                     ebn0_db=db, test_ber=be))
                tag = "canal-aware" if aware else "sin canal"
                s = "  ".join(f"{db}dB:{be:.4f}" for db, be in curve)
                print(f"  {regime:7s} L={L:3d} ({tag:11s}) -> {s}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v2_jscc.csv"), index=False)
    return df


# =========================================================================== #
# 6. GRAFICOS
# =========================================================================== #
def plot_sweep(df, cfg):
    d = df[df.arch == "deep"]
    if d.empty:
        return
    plt.figure(figsize=(7.6, 5.0))
    colors = {"random": "tab:blue", "lowdim": "tab:orange",
              "code": "tab:green", "markov": "tab:red"}
    for reg, g in d.groupby("regime"):
        g = g.sort_values("rate")
        plt.plot(g.rate, g.test_ber, "o-", color=colors.get(reg), label=f"AE {reg}")
        h = g.entropy_bits.iloc[0] / cfg["dim"]
        plt.axvline(h, ls=":", lw=1.2, color=colors.get(reg), alpha=0.8)
    rr = np.linspace(0.01, 1.0, 200)
    plt.plot(rr, rd_bound_ber(rr), "k--", lw=1.6,
             label="cota R(D)=1-H(D)  [i.i.d.]")
    plt.axhline(1e-2, ls=":", c="grey", lw=1)
    plt.text(0.62, 1.15e-2, "1e-2: corregible con FEC", fontsize=8, color="grey")
    plt.xlabel("Tasa R = bits del latente / 500 bits de fuente")
    plt.ylabel("BER (test)")
    plt.title("BER vs TASA REAL (latente binario) + cotas de entropia\n"
              "lineas verticales punteadas = entropia de cada fuente")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v2_ber_vs_tasa.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")

    # random: AE vs cota tasa-distorsion (que tan lejos del optimo)
    r = df[(df.regime == "random") & (df.arch == "deep")].sort_values("rate")
    if not r.empty:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r.rate, r.test_ber, "o-", label="AE (latente binario)")
        plt.plot(r.rate, r.rd_bound_ber, "k--", label="optimo teorico R(D)")
        plt.plot(r.rate, r.train_ber, "s:", alpha=0.7, label="AE train (memoriza)")
        plt.xlabel("Tasa R"); plt.ylabel("BER")
        plt.title("Fuente incompresible: el AE frente al limite de Shannon")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        p = os.path.join(cfg["outdir"], "v2_random_vs_shannon.png")
        plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")

    # lineal vs profundo
    plt.figure(figsize=(7.2, 4.6))
    for reg, g0 in df.groupby("regime"):
        for arch, g in g0.groupby("arch"):
            g = g.sort_values("rate")
            plt.plot(g.rate, g.test_ber, "o-" if arch == "deep" else "s--",
                     color=colors.get(reg), alpha=1.0 if arch == "deep" else 0.55,
                     label=f"{reg}/{arch}")
    plt.xlabel("Tasa R"); plt.ylabel("BER (test)")
    plt.title("Lineal (~PCA) vs Profundo, con latente binario")
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v2_lineal_vs_profundo.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")


def plot_jscc(df, cfg):
    if df.empty:
        return
    plt.figure(figsize=(7.6, 5.0))
    grid = np.array(cfg["ebn0_grid"], dtype=float)
    plt.semilogy(grid, np.maximum(qfunc(np.sqrt(2 * 10 ** (grid / 10))), 1e-6),
                 "k--", lw=1.8, label="BPSK sin codificar (R=1)")
    mk = {True: "o-", False: "s:"}
    for (reg, L, aw), g in df.groupby(["regime", "latent_bits", "canal_en_entrenamiento"]):
        g = g.sort_values("ebn0_db")
        plt.semilogy(g.ebn0_db, np.maximum(g.test_ber, 1e-6), mk[aw],
                     label=f"{reg} L={L} ({'canal-aware' if aw else 'sin canal'})",
                     alpha=1.0 if aw else 0.6)
    plt.xlabel("Eb/N0 [dB]  (Eb = energia por bit de FUENTE)")
    plt.ylabel("BER (test)")
    plt.title("Joint source-channel coding: latente binario sobre AWGN\n"
              "presupuesto de energia identico para todas las curvas")
    plt.legend(fontsize=7.5); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v2_jscc_ebn0.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")


def plot_ablation(df, cfg):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, reg in zip(axes, df.regime.unique()):
        d = df[df.regime == reg]
        labels = [f"{r.latent_type[:4]}\n{r.loss}" for r in d.itertuples()]
        ax.bar(labels, d.test_ber, color=["#888", "#888", "#2b7", "#2b7"])
        for i, r in enumerate(d.itertuples()):
            ax.text(i, r.test_ber + 0.004, f"{r.bits_reales}b", ha="center", fontsize=8)
        ax.set_title(f"fuente: {reg}"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("BER (test)")
    fig.suptitle(f"Ablacion: tipo de latente x perdida (L={cfg['abl_latent']}); "
                 "la etiqueta es el coste en bits REALES")
    plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v2_ablacion.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")


# =========================================================================== #
# 7. CONFIG / MAIN
# =========================================================================== #
def get_cfg(a):
    c = dict(dim=500, k=32, markov_p=0.05, hidden=512, lr=2e-3, seed=0,
             device=a.device, outdir=a.outdir,
             regimes=["random", "lowdim", "code", "markov"],
             archs=["deep", "linear"],
             jscc_regimes=["markov", "random"],
             ebn0_grid=[-2, 0, 2, 4, 6, 8, 10], train_ebn0=4.0)
    if a.mode == "quick":
        c.update(n_train=16000, n_test=5000, epochs=40, batch=1024, hidden=384,
                 latents=[250, 128, 64, 32], abl_latent=128,
                 jscc_latents=[128], archs=["deep", "linear"])
    else:
        c.update(n_train=200000, n_test=40000, epochs=200, batch=1024, hidden=1024,
                 latents=[400, 300, 250, 200, 150, 128, 100, 64, 48, 32, 16, 8],
                 abl_latent=128, jscc_latents=[250, 128, 64])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--stage", choices=["all", "ablation", "sweep", "jscc"], default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="./ae_results_v2")
    a = ap.parse_args()
    cfg = get_cfg(a)
    os.makedirs(cfg["outdir"], exist_ok=True)
    if a.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    print("CONFIG:", json.dumps(cfg, default=str))

    print(f"\nENTROPIA DE CADA FUENTE (dim={cfg['dim']}):")
    for reg in ["random", "lowdim", "code", "markov"]:
        _, _, m = build_dataset(reg, 8, 8, cfg["dim"], cfg, cfg["seed"])
        print(f"  {reg:7s}: H = {m['entropy_bits']:7.1f} bits  -> tasa minima "
              f"sin perdida = {m['entropy_bits']/cfg['dim']:.3f} bits/bit  "
              f"(compresion max {cfg['dim']/m['entropy_bits']:.2f}x)")

    t0 = time.time()
    if a.stage in ("all", "ablation"):
        plot_ablation(stage_ablation(cfg), cfg)
    if a.stage in ("all", "sweep"):
        plot_sweep(stage_sweep(cfg), cfg)
    if a.stage in ("all", "jscc"):
        plot_jscc(stage_jscc(cfg), cfg)
    print(f"\nTiempo total: {time.time()-t0:.1f}s   (salidas en {cfg['outdir']})")


if __name__ == "__main__":
    main()
