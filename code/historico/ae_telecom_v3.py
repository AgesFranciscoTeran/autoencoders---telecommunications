#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AUTOENCODERS PARA SENALES BPSK (+/-1, dim 500)  --  HARNESS v3
================================================================================
v3 = v2 + lo que faltaba para que el resultado sea defendible:

 (D) BASELINES CLASICOS  <-- lo mas importante que faltaba.
     Un AE solo "sirve" si le gana a algo simple. Se comparan, en el MISMO eje
     BER-vs-tasa que el AE:
       * decimacion + hold      (markov): manda 1 de cada m bits, sostiene.
       * oraculo sistematico    (code)  : manda los 250 bits de informacion.
       * cuantizacion de z      (lowdim): el codificador conoce W y manda z.
       * PCA + cuantizacion     (todas) : codificacion por transformada clasica.
         ^ este es el competidor justo del AE: sin oraculo, aprendido de datos.

 (E) DIAGNOSTICO DE PARIDAD.
     En v2 la fuente 'code' (250 bits de entropia, comprimible 2x) dio la MISMA
     curva que ruido puro. Hipotesis: no es falta de capacidad, es que SGD no
     aprende XOR de grado >= 3. Se testea en aislamiento: entrenar un MLP para
     predecir el producto de d simbolos de un subconjunto fijo, d = 1..4.
     Si falla en d=3 con datos de sobra, la hipotesis queda demostrada y explica
     por que el AE no ve la redundancia algebraica.

 (F) JSCC CORREGIDO.
     En v2 la curva BER-vs-Eb/N0 salio plana: el piso de distorsion de la
     compresion tapaba por completo el ruido del canal. Se arregla usando
     latentes chicos + Eb/N0 hasta -6 dB, y se reporta el EXCESO de BER sobre
     el piso sin ruido, que es lo que realmente mide el efecto del canal.

Cotas teoricas incluidas:
   * fuente i.i.d.: R(D) = 1 - H_b(D)                    (exacta)
   * fuente markov: R(D) >= H_b(p) - H_b(D)              (Shannon Lower Bound)

Uso:
    python3 ae_telecom_v3.py --stage all --mode full --device cuda --outdir ./res
    python3 ae_telecom_v3.py --stage baselines,parity --mode quick   # sin GPU
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
# 0. TEORIA
# =========================================================================== #
def Hb(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def Hb_inv(h):
    """Inversa de H_b en la rama [0, 0.5]."""
    h = float(np.clip(h, 0.0, 1.0))
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Hb(mid) < h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rd_bound_iid(rate):
    """BER minimo a tasa R para fuente Bernoulli(1/2): 1 - H_b(D) = R."""
    r = np.atleast_1d(np.asarray(rate, dtype=np.float64))
    return np.array([Hb_inv(1.0 - x) if x < 1.0 else 0.0 for x in r]).reshape(np.shape(rate))


def rd_bound_markov(rate, p):
    """Shannon Lower Bound: R >= H_b(p) - H_b(D)  =>  D >= H_b^{-1}(H_b(p) - R)."""
    r = np.atleast_1d(np.asarray(rate, dtype=np.float64))
    hp = float(Hb(p))
    return np.array([Hb_inv(max(0.0, hp - x)) for x in r]).reshape(np.shape(rate))


def qfunc(x):
    return 0.5 * torch.erfc(torch.as_tensor(x, dtype=torch.float64) / math.sqrt(2.0)).numpy()


def log2_regions(n, k):
    """log2 de C(n,k) = 2*sum_{i<k} binom(n-1,i): entropia de la fuente lowdim."""
    logs = [math.lgamma(n) - math.lgamma(i + 1) - math.lgamma(n - i) for i in range(k)]
    mx = max(logs)
    tot = mx + math.log(sum(math.exp(l - mx) for l in logs))
    return (tot + math.log(2.0)) / math.log(2.0)


# =========================================================================== #
# 1. FUENTES
# =========================================================================== #
def src_random(n, dim, rng):
    return 2.0 * rng.integers(0, 2, size=(n, dim)).astype(np.float32) - 1.0


def src_lowdim(n, dim, k, W, rng):
    z = rng.standard_normal(size=(n, k)).astype(np.float32)
    x = np.sign(z @ W.T).astype(np.float32)
    x[x == 0.0] = 1.0
    return x, z


def src_code(n, dim, P, rng):
    """Codigo lineal sistematico tasa 1/2 sobre GF(2), checks de grado 3."""
    k = dim // 2
    u = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    par = (u.astype(np.int32) @ P) % 2
    bits = np.concatenate([u, par.astype(np.int8)], axis=1).astype(np.float32)
    return 2.0 * bits - 1.0


def src_markov(n, dim, p, rng):
    flips = (rng.random(size=(n, dim)) < p).astype(np.int8)
    flips[:, 0] = rng.integers(0, 2, size=n).astype(np.int8)
    return (2.0 * (np.cumsum(flips, axis=1) % 2) - 1.0).astype(np.float32)


def build_dataset(regime, n_train, n_test, dim, cfg, seed, return_side=False):
    """La fuente (W, P) es la MISMA en train y test; los simbolos son frescos."""
    rng = np.random.default_rng(seed)
    side = {}
    if regime == "random":
        Xtr, Xte = src_random(n_train, dim, rng), src_random(n_test, dim, rng)
        meta = {"entropy_bits": float(dim), "manifold_dim": dim}
    elif regime == "lowdim":
        k = cfg["k"]
        W = rng.standard_normal(size=(dim, k)).astype(np.float32) / np.sqrt(k)
        Xtr, ztr = src_lowdim(n_train, dim, k, W, rng)
        Xte, zte = src_lowdim(n_test, dim, k, W, rng)
        meta = {"entropy_bits": log2_regions(dim, k), "manifold_dim": k}
        side = {"W": W, "z_test": zte, "z_train": ztr}
    elif regime == "code":
        k = dim // 2
        P = np.zeros((k, k), dtype=np.int32)
        for j in range(k):
            P[rng.choice(k, size=3, replace=False), j] = 1
        Xtr, Xte = src_code(n_train, dim, P, rng), src_code(n_test, dim, P, rng)
        meta = {"entropy_bits": float(k), "manifold_dim": k}
        side = {"P": P}
    elif regime == "markov":
        p = cfg["markov_p"]
        Xtr, Xte = src_markov(n_train, dim, p, rng), src_markov(n_test, dim, p, rng)
        meta = {"entropy_bits": float(1 + (dim - 1) * Hb(p)), "manifold_dim": None}
    else:
        raise ValueError(regime)
    out = (torch.from_numpy(Xtr), torch.from_numpy(Xte), meta)
    return out + (side,) if return_side else out


# =========================================================================== #
# 2. MODELOS
# =========================================================================== #
class BinarizeSTE(torch.autograd.Function):
    """sign(u) hacia adelante, identidad hacia atras."""
    @staticmethod
    def forward(ctx, u):
        s = torch.sign(u); s[s == 0] = 1.0
        return s

    @staticmethod
    def backward(ctx, g):
        return g


class AE(nn.Module):
    """Encoder 500->h->h->L (latente binario = BPSK); decoder devuelve LOGITS."""
    def __init__(self, dim, latent, hidden=512, arch="deep", latent_type="binary"):
        super().__init__()
        self.latent_type, self.arch, self.latent = latent_type, arch, latent
        if arch == "deep":
            self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, latent))
            self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, dim))
        elif arch == "linear":
            self.enc = nn.Linear(dim, latent, bias=False)
            self.dec = nn.Linear(latent, dim, bias=False)
        else:
            raise ValueError(arch)

    def encode(self, x):
        pre = self.enc(x)
        return BinarizeSTE.apply(torch.tanh(pre)) if self.latent_type == "binary" else pre

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, noise_std=0.0):
        z = self.encode(x)
        if noise_std > 0:
            z = z + noise_std * torch.randn_like(z)
        return self.decode(z)


def ber_from_logits(logits, x):
    hard = torch.sign(logits); hard[hard == 0] = -1.0
    return (hard != x).float().mean().item()


def ber_np(rec_bits, x_bits):
    return float((np.sign(rec_bits) != np.sign(x_bits)).mean())


def loss_fn(kind, out, x):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(out, (x + 1.0) * 0.5)
    if kind == "mse":
        return F.mse_loss(torch.tanh(out), x)
    raise ValueError(kind)


# =========================================================================== #
# 3. ENTRENAMIENTO
# =========================================================================== #
def sigma_from_ebn0(ebn0_db, latent, dim):
    """Eb = energia por bit de FUENTE => sigma^2 = (L/dim) / (2 Eb/N0)."""
    return math.sqrt((latent / dim) / (2.0 * 10.0 ** (ebn0_db / 10.0)))


def train_model(model, Xtr, Xte, cfg, loss_kind="bce", train_ebn0_db=None, verbose=False):
    dev = cfg["device"]
    model.to(dev); Xtr, Xte = Xtr.to(dev), Xte.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n, bs, dim = Xtr.shape[0], cfg["batch"], Xtr.shape[1]

    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            xb = Xtr[perm[i:i + bs]]
            ns = 0.0 if train_ebn0_db is None else sigma_from_ebn0(
                train_ebn0_db + float(np.random.uniform(-3, 3)), model.latent, dim)
            loss = loss_fn(loss_kind, model(xb, noise_std=ns), xb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if verbose and (ep + 1) % max(1, cfg["epochs"] // 5) == 0:
            print(f"      ep {ep+1}/{cfg['epochs']}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        ns = 0.0 if train_ebn0_db is None else sigma_from_ebn0(train_ebn0_db, model.latent, dim)
        return {"train_ber": ber_from_logits(model(Xtr, ns), Xtr),
                "test_ber": ber_from_logits(model(Xte, ns), Xte)}


@torch.no_grad()
def eval_ebn0_curve(model, Xte, ebn0_list, dim, device, reps=3):
    model.eval(); Xte = Xte.to(device)
    out = []
    for db in ebn0_list:
        s = sigma_from_ebn0(db, model.latent, dim)
        out.append((db, float(np.mean([ber_from_logits(model(Xte, s), Xte) for _ in range(reps)]))))
    return out


# =========================================================================== #
# 4. BASELINES CLASICOS   (ETAPA D)
# =========================================================================== #
def baseline_decimacion(X, m):
    """
    Manda 1 de cada m bits; reconstruye cada bit con el transmitido MAS CERCANO.
    Tasa = ceil(dim/m)/dim. Optimo para fuentes muy correlacionadas.
    """
    X = X.numpy() if torch.is_tensor(X) else X
    n, dim = X.shape
    keep = np.arange(0, dim, m)
    idx = keep[np.argmin(np.abs(np.arange(dim)[:, None] - keep[None, :]), axis=1)]
    rec = X[:, idx]
    return {"rate": len(keep) / dim, "ber": ber_np(rec, X), "bits": len(keep)}


def baseline_oraculo_code(X, P, frac):
    """
    Oraculo del codigo: manda los primeros frac*k bits sistematicos y RECALCULA
    las paridades que puede; el resto se adivina. Con frac=1 -> BER exacto 0.
    """
    X = X.numpy() if torch.is_tensor(X) else X
    n, dim = X.shape
    k = dim // 2
    b = ((X + 1) / 2).astype(np.int32)
    n_send = int(round(frac * k))
    u_hat = np.zeros((n, k), dtype=np.int32)
    u_hat[:, :n_send] = b[:, :n_send]          # lo conocido
    u_hat[:, n_send:] = 0                       # lo desconocido: adivina 0
    par_hat = (u_hat @ P) % 2
    rec = 2.0 * np.concatenate([u_hat, par_hat], axis=1) - 1.0
    return {"rate": n_send / dim, "ber": ber_np(rec, X), "bits": n_send}


def baseline_oraculo_lowdim(z, W, b_bits):
    """
    Oraculo del manifold: el codificador conoce W y transmite z cuantizado a
    b_bits por dimension. Tasa = k*b/dim. Es el 'techo' de lo alcanzable.
    """
    k = z.shape[1]
    dim = W.shape[0]
    lo, hi = np.percentile(z, 0.5, axis=0), np.percentile(z, 99.5, axis=0)
    lev = 2 ** b_bits
    zq = np.clip(np.round((z - lo) / np.maximum(hi - lo, 1e-9) * (lev - 1)), 0, lev - 1)
    zq = zq / (lev - 1) * (hi - lo) + lo
    rec = np.sign(zq @ W.T); rec[rec == 0] = 1.0
    x = np.sign(z @ W.T); x[x == 0] = 1.0
    return {"rate": k * b_bits / dim, "ber": ber_np(rec, x), "bits": k * b_bits}


def baseline_pca(Xtr, Xte, d, b_bits):
    """
    Codificacion por transformada clasica: PCA (d componentes) + cuantizacion
    uniforme a b_bits por componente. Tasa = d*b/dim. Competidor JUSTO del AE:
    aprendido de datos, sin oraculo, sin red neuronal.
    """
    Xtr = Xtr.numpy() if torch.is_tensor(Xtr) else Xtr
    Xte = Xte.numpy() if torch.is_tensor(Xte) else Xte
    dim = Xtr.shape[1]
    mu = Xtr.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    V = Vt[:d].T
    ztr, zte = (Xtr - mu) @ V, (Xte - mu) @ V
    lo, hi = ztr.min(0), ztr.max(0)
    lev = 2 ** b_bits
    zq = np.clip(np.round((zte - lo) / np.maximum(hi - lo, 1e-9) * (lev - 1)), 0, lev - 1)
    zq = zq / (lev - 1) * (hi - lo) + lo
    rec = zq @ V.T + mu
    return {"rate": d * b_bits / dim, "ber": ber_np(rec, Xte), "bits": d * b_bits}


def stage_baselines(cfg):
    print(f"\n{'='*78}\nETAPA D - BASELINES CLASICOS (el AE tiene que ganarle a esto)\n{'='*78}")
    rows = []
    for regime in cfg["regimes"]:
        Xtr, Xte, meta, side = build_dataset(regime, cfg["n_train"], cfg["n_test"],
                                             cfg["dim"], cfg, cfg["seed"], return_side=True)
        print(f"\n  --- {regime} (H = {meta['entropy_bits']:.1f} bits) ---")

        for d in cfg["pca_dims"]:
            for b in cfg["pca_bits"]:
                r = baseline_pca(Xtr, Xte, d, b)
                if r["rate"] > 1.05:
                    continue
                rows.append(dict(regime=regime, metodo=f"PCA d={d} b={b}", **r))
                print(f"    PCA  d={d:3d} b={b} | R={r['rate']:.3f} "
                      f"({cfg['dim']/r['bits']:5.2f}x) | BER={r['ber']:.4f}")

        if regime == "markov":
            for m in [2, 3, 4, 6, 8, 12, 16]:
                r = baseline_decimacion(Xte, m)
                rows.append(dict(regime=regime, metodo=f"decimacion m={m}", **r))
                print(f"    DECIM m={m:2d}    | R={r['rate']:.3f} "
                      f"({cfg['dim']/r['bits']:5.2f}x) | BER={r['ber']:.4f}")
        if regime == "code":
            for f in [1.0, 0.75, 0.5, 0.25]:
                r = baseline_oraculo_code(Xte, side["P"], f)
                rows.append(dict(regime=regime, metodo=f"oraculo frac={f}", **r))
                print(f"    ORACULO f={f:4.2f} | R={r['rate']:.3f} "
                      f"({cfg['dim']/max(r['bits'],1):5.2f}x) | BER={r['ber']:.4f}")
        if regime == "lowdim":
            for b in [1, 2, 3, 4, 6, 8]:
                r = baseline_oraculo_lowdim(side["z_test"], side["W"], b)
                rows.append(dict(regime=regime, metodo=f"oraculo z b={b}", **r))
                print(f"    ORACULO z b={b} | R={r['rate']:.3f} "
                      f"({cfg['dim']/r['bits']:5.2f}x) | BER={r['ber']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v3_baselines.csv"), index=False)
    return df


# =========================================================================== #
# 5. DIAGNOSTICO DE PARIDAD   (ETAPA E)
# =========================================================================== #
def stage_parity(cfg):
    """
    Aisla la hipotesis: SGD no aprende XOR de grado alto, aunque haya datos y
    capacidad de sobra. Si falla en d=3, queda explicado por que el AE no
    comprime la fuente 'code' (cuyos checks son exactamente de grado 3).
    """
    print(f"\n{'='*78}\nETAPA E - DIAGNOSTICO: puede un MLP aprender XOR de grado d?\n{'='*78}")
    dev, dim = cfg["device"], cfg["dim"]
    rng = np.random.default_rng(cfg["seed"])
    n_tr, n_te = cfg["par_train"], 20000
    rows = []
    for d in cfg["par_degrees"]:
        sub = rng.choice(dim, size=d, replace=False)
        Xtr = torch.from_numpy(src_random(n_tr, dim, rng))
        Xte = torch.from_numpy(src_random(n_te, dim, rng))
        ytr = (Xtr[:, sub].prod(1) > 0).float()
        yte = (Xte[:, sub].prod(1) > 0).float()

        torch.manual_seed(cfg["seed"])
        net = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(),
                            nn.Linear(512, 512), nn.ReLU(),
                            nn.Linear(512, 1)).to(dev)
        Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for ep in range(cfg["par_epochs"]):
            perm = torch.randperm(n_tr, device=dev)
            for i in range(0, n_tr, 512):
                idx = perm[i:i + 512]
                loss = F.binary_cross_entropy_with_logits(net(Xtr[idx]).squeeze(1), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc_tr = ((net(Xtr).squeeze(1) > 0).float() == ytr).float().mean().item()
            acc_te = ((net(Xte).squeeze(1) > 0).float() == yte).float().mean().item()
        rows.append(dict(grado=d, acc_train=acc_tr, acc_test=acc_te))
        veredicto = "APRENDE" if acc_te > 0.9 else ("parcial" if acc_te > 0.6 else "FALLA (azar)")
        print(f"  grado d={d}: acc_train={acc_tr:.4f}  acc_test={acc_te:.4f}   -> {veredicto}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v3_paridad.csv"), index=False)
    return df


# =========================================================================== #
# 6. BARRIDO AE  (ETAPA B, igual que v2)
# =========================================================================== #
def stage_sweep(cfg):
    print(f"\n{'='*78}\nETAPA B - BARRIDO AE: BER vs TASA REAL (latente binario)\n{'='*78}")
    rows = []
    for regime in cfg["regimes"]:
        Xtr, Xte, meta = build_dataset(regime, cfg["n_train"], cfg["n_test"],
                                       cfg["dim"], cfg, cfg["seed"])
        print(f"\n  --- {regime}: H = {meta['entropy_bits']:.1f} bits "
              f"(tasa min sin perdida = {meta['entropy_bits']/cfg['dim']:.3f}) ---")
        for arch in cfg["archs"]:
            for L in cfg["latents"]:
                torch.manual_seed(cfg["seed"])
                m = AE(cfg["dim"], L, cfg["hidden"], arch, "binary")
                r = train_model(m, Xtr, Xte, cfg, loss_kind="bce")
                rate = L / cfg["dim"]
                rows.append(dict(regime=regime, arch=arch, latent_bits=L, rate=rate,
                                 compresion=cfg["dim"] / L,
                                 entropy_bits=meta["entropy_bits"],
                                 rd_bound_ber=float(rd_bound_iid(rate)), **r))
                print(f"    {arch:6s} | L={L:4d} ({cfg['dim']/L:5.2f}x) | R={rate:.3f} "
                      f"| test_BER={r['test_ber']:.4f} | train_BER={r['train_ber']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v3_barrido.csv"), index=False)
    return df


# =========================================================================== #
# 7. JSCC CORREGIDO  (ETAPA C)
# =========================================================================== #
def stage_jscc(cfg):
    print(f"\n{'='*78}\nETAPA C - JSCC: BER vs Eb/N0 (latentes chicos, SNR baja)\n{'='*78}")
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
                piso = eval_ebn0_curve(m, Xte, [99], cfg["dim"], cfg["device"], reps=1)[0][1]
                for db, be in eval_ebn0_curve(m, Xte, cfg["ebn0_grid"], cfg["dim"], cfg["device"]):
                    rows.append(dict(regime=regime, latent_bits=L, rate=L / cfg["dim"],
                                     canal_en_entrenamiento=aware, ebn0_db=db,
                                     test_ber=be, piso_sin_ruido=piso,
                                     exceso_ber=be - piso))
                tag = "canal-aware" if aware else "sin canal"
                print(f"  {regime:7s} L={L:3d} ({tag:11s}) piso={piso:.4f} | " +
                      "  ".join(f"{db}dB:{be:.4f}" for db, be in
                                eval_ebn0_curve(m, Xte, cfg["ebn0_grid"], cfg["dim"], cfg["device"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["outdir"], "v3_jscc.csv"), index=False)
    return df


# =========================================================================== #
# 8. GRAFICOS
# =========================================================================== #
def plot_ae_vs_baselines(df_ae, df_bl, cfg):
    """El grafico clave del informe: AE contra baselines contra cotas."""
    if df_ae is None or df_bl is None:
        return
    regs = [r for r in cfg["regimes"] if r in set(df_ae.regime)]
    fig, axes = plt.subplots(1, len(regs), figsize=(4.4 * len(regs), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    rr = np.linspace(0.02, 1.0, 150)
    for ax, reg in zip(axes, regs):
        a = df_ae[(df_ae.regime == reg) & (df_ae.arch == "deep")].sort_values("rate")
        al = df_ae[(df_ae.regime == reg) & (df_ae.arch == "linear")].sort_values("rate")
        ax.plot(a.rate, a.test_ber, "o-", color="tab:blue", label="AE profundo")
        if not al.empty:
            ax.plot(al.rate, al.test_ber, "^-", color="tab:cyan", label="AE lineal")
        b = df_bl[df_bl.regime == reg]
        for kind, mk, c in [("PCA", "s--", "tab:orange"),
                            ("decim", "d--", "tab:green"),
                            ("oraculo", "*--", "tab:red")]:
            sel = b[b.metodo.str.contains(kind, case=False)].sort_values("rate")
            if not sel.empty:
                ax.plot(sel.rate, sel.ber, mk, color=c, label=kind, alpha=0.85)
        ax.plot(rr, rd_bound_iid(rr), "k--", lw=1.3, label="R(D) i.i.d.")
        if reg == "markov":
            ax.plot(rr, rd_bound_markov(rr, cfg["markov_p"]), "k:", lw=1.6, label="SLB markov")
        h = df_ae[df_ae.regime == reg].entropy_bits.iloc[0] / cfg["dim"]
        ax.axvline(h, ls=":", c="grey"); ax.set_yscale("log")
        ax.set_ylim(1e-4, 0.6); ax.set_title(f"{reg}  (H/n={h:.2f})")
        ax.set_xlabel("Tasa R"); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("BER (test)")
    axes[-1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("AE vs baselines clasicos vs cotas teoricas  (linea gris = entropia de la fuente)")
    plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v3_ae_vs_baselines.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")


def plot_parity(df, cfg):
    if df is None or df.empty:
        return
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(df.grado, df.acc_train, "o-", label="train")
    plt.plot(df.grado, df.acc_test, "s-", label="test")
    plt.axhline(0.5, ls="--", c="grey", label="azar")
    plt.ylim(0.4, 1.02); plt.xticks(df.grado)
    plt.xlabel("grado d del XOR"); plt.ylabel("exactitud")
    plt.title("Un MLP no aprende paridad de grado alto\n(por eso el AE no ve la redundancia algebraica)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v3_paridad.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")


def plot_jscc(df, cfg):
    if df is None or df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    grid = np.array(cfg["ebn0_grid"], dtype=float)
    axes[0].semilogy(grid, np.maximum(qfunc(np.sqrt(2 * 10 ** (grid / 10))), 1e-6),
                     "k--", lw=1.8, label="BPSK sin codificar (R=1)")
    for (reg, L, aw), g in df.groupby(["regime", "latent_bits", "canal_en_entrenamiento"]):
        g = g.sort_values("ebn0_db")
        lab = f"{reg} L={L} ({'aware' if aw else 'sin canal'})"
        axes[0].semilogy(g.ebn0_db, np.maximum(g.test_ber, 1e-6), "o-" if aw else "s:",
                         label=lab, alpha=1.0 if aw else 0.6)
        axes[1].semilogy(g.ebn0_db, np.maximum(g.exceso_ber, 1e-6), "o-" if aw else "s:",
                         label=lab, alpha=1.0 if aw else 0.6)
    axes[0].set_title("BER total"); axes[1].set_title("EXCESO sobre el piso de compresion\n(esto si mide el efecto del canal)")
    for ax in axes:
        ax.set_xlabel("Eb/N0 [dB] (Eb = energia por bit de FUENTE)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    axes[0].set_ylabel("BER (test)")
    plt.tight_layout()
    p = os.path.join(cfg["outdir"], "v3_jscc.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"[guardado] {p}")


# =========================================================================== #
# 9. CONFIG / MAIN
# =========================================================================== #
def get_cfg(a):
    c = dict(dim=500, k=32, markov_p=0.05, hidden=512, lr=2e-3, seed=0,
             device=a.device, outdir=a.outdir,
             regimes=["random", "lowdim", "code", "markov"],
             archs=["deep", "linear"],
             pca_dims=[8, 16, 32, 64, 128, 250], pca_bits=[1, 2, 4],
             par_degrees=[1, 2, 3, 4],
             jscc_regimes=["markov", "lowdim"], train_ebn0=0.0,
             ebn0_grid=[-6, -4, -2, 0, 2, 4, 6, 8, 10])
    if a.mode == "quick":
        c.update(n_train=16000, n_test=5000, epochs=40, batch=1024, hidden=384,
                 latents=[250, 128, 64, 32], jscc_latents=[64],
                 par_train=60000, par_epochs=25)
    else:
        c.update(n_train=200000, n_test=40000, epochs=200, batch=1024, hidden=1024,
                 latents=[400, 300, 250, 200, 150, 128, 100, 64, 48, 32, 16, 8],
                 jscc_latents=[128, 64, 32], par_train=400000, par_epochs=80)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--stage", default="all",
                    help="all | lista separada por comas: baselines,parity,sweep,jscc")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="./ae_results_v3")
    a = ap.parse_args()
    cfg = get_cfg(a)
    os.makedirs(cfg["outdir"], exist_ok=True)
    if a.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    stages = ["baselines", "parity", "sweep", "jscc"] if a.stage == "all" \
        else [s.strip() for s in a.stage.split(",")]
    print("CONFIG:", json.dumps(cfg, default=str))

    print(f"\nENTROPIA DE CADA FUENTE (dim={cfg['dim']}):")
    for reg in cfg["regimes"]:
        _, _, m = build_dataset(reg, 8, 8, cfg["dim"], cfg, cfg["seed"])
        print(f"  {reg:7s}: H = {m['entropy_bits']:7.1f} bits -> tasa min = "
              f"{m['entropy_bits']/cfg['dim']:.3f} b/b (compresion max "
              f"{cfg['dim']/m['entropy_bits']:.2f}x)")

    t0 = time.time()
    df_bl = stage_baselines(cfg) if "baselines" in stages else None
    if "parity" in stages:
        plot_parity(stage_parity(cfg), cfg)
    df_ae = stage_sweep(cfg) if "sweep" in stages else None
    if df_ae is not None and df_bl is not None:
        plot_ae_vs_baselines(df_ae, df_bl, cfg)
    if "jscc" in stages:
        plot_jscc(stage_jscc(cfg), cfg)
    print(f"\nTiempo total: {time.time()-t0:.1f}s   (salidas en {cfg['outdir']})")


if __name__ == "__main__":
    main()
