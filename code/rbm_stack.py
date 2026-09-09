#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBM_STACK -- ablacion de la pregunta abierta 5:
  "Por que el preentrenamiento voraz por capas perjudica en este problema,
   cuando en Hinton y Salakhutdinov (2006) ayudaba?"

El modo `stacked` de ae2.py ya implementa la reduccion paulatina
500 -> 250 -> 125 -> 70 -> 35 con autoencoders por etapa, y pierde contra
`direct`. Este script separa DOS explicaciones posibles:

  (a) el culpable es el ESQUEMA voraz: cada etapa descarta informacion antes
      de ver el objetivo final, y el ajuste fino arranca en una cuenca peor
      que la inicializacion aleatoria + BatchNorm.
  (b) el culpable es el OBJETIVO por etapa: la reconstruccion local no es una
      buena senal; un modelo generativo (RBM con divergencia contrastiva,
      la receta original de 2006) si lo seria.

Tres brazos con la MISMA arquitectura en el ajuste fino (escalera estrecha
500-250-125-70-35, sigmoides intermedias, BatchNorm sin afin + STE en el
cuello, decoder espejo con logits). Solo cambia de donde sale la
inicializacion:

  ladder_random   inicializacion aleatoria, extremo a extremo.   <- control
  stacked_ae      voraz por capas con AE (reconstruccion), luego ajuste fino.
  stacked_rbm     voraz por capas con RBM binaria-binaria (CD-1), luego
                  ajuste fino. Encoder = W_j, decoder = W_j^T (Hinton 2006).

Nota: `stacked_ae` aqui usa sigmoides intermedias en vez de latentes
binarios por etapa (lo que hace ae2.py), para que el unico cambio frente a
`stacked_rbm` sea el objetivo de la etapa. Es la ablacion, no una replica.

PREDICCION REGISTRADA ANTES DE CORRER:
  1. stacked_rbm pierde contra ladder_random en 4/4 semillas en lowdim y
     markov, en todos los escalones.  (favorece (a))
  2. stacked_rbm empata con stacked_ae: |dif| < 2 sd sobre semillas.
  3. En `code`, los tres brazos quedan al nivel de ruido (BER ~ igual que
     random a la misma L): la RBM no ve la paridad de grado 3.
Si la RBM gana a ladder_random de forma consistente, (b) era el culpable y
hay que decirlo.

Uso:
    python3 rbm_stack.py --device cuda
    python3 rbm_stack.py --device cuda --fuentes lowdim markov --seeds 0 1 2 3
    python3 rbm_stack.py --device cuda --quick        # humo, ~minutos
"""
import argparse, copy, json, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# El arnes se llama ae2.py en el servidor y ae_telecom_v4.py en el repo.
# Solo se usan build_dataset, BinarizeSTE y ber_hard, identicas en ambos.
try:
    import ae2 as A
except ModuleNotFoundError:
    import ae_telecom_v4 as A

for _s in ("build_dataset", "BinarizeSTE", "ber_hard"):
    if not hasattr(A, _s):
        raise ImportError(f"el arnes importado ({A.__file__}) no expone {_s}")

LADDER = [250, 125, 70, 35]
ARMS = ["ladder_random", "stacked_ae", "stacked_rbm"]

# Varas clasicas (mejor metodo con el mismo numero de bits), de 03-resultados.md.
# Completar desde data/ los escalones que falten; NaN = sin referencia.
VARAS = {
    ("lowdim", 70): 0.1861, ("lowdim", 35): 0.1954,
    ("markov", 70): 0.0914, ("markov", 35): 0.1531,
}


# =========================================================================== #
# 0. UTILIDADES
# =========================================================================== #
def to01(x):        # {-1,+1} -> {0,1}
    return (x + 1.0) * 0.5


def bce(logits, x):
    return F.binary_cross_entropy_with_logits(logits, to01(x))


def layer_from_01(W, b):
    """
    Capa entrenada para entrada v en {0,1}, que ahora recibira x en {-1,+1}.
    W v + b = W (x+1)/2 + b = (W/2) x + (W.sum(1)/2 + b).
    """
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W * 0.5)
        lin.bias.copy_(b + W.sum(1) * 0.5)
    return lin


def layer_plain(W, b):
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    return lin


@torch.no_grad()
def apply_chunks(fn, X, bs=16384):
    return torch.cat([fn(X[i:i + bs]) for i in range(0, X.shape[0], bs)], 0)


# =========================================================================== #
# 1. ARQUITECTURA COMUN DEL AJUSTE FINO
# =========================================================================== #
class LadderAE(nn.Module):
    """
    Encoder: dims[0] -> dims[1] -> ... -> dims[-1], sigmoides intermedias,
             BatchNorm(affine=False) + STE en el cuello (config validada).
    Decoder: espejo, sigmoides intermedias, logits al final.
    La primera capa del encoder y la primera del decoder reciben +/-1.
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = list(dims)
        self.enc = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        rd = list(reversed(dims))
        self.dec = nn.ModuleList([nn.Linear(rd[i], rd[i + 1])
                                  for i in range(len(rd) - 1)])
        self.bn = nn.BatchNorm1d(dims[-1], affine=False)

    def encode(self, x):
        h = x
        for i, l in enumerate(self.enc):
            h = l(h)
            if i < len(self.enc) - 1:
                h = torch.sigmoid(h)
        return A.BinarizeSTE.apply(self.bn(h))

    def decode(self, z):
        h = z
        for i, l in enumerate(self.dec):
            h = l(h)
            if i < len(self.dec) - 1:
                h = torch.sigmoid(h)
        return h

    def forward(self, x):
        return self.decode(self.encode(x))


def build_ladder(depth):
    return [500] + LADDER[:depth]


# =========================================================================== #
# 2. PREENTRENAMIENTO VORAZ: RBM (CD-1)
# =========================================================================== #
def train_rbm(V, n_h, cfg, gen):
    """
    RBM binaria-binaria. V en [0,1] (bits o probabilidades de la etapa previa).
    CD-1 con momentum, como en Hinton (2002/2006). Devuelve W (n_h x n_v),
    b (visible), c (oculto) y el error de reconstruccion final.
    """
    dev = V.device
    n, n_v = V.shape
    W = 0.01 * torch.randn(n_h, n_v, device=dev, generator=gen)
    p = V.mean(0).clamp(1e-3, 1 - 1e-3)
    b = torch.log(p / (1 - p))                      # sesgo visible = log-odds
    c = torch.zeros(n_h, device=dev)
    dW = torch.zeros_like(W); db = torch.zeros_like(b); dc = torch.zeros_like(c)
    bs, lr, wd = cfg["rbm_batch"], cfg["rbm_lr"], cfg["rbm_wd"]
    err = float("nan")
    for step in range(cfg["rbm_steps"]):
        mom = cfg["rbm_mom0"] if step < cfg["rbm_steps"] // 5 else cfg["rbm_mom1"]
        idx = torch.randint(0, n, (bs,), device=dev, generator=gen)
        v0 = V[idx]
        h0p = torch.sigmoid(v0 @ W.T + c)
        h0 = torch.bernoulli(h0p, generator=gen)
        v1p = torch.sigmoid(h0 @ W + b)             # visible: probabilidades
        h1p = torch.sigmoid(v1p @ W.T + c)
        gW = (h0p.T @ v0 - h1p.T @ v1p) / bs
        gb = (v0 - v1p).mean(0)
        gc = (h0p - h1p).mean(0)
        dW.mul_(mom).add_(lr * (gW - wd * W)); W.add_(dW)
        db.mul_(mom).add_(lr * gb);            b.add_(db)
        dc.mul_(mom).add_(lr * gc);            c.add_(dc)
        if step == cfg["rbm_steps"] - 1 or step % 1000 == 0:
            err = ((v0 - v1p) ** 2).mean().item()
    return W, b, c, err


def pretrain_rbm_stack(X, cfg, gen):
    """Devuelve lista de etapas [(W_j, b_j, c_j, err_j)] y el log."""
    V = to01(X)
    etapas, log = [], []
    dims = [500] + LADDER
    for j in range(len(LADDER)):
        t0 = time.time()
        W, b, c, err = train_rbm(V, dims[j + 1], cfg, gen)
        etapas.append((W, b, c))
        log.append(dict(etapa=j, n_v=dims[j], n_h=dims[j + 1],
                        recon_err=err, seg=round(time.time() - t0, 1)))
        print(f"      RBM {dims[j]:3d}->{dims[j+1]:3d}  err_recon={err:.4f}  "
              f"({time.time()-t0:.0f}s)")
        V = apply_chunks(lambda v: torch.sigmoid(v @ W.T + c), V)  # probs
    return etapas, log


# =========================================================================== #
# 3. PREENTRENAMIENTO VORAZ: AUTOENCODER POR ETAPA
# =========================================================================== #
def train_ae_stage(V, n_h, cfg, gen):
    """
    Etapa AE de una capa: V -> sigmoid(Wv+c) -> logits W'h+b, BCE contra V.
    Misma forma que la RBM (una capa, misma no linealidad); cambia el objetivo.
    """
    dev = V.device
    n, n_v = V.shape
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    enc = nn.Linear(n_v, n_h).to(dev)
    dec = nn.Linear(n_h, n_v).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=cfg["ae_lr"])
    bs = cfg["rbm_batch"]
    for step in range(cfg["rbm_steps"]):
        idx = torch.randint(0, n, (bs,), device=dev, generator=gen)
        v0 = V[idx]
        lg = dec(torch.sigmoid(enc(v0)))
        loss = F.binary_cross_entropy_with_logits(lg, v0)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        idx = torch.randint(0, n, (bs,), device=dev, generator=gen)
        v0 = V[idx]
        err = ((torch.sigmoid(dec(torch.sigmoid(enc(v0)))) - v0) ** 2).mean().item()
    return (enc.weight.detach().clone(), enc.bias.detach().clone(),
            dec.weight.detach().clone(), dec.bias.detach().clone(), err)


def pretrain_ae_stack(X, cfg, gen):
    V = to01(X)
    etapas, log = [], []
    dims = [500] + LADDER
    for j in range(len(LADDER)):
        t0 = time.time()
        We, ce, Wd, bd, err = train_ae_stage(V, dims[j + 1], cfg, gen)
        etapas.append((We, ce, Wd, bd))
        log.append(dict(etapa=j, n_v=dims[j], n_h=dims[j + 1],
                        recon_err=err, seg=round(time.time() - t0, 1)))
        print(f"      AE  {dims[j]:3d}->{dims[j+1]:3d}  err_recon={err:.4f}  "
              f"({time.time()-t0:.0f}s)")
        V = apply_chunks(lambda v: torch.sigmoid(v @ We.T + ce), V)
    return etapas, log


# =========================================================================== #
# 4. DESENROLLAR A LadderAE
# =========================================================================== #
def unroll(arm, depth, etapas_rbm, etapas_ae, dev):
    m = LadderAE(build_ladder(depth))
    if arm == "ladder_random":
        return m.to(dev)
    # encoder: capa 0 recibe +/-1, las demas reciben probabilidades
    for j in range(depth):
        if arm == "stacked_rbm":
            W, b, c = etapas_rbm[j]; We, ce = W, c
        else:
            We, ce, _, _ = etapas_ae[j]
        m.enc[j] = layer_from_01(We, ce) if j == 0 else layer_plain(We, ce)
    # decoder: capa 0 recibe z en +/-1 (el cuello), las demas probabilidades
    for i in range(depth):
        j = depth - 1 - i                     # etapa que se invierte
        if arm == "stacked_rbm":
            W, b, c = etapas_rbm[j]; Wd, bd = W.T.contiguous(), b
        else:
            _, _, Wd, bd = etapas_ae[j]
        m.dec[i] = layer_from_01(Wd, bd) if i == 0 else layer_plain(Wd, bd)
    return m.to(dev)


# =========================================================================== #
# 5. AJUSTE FINO (config validada: BN sin afin + STE, lr 1e-3, mejor checkpoint)
# =========================================================================== #
@torch.no_grad()
def ber(m, X):
    m.eval()
    return A.ber_hard(m(X).float(), X)


def finetune(m, Xtr, Xva, Xte, cfg):
    dev = cfg["device"]
    n, bs = Xtr.shape[0], cfg["batch"]
    steps = cfg["ft_steps"]
    amp = (dev == "cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                              total_steps=steps, pct_start=0.1)
    best_va, best_state, best_step = 1.0, copy.deepcopy(m.state_dict()), 0
    eval_every = max(1, steps // 40)
    for step in range(steps):
        m.train()
        idx = torch.randint(0, n, (bs,), device=dev)
        xb = Xtr[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = bce(m(xb), xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            va = ber(m, Xva)
            if va < best_va:
                best_va, best_step = va, step + 1
                best_state = copy.deepcopy(m.state_dict())
    final_te = ber(m, Xte)
    m.load_state_dict(best_state)
    best_te = ber(m, Xte)
    return dict(ber_test=best_te, ber_test_final=final_te,
                ber_val=best_va, mejor_step=best_step)


# =========================================================================== #
# 6. CORRIDA
# =========================================================================== #
def corre_fuente_semilla(fuente, seed, cfg, out):
    dev = cfg["device"]
    Xtr, Xrest, meta, _ = A.build_dataset(fuente, cfg["n_train"], cfg["n_test"] * 2,
                                          500, cfg, seed)
    Xva, Xte = Xrest[:cfg["n_test"]], Xrest[cfg["n_test"]:]
    Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)
    print(f"\n[{fuente} seed={seed}]  H={meta['entropy_bits']:.1f} bits  "
          f"train={Xtr.shape[0]} val={Xva.shape[0]} test={Xte.shape[0]}")

    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    print("    preentrenamiento RBM:")
    etapas_rbm, log_rbm = pretrain_rbm_stack(Xtr, cfg, gen)
    gen.manual_seed(seed + 1000)
    print("    preentrenamiento AE:")
    etapas_ae, log_ae = pretrain_ae_stack(Xtr, cfg, gen)

    filas = []
    print(f"    {'brazo':14s} {'L':>4s} {'BER test':>9s} {'final':>8s} "
          f"{'@step':>6s} {'vara':>7s}")
    for depth, L in enumerate(LADDER, start=1):
        for arm in cfg["arms"]:
            torch.manual_seed(seed)              # misma init aleatoria en BN etc.
            m = unroll(arm, depth, etapas_rbm, etapas_ae, dev)
            t0 = time.time()
            r = finetune(m, Xtr, Xva, Xte, cfg)
            vara = VARAS.get((fuente, L), float("nan"))
            print(f"    {arm:14s} {L:4d} {r['ber_test']:9.4f} {r['ber_test_final']:8.4f} "
                  f"{r['mejor_step']:6d} {vara:7.4f}  ({time.time()-t0:.0f}s)")
            filas.append(dict(fuente=fuente, seed=seed, arm=arm, L=L,
                              rate=L / 500, vara=vara, **r))
            del m
            if dev == "cuda": torch.cuda.empty_cache()

    # control de monotonia: mas bits siempre debe dar menos BER
    for arm in cfg["arms"]:
        s = sorted([f for f in filas if f["arm"] == arm], key=lambda f: -f["L"])
        bers = [f["ber_test"] for f in s]
        if any(bers[i] > bers[i + 1] + 1e-4 for i in range(len(bers) - 1)):
            print(f"    AVISO monotonia violada en {arm}: {[round(b,4) for b in bers]}"
                  f"  -> alguna corrida divergio; mirar mejor_step")

    json.dump(dict(fuente=fuente, seed=seed, log_rbm=log_rbm, log_ae=log_ae,
                   filas=filas),
              open(os.path.join(out, f"rbm_stack_{fuente}_s{seed}.json"), "w"),
              indent=1)
    del Xtr, Xva, Xte
    if dev == "cuda": torch.cuda.empty_cache()
    return filas


def resumen(filas, cfg, out):
    import pandas as pd
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(out, "rbm_stack.csv"), index=False)
    g = df.groupby(["fuente", "L", "arm"]).ber_test.agg(["mean", "std", "count"])
    print("\n" + "=" * 78)
    print("RESUMEN  (media +/- sd sobre semillas; BER de test al mejor checkpoint)")
    print("=" * 78)
    piv = g["mean"].unstack("arm")
    sd = g["std"].unstack("arm")
    for (fuente, L), row in piv.iterrows():
        vara = VARAS.get((fuente, L), float("nan"))
        cad = "  ".join(f"{a}={row[a]:.4f}+/-{sd.loc[(fuente, L), a]:.4f}"
                        for a in cfg["arms"] if a in row)
        print(f"{fuente:8s} L={L:3d}  {cad}  vara={vara:.4f}")

    print("\nCONTRASTE CON LA PREDICCION")
    for fuente in df.fuente.unique():
        for L in LADDER:
            s = df[(df.fuente == fuente) & (df.L == L)]
            if not {"ladder_random", "stacked_rbm", "stacked_ae"} <= set(s.arm):
                continue
            p = s.pivot(index="seed", columns="arm", values="ber_test")
            n = len(p)
            rbm_gana_random = int((p.stacked_rbm < p.ladder_random).sum())
            d = p.stacked_rbm - p.stacked_ae
            empate = abs(d.mean()) < 2 * max(d.std(ddof=1) if n > 1 else 0, 1e-4)
            rbm_gana_ae = int((p.stacked_rbm < p.stacked_ae).sum())
            print(f"  {fuente:8s} L={L:3d}: RBM<random en {rbm_gana_random}/{n}  "
                  f"RBM<AE en {rbm_gana_ae}/{n}  "
                  f"RBM-AE={d.mean():+.4f}  "
                  f"{'EMPATE' if empate else 'DIFERENCIA'}")
    print("\n  Prediccion 1 (RBM pierde contra ladder_random 4/4 en lowdim/markov)")
    print("  Prediccion 2 (RBM empata con stacked_ae)")
    print("  Prediccion 3 (en code, todo al nivel de ruido)")
    print("  Si RBM<random sale consistente, el culpable era el OBJETIVO por")
    print("  etapa, no el esquema voraz: cambia la respuesta a la pregunta 5.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="./rbm_stack")
    ap.add_argument("--fuentes", nargs="*", default=["lowdim", "markov", "code"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--ft_steps", type=int, default=30000)   # mismos pasos que cierre --stage datos
    ap.add_argument("--rbm_steps", type=int, default=10000)  # por etapa, RBM y AE por igual
    ap.add_argument("--n_train", type=int, default=400000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.outdir, exist_ok=True)

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               n_train=a.n_train, n_test=20000,
               batch=4096, lr=1e-3, wd=1e-4, ft_steps=a.ft_steps,
               rbm_batch=256, rbm_lr=0.05, rbm_wd=1e-4,
               rbm_mom0=0.5, rbm_mom1=0.9, rbm_steps=a.rbm_steps,
               ae_lr=1e-3, arms=a.arms)
    if a.quick:
        cfg.update(n_train=40000, n_test=5000, ft_steps=600, rbm_steps=400)
        a.seeds = a.seeds[:1]

    print("=" * 78)
    print("RBM_STACK  |  escalera", " -> ".join(map(str, [500] + LADDER)))
    print(f"fuentes={a.fuentes}  semillas={a.seeds}  brazos={cfg['arms']}")
    print(f"ft_steps={cfg['ft_steps']}  rbm_steps/etapa={cfg['rbm_steps']}  "
          f"n_train={cfg['n_train']}  device={dev}")
    print("=" * 78)

    filas = []
    for fuente in a.fuentes:
        for seed in a.seeds:
            filas += corre_fuente_semilla(fuente, seed, cfg, a.outdir)
    resumen(filas, cfg, a.outdir)
    json.dump(dict(cfg={k: v for k, v in cfg.items()}, fuentes=a.fuentes,
                   seeds=a.seeds, ladder=LADDER),
              open(os.path.join(a.outdir, "rbm_stack_cfg.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
