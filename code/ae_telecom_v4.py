#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AUTOENCODERS PARA SENALES BPSK (+/-1, dim 500)  --  HARNESS v4  (H200)
================================================================================
Cambios frente a v3:

 (1) ESCALERA ANIDADA (successive refinement / codificacion escalable).
     UN solo modelo con latente binario de L_max bits, entrenado para que
     CUALQUIER prefijo sea decodificable:  35 | 70 | 125 | 250 bits.
     -> codec compatible en tasa: 4 puntos de operacion, 1 modelo.
     Es el equivalente ML de los codigos rate-compatible punctured.

 (2) PREENTRENAMIENTO APILADO (Hinton & Salakhutdinov 2006).
     500->250, congelar, 250->125, 125->70, 70->35, luego fine-tuning e2e.
     Modo --train stacked.  Modo --train direct = un AE independiente por
     escalon (control, para saber cuanto cuesta la anidacion).

 (3) FUENTE 'oversamp' (500 = 125 simbolos x 4): senal sobremuestreada, el
     caso realista donde el AE SI debe ganar. H = 125 bits exactos, que cae
     justo en un escalon de la escalera.

 (4) SALIDA BLANDA / LLR.  El decoder emite logits ~ LLR. Se reporta el BER
     restringido a los bits mas confiables: si los errores se concentran en
     los bits de baja confianza, un FEC con decision blanda los limpia. Es
     la metrica que convierte "BER 2%" en "viable con FEC".

 (5) FIXES de v3: eigendecomposicion de PCA cacheada (v3 recalculaba el SVD
     hasta 18 veces por fuente), decimacion hold ADEMAS de vecino (en
     'oversamp' el hold da BER=0 y el vecino 0.12: usar solo vecino
     SUBESTIMA la vara), RNG del canal sembrado, y sin llamadas repetidas
     a la curva Eb/N0.

 (6) H200: bf16 autocast, TF32, datos residentes en GPU, batch grande,
     reanudacion por JSONL (no repite configs hechas) y sharding por GPU.

Uso:
    # smoke test (CPU, segundos)
    python3 ae_telecom_v4.py --mode smoke --device cpu

    # produccion, 2 GPUs en paralelo (shards independientes)
    CUDA_VISIBLE_DEVICES=0 python3 ae_telecom_v4.py --mode full --device cuda \
        --shard 0 --num-shards 2 --outdir ./res &
    CUDA_VISIBLE_DEVICES=1 python3 ae_telecom_v4.py --mode full --device cuda \
        --shard 1 --num-shards 2 --outdir ./res &
    wait
    python3 ae_telecom_v4.py --report --outdir ./res
================================================================================
"""

import argparse, hashlib, json, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LADDER = [35, 70, 125, 250]          # escalones (bits), de grueso a fino


# =========================================================================== #
# 0. TEORIA
# =========================================================================== #
def Hb(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def Hb_inv(h):
    h = float(np.clip(h, 0.0, 1.0)); lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Hb(mid) < h: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def slb_ber(rate, H_over_n):
    """Cota inferior de Shannon, fuente binaria + distorsion de Hamming:
       R(D) >= H/n - Hb(D)   =>   D >= Hb^{-1}(H/n - R).
       Vale para CUALQUIER fuente estacionaria binaria (no solo i.i.d.)."""
    return Hb_inv(max(0.0, H_over_n - rate))


def log2_regions(n, k):
    """log2 C(n,k) = log2( 2 * sum_{i<k} binom(n-1,i) ): entropia de 'lowdim'."""
    logs = [math.lgamma(n) - math.lgamma(i + 1) - math.lgamma(n - i) for i in range(k)]
    mx = max(logs)
    tot = mx + math.log(sum(math.exp(l - mx) for l in logs))
    return (tot + math.log(2.0)) / math.log(2.0)


# =========================================================================== #
# 1. FUENTES
# =========================================================================== #
def build_dataset(regime, n_train, n_test, dim, cfg, seed):
    """La fuente (W, P) es la MISMA en train y test; los simbolos son frescos."""
    rng = np.random.default_rng(seed)
    side = {}

    if regime == "random":
        f = lambda n: 2.0 * rng.integers(0, 2, (n, dim)).astype(np.float32) - 1.0
        Xtr, Xte = f(n_train), f(n_test)
        H = float(dim)

    elif regime == "oversamp":
        m = cfg["oversample"]; k = dim // m
        f = lambda n: np.repeat(
            2.0 * rng.integers(0, 2, (n, k)).astype(np.float32) - 1.0, m, axis=1)
        Xtr, Xte = f(n_train), f(n_test)
        H = float(k)

    elif regime == "markov":
        p = cfg["markov_p"]
        def f(n):
            fl = (rng.random((n, dim)) < p).astype(np.int8)
            fl[:, 0] = rng.integers(0, 2, n).astype(np.int8)
            return (2.0 * (np.cumsum(fl, 1) % 2) - 1.0).astype(np.float32)
        Xtr, Xte = f(n_train), f(n_test)
        H = float(1 + (dim - 1) * Hb(p))

    elif regime == "lowdim":
        k = cfg["k"]
        W = rng.standard_normal((dim, k)).astype(np.float32) / np.sqrt(k)
        def f(n):
            z = rng.standard_normal((n, k)).astype(np.float32)
            x = np.sign(z @ W.T).astype(np.float32); x[x == 0.0] = 1.0
            return x
        Xtr, Xte = f(n_train), f(n_test)
        H = log2_regions(dim, k); side = {"W": W}

    elif regime == "code":
        k = dim // 2
        P = np.zeros((k, k), dtype=np.int32)
        for j in range(k):
            P[rng.choice(k, size=3, replace=False), j] = 1
        def f(n):
            u = rng.integers(0, 2, (n, k)).astype(np.int8)
            par = (u.astype(np.int32) @ P) % 2
            return 2.0 * np.concatenate([u, par.astype(np.int8)], 1).astype(np.float32) - 1.0
        Xtr, Xte = f(n_train), f(n_test)
        H = float(k); side = {"P": P}

    else:
        raise ValueError(regime)

    meta = {"entropy_bits": H, "H_over_n": H / dim}
    return torch.from_numpy(Xtr), torch.from_numpy(Xte), meta, side


# =========================================================================== #
# 2. MODELO: latente binario ANIDADO
# =========================================================================== #
class BinarizeSTE(torch.autograd.Function):
    """sign(u) hacia adelante; hacia atras, identidad recortada (|u|<=1).
       El recorte evita que gradientes de unidades ya saturadas sigan
       empujando: es el straight-through estandar (Bengio et al. 2013)."""
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        s = torch.sign(u); s[s == 0] = 1.0
        return s

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * (u.abs() <= 1.0).to(g.dtype)


def mlp(sizes, out_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


class NestedAE(nn.Module):
    """
    Encoder -> L_max bits binarios. Para el escalon L se ANULAN los bits > L
    (puncturing) y el decoder reconstruye desde el prefijo.
    Un indicador de tasa (one-hot del escalon) se concatena al decoder para
    que sepa cuantos bits recibio realmente.
    """
    def __init__(self, dim, ladder, hidden=1024, depth=3):
        super().__init__()
        self.ladder = list(ladder); self.Lmax = max(ladder); self.dim = dim
        enc_sizes = [dim] + [hidden] * depth + [self.Lmax]
        dec_sizes = [self.Lmax + len(ladder)] + [hidden] * depth + [dim]
        self.enc = mlp(enc_sizes)
        self.dec = mlp(dec_sizes)

    def encode(self, x):
        return BinarizeSTE.apply(torch.tanh(self.enc(x)))

    def decode(self, z, rung_idx):
        L = self.ladder[rung_idx]
        mask = torch.zeros(self.Lmax, device=z.device, dtype=z.dtype)
        mask[:L] = 1.0
        tag = torch.zeros(z.shape[0], len(self.ladder), device=z.device, dtype=z.dtype)
        tag[:, rung_idx] = 1.0
        return self.dec(torch.cat([z * mask, tag], dim=1))

    def forward(self, x, rung_idx):
        return self.decode(self.encode(x), rung_idx)


class PlainAE(nn.Module):
    """AE de un solo escalon (modo direct / etapas del modo stacked)."""
    def __init__(self, dim_in, latent, dim_out=None, hidden=1024, depth=3):
        super().__init__()
        dim_out = dim_out or dim_in
        self.latent = latent
        self.enc = mlp([dim_in] + [hidden] * depth + [latent])
        self.dec = mlp([latent] + [hidden] * depth + [dim_out])

    def encode(self, x): return BinarizeSTE.apply(torch.tanh(self.enc(x)))
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))


# =========================================================================== #
# 3. METRICAS
# =========================================================================== #
def ber_hard(logits, x):
    h = torch.sign(logits); h[h == 0] = -1.0
    return (h != x).float().mean().item()


def ber_by_confidence(logits, x, fracs=(0.5, 0.9, 1.0)):
    """
    BER restringido a la fraccion q de bits MAS confiables (|LLR| alto).
    Si BER(q=0.9) << BER(q=1.0), los errores viven en bits que el decoder
    ya marca como dudosos -> un FEC de decision blanda los corrige.
    """
    h = torch.sign(logits); h[h == 0] = -1.0
    err = (h != x).flatten()
    conf = logits.abs().flatten()
    order = torch.argsort(conf, descending=True)
    e = err[order].float()
    n = e.numel()
    return {f"ber_top{int(q*100)}": e[:max(1, int(q * n))].mean().item() for q in fracs}


# =========================================================================== #
# 4. ENTRENAMIENTO
# =========================================================================== #
def make_opt(model, cfg, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(1, steps), pct_start=0.1)
    return opt, sch


def bce(logits, x):
    return F.binary_cross_entropy_with_logits(logits, (x + 1.0) * 0.5)


def train_nested(Xtr, Xte, cfg):
    """Entrena UN modelo valido en los 4 escalones simultaneamente."""
    dev, dim = cfg["device"], Xtr.shape[1]
    model = NestedAE(dim, LADDER, cfg["hidden"], cfg["depth"]).to(dev)
    n, bs = Xtr.shape[0], cfg["batch"]
    spe = max(1, n // bs)
    opt, sch = make_opt(model, cfg, cfg["epochs"] * spe)
    amp = (dev == "cuda")

    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            xb = Xtr[perm[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                # todos los escalones en el mismo paso -> el prefijo se vuelve
                # el sub-codigo optimo (esto es lo que fuerza la anidacion)
                loss = sum(bce(model(xb, r), xb) for r in range(len(LADDER)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

    model.eval()
    out = []
    with torch.no_grad():
        for r, L in enumerate(LADDER):
            lg_te = model(Xte, r).float()
            lg_tr = model(Xtr[:cfg["n_test"]], r).float()
            out.append(dict(latent_bits=L, rate=L / dim,
                            test_ber=ber_hard(lg_te, Xte),
                            train_ber=ber_hard(lg_tr, Xtr[:cfg["n_test"]]),
                            **ber_by_confidence(lg_te, Xte)))
    return out, model


def train_direct(Xtr, Xte, cfg):
    """Un AE independiente por escalon (control: cuanto cuesta anidar)."""
    dev, dim = cfg["device"], Xtr.shape[1]
    n, bs = Xtr.shape[0], cfg["batch"]
    spe = max(1, n // bs); amp = (dev == "cuda")
    out = []
    for L in LADDER:
        model = PlainAE(dim, L, dim, cfg["hidden"], cfg["depth"]).to(dev)
        opt, sch = make_opt(model, cfg, cfg["epochs"] * spe)
        for ep in range(cfg["epochs"]):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n - bs + 1, bs):
                xb = Xtr[perm[i:i + bs]]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = bce(model(xb), xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            lg = model(Xte).float()
            out.append(dict(latent_bits=L, rate=L / dim, test_ber=ber_hard(lg, Xte),
                            train_ber=ber_hard(model(Xtr[:cfg["n_test"]]).float(),
                                               Xtr[:cfg["n_test"]]),
                            **ber_by_confidence(lg, Xte)))
    return out, None


def train_stacked(Xtr, Xte, cfg):
    """
    Hinton & Salakhutdinov: greedy por capas y luego fine-tuning e2e.
    Etapa j comprime el codigo binario de la etapa j-1:
        500 -> 250 -> 125 -> 70 -> 35
    Evaluar a profundidad d = detenerse en la etapa d (escalon LADDER[-d]).
    """
    dev, dim = cfg["device"], Xtr.shape[1]
    n, bs = Xtr.shape[0], cfg["batch"]
    spe = max(1, n // bs); amp = (dev == "cuda")
    chain = list(reversed(LADDER))              # [250, 125, 70, 35]
    stages, cur, cur_dim = [], Xtr, dim

    for L in chain:                             # --- preentrenamiento greedy ---
        ae = PlainAE(cur_dim, L, cur_dim, cfg["hidden"], cfg["depth"]).to(dev)
        opt, sch = make_opt(ae, cfg, cfg["pre_epochs"] * spe)
        for ep in range(cfg["pre_epochs"]):
            ae.train()
            perm = torch.randperm(cur.shape[0], device=dev)
            for i in range(0, cur.shape[0] - bs + 1, bs):
                xb = cur[perm[i:i + bs]]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = bce(ae(xb), xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                opt.step(); sch.step()
        ae.eval()
        with torch.no_grad():                   # codigo de entrada de la siguiente etapa
            cur = torch.cat([ae.encode(cur[i:i + 8192])
                             for i in range(0, cur.shape[0], 8192)], 0)
        stages.append(ae); cur_dim = L

    def run(x, depth):
        z = x
        for j in range(depth):
            z = stages[j].encode(z)
        for j in reversed(range(depth)):
            lg = stages[j].decode(z)
            z = BinarizeSTE.apply(torch.tanh(lg)) if j > 0 else lg
        return z                                # logits en el espacio original

    out = []
    for depth in range(1, len(chain) + 1):      # --- fine-tuning e2e por profundidad ---
        params = [p for j in range(depth) for p in stages[j].parameters()]
        opt = torch.optim.AdamW(params, lr=cfg["lr"] * 0.3, weight_decay=1e-4)
        for ep in range(cfg["ft_epochs"]):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n - bs + 1, bs):
                xb = Xtr[perm[i:i + bs]]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = bce(run(xb, depth), xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
        with torch.no_grad():
            lg = run(Xte, depth).float()
            out.append(dict(latent_bits=chain[depth - 1], rate=chain[depth - 1] / dim,
                            test_ber=ber_hard(lg, Xte),
                            train_ber=ber_hard(run(Xtr[:cfg["n_test"]], depth).float(),
                                               Xtr[:cfg["n_test"]]),
                            **ber_by_confidence(lg, Xte)))
    return out, None


TRAINERS = {"nested": train_nested, "direct": train_direct, "stacked": train_stacked}


# =========================================================================== #
# 5. BASELINES CLASICOS  (misma escalera, misma vara)
# =========================================================================== #
def ber_np(rec, x): return float((np.sign(rec) != np.sign(x)).mean())


def baselines_for(Xtr, Xte, regime, side, dim):
    """PCA+cuantizacion (eigendecomp UNA vez), decimacion hold Y vecino, oraculos."""
    Xtr = Xtr.cpu().numpy(); Xte = Xte.cpu().numpy()
    mu = Xtr.mean(0, keepdims=True)
    C = (Xtr - mu).T @ (Xtr - mu) / (Xtr.shape[0] - 1)
    _, V = np.linalg.eigh(C); V = V[:, ::-1]               # <- fix v3: una sola vez
    best = {L: (1.0, "-") for L in LADDER}

    for b in [1, 2, 4]:
        for L in LADDER:
            d = L // b
            if d < 1: continue
            Vd = V[:, :d]; ztr = (Xtr - mu) @ Vd; zte = (Xte - mu) @ Vd
            lo, hi = ztr.min(0), ztr.max(0); lev = 2 ** b
            zq = np.clip(np.round((zte - lo) / np.maximum(hi - lo, 1e-9) * (lev - 1)), 0, lev - 1)
            rec = (zq / (lev - 1) * (hi - lo) + lo) @ Vd.T + mu
            e = ber_np(rec, Xte)
            if e < best[L][0]: best[L] = (e, f"PCA d={d} b={b}")

    if regime in ("markov", "oversamp"):
        for m in range(2, 33):
            keep = np.arange(0, dim, m)
            if len(keep) > max(LADDER): continue
            hold = keep[np.clip(np.arange(dim) // m, 0, len(keep) - 1)]
            near = keep[np.argmin(np.abs(np.arange(dim)[:, None] - keep[None, :]), 1)]
            for idx, nm in ((hold, "hold"), (near, "vecino")):
                e = ber_np(Xte[:, idx], Xte)
                for L in LADDER:
                    if len(keep) <= L and e < best[L][0]:
                        best[L] = (e, f"decim m={m} ({nm})")

    if regime == "code":
        b = ((Xte + 1) / 2).astype(np.int32); k = dim // 2
        for L in LADDER:
            ns = min(k, L)
            u = np.zeros((Xte.shape[0], k), np.int32); u[:, :ns] = b[:, :ns]
            rec = 2.0 * np.concatenate([u, (u @ side["P"]) % 2], 1) - 1.0
            e = ber_np(rec, Xte)
            if e < best[L][0]: best[L] = (e, f"oraculo {ns}/{k}")

    return [dict(latent_bits=L, rate=L / dim, baseline_ber=best[L][0],
                 baseline_metodo=best[L][1]) for L in LADDER]


# =========================================================================== #
# 6. ORQUESTACION (reanudable + sharding)
# =========================================================================== #
def job_key(j): return hashlib.md5(json.dumps(j, sort_keys=True).encode()).hexdigest()[:12]


def done_keys(path):
    if not os.path.exists(path): return set()
    return {json.loads(l)["key"] for l in open(path) if l.strip()}


def run(cfg, args):
    os.makedirs(cfg["outdir"], exist_ok=True)
    res_path = os.path.join(cfg["outdir"], "v4_resultados.jsonl")
    bl_path = os.path.join(cfg["outdir"], "v4_baselines.jsonl")
    done = done_keys(res_path)

    jobs = [{"regime": r, "train": t} for r in cfg["regimes"] for t in cfg["trainers"]]
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(jobs)} trabajos", flush=True)

    print(f"\n{'fuente':10s} {'H(bits)':>8s} {'H/n':>6s} | cota SLB por escalon")
    for r in cfg["regimes"]:
        _, _, meta, _ = build_dataset(r, 8, 8, cfg["dim"], cfg, cfg["seed"])
        cs = " ".join(f"L{L}:{slb_ber(L/cfg['dim'], meta['H_over_n']):.4f}" for L in LADDER)
        print(f"{r:10s} {meta['entropy_bits']:8.1f} {meta['H_over_n']:6.3f} | {cs}")

    bl_done = done_keys(bl_path)
    for job in jobs:
        key = job_key(job)
        if key in done:
            print(f"  [skip] {job}", flush=True); continue
        t0 = time.time()
        Xtr, Xte, meta, side = build_dataset(job["regime"], cfg["n_train"], cfg["n_test"],
                                             cfg["dim"], cfg, cfg["seed"])
        bk = job_key({"bl": job["regime"]})
        if bk not in bl_done:
            with open(bl_path, "a") as f:
                for row in baselines_for(Xtr, Xte, job["regime"], side, cfg["dim"]):
                    f.write(json.dumps({"key": bk, "regime": job["regime"], **row}) + "\n")
            bl_done.add(bk)

        Xtr, Xte = Xtr.to(cfg["device"]), Xte.to(cfg["device"])
        torch.manual_seed(cfg["seed"])
        rows, _ = TRAINERS[job["train"]](Xtr, Xte, cfg)
        with open(res_path, "a") as f:
            for row in rows:
                f.write(json.dumps({"key": key, "regime": job["regime"],
                                    "train_mode": job["train"],
                                    "entropy_bits": meta["entropy_bits"],
                                    "slb_ber": slb_ber(row["rate"], meta["H_over_n"]),
                                    **row}) + "\n")
        print(f"  [ok] {job['regime']:10s} {job['train']:8s} ({time.time()-t0:6.1f}s) | " +
              " ".join(f"L{r['latent_bits']}:{r['test_ber']:.4f}" for r in rows), flush=True)
        del Xtr, Xte
        if cfg["device"] == "cuda": torch.cuda.empty_cache()


def report(outdir):
    res = pd.read_json(os.path.join(outdir, "v4_resultados.jsonl"), lines=True)
    bl = pd.read_json(os.path.join(outdir, "v4_baselines.jsonl"), lines=True)
    m = res.merge(bl[["regime", "latent_bits", "baseline_ber", "baseline_metodo"]],
                  on=["regime", "latent_bits"], how="left")
    m["gana_baseline"] = m.test_ber < m.baseline_ber
    m["sobre_slb"] = m.test_ber - m.slb_ber
    m.to_csv(os.path.join(outdir, "v4_tabla.csv"), index=False)

    print("\n=== VIABILIDAD: AE vs vara clasica vs cota (BER de test) ===")
    cols = ["regime", "train_mode", "latent_bits", "rate", "test_ber",
            "baseline_ber", "baseline_metodo", "slb_ber", "sobre_slb",
            "ber_top90", "gana_baseline"]
    print(m[cols].sort_values(["regime", "train_mode", "latent_bits"]).to_string(index=False))

    regs = sorted(m.regime.unique())
    fig, axes = plt.subplots(1, len(regs), figsize=(4.2 * len(regs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, reg in zip(axes, regs):
        s = m[m.regime == reg]
        for tm, g in s.groupby("train_mode"):
            g = g.sort_values("latent_bits")
            ax.plot(g.rate, np.maximum(g.test_ber, 1e-5), "o-", label=f"AE {tm}")
        b = s.drop_duplicates("latent_bits").sort_values("latent_bits")
        ax.plot(b.rate, np.maximum(b.baseline_ber, 1e-5), "s--", c="tab:orange", label="mejor clasico")
        ax.plot(b.rate, np.maximum(b.slb_ber, 1e-5), "k:", lw=1.6, label="cota SLB")
        ax.axhline(1e-2, ls="--", c="green", lw=1, label="1e-2 (FEC)")
        ax.set_yscale("log"); ax.set_title(reg); ax.set_xlabel("tasa R = L/500")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("BER (test)"); axes[-1].legend(fontsize=7)
    fig.suptitle("Mapa de viabilidad: AE vs baseline clasico vs cota teorica")
    plt.tight_layout()
    p = os.path.join(outdir, "v4_viabilidad.png")
    plt.savefig(p, dpi=135); plt.close(); print(f"\n[guardado] {p}")


# =========================================================================== #
# 7. CONFIG / MAIN
# =========================================================================== #
def get_cfg(a):
    c = dict(dim=500, k=32, markov_p=0.05, oversample=4, lr=3e-3, seed=0,
             device=a.device, outdir=a.outdir,
             regimes=["random", "oversamp", "markov", "lowdim", "code"],
             trainers=["nested", "direct", "stacked"])
    if a.mode == "smoke":
        c.update(n_train=2000, n_test=1000, epochs=3, pre_epochs=2, ft_epochs=2,
                 batch=256, hidden=128, depth=2, regimes=["oversamp", "code"],
                 trainers=["nested"])
    elif a.mode == "quick":
        c.update(n_train=50000, n_test=10000, epochs=60, pre_epochs=30, ft_epochs=20,
                 batch=2048, hidden=512, depth=3)
    else:  # full  (H200)
        c.update(n_train=400000, n_test=50000, epochs=300, pre_epochs=120, ft_epochs=80,
                 batch=4096, hidden=1536, depth=4)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="./ae_results_v4")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        report(a.outdir); return

    if a.device == "cuda" and not torch.cuda.is_available():
        print("[aviso] CUDA no disponible -> CPU"); a.device = "cpu"
    if a.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    cfg = get_cfg(a)
    t0 = time.time()
    run(cfg, a)
    print(f"\nTiempo total: {time.time()-t0:.1f}s  (salidas en {cfg['outdir']})")


if __name__ == "__main__":
    main()
