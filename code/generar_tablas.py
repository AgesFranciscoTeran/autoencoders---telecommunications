#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIVEL 1 DE REPRODUCCION -- sin GPU, sin PyTorch, corre en segundos.

Regenera las tablas verificadas del proyecto:
  data/cotas_teoricas.csv      cota de Shannon (SLB) por fuente y escalon
  data/baselines_clasicos.csv  mejor metodo clasico por fuente y escalon
  data/pca_vs_random.csv       evidencia de que PCA es ciego al codigo de bloque

Uso:  python3 code/generar_tablas.py
"""
import math, os
import numpy as np
import pandas as pd

DIM, LADDER = 500, [35, 70, 125, 250]
K_LOWDIM, P_MARKOV, M_OVER = 32, 0.05, 4
N_TR, N_TE, SEED = 20000, 5000, 0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ----------------------------- teoria ------------------------------------- #
def Hb(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def Hb_inv(h):
    """Inversa de la entropia binaria en la rama [0, 0.5], por biseccion."""
    h = float(np.clip(h, 0, 1)); lo, hi = 0.0, 0.5
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if Hb(m) < h:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def slb(rate, H_over_n):
    """Cota inferior de Shannon, fuente binaria, distorsion de Hamming:
         R(D) >= H/n - Hb(D)   =>   D >= Hb^-1(H/n - R)
       Vale para cualquier fuente estacionaria binaria, no solo i.i.d."""
    return Hb_inv(max(0.0, H_over_n - rate))


def log2_regiones(n, k):
    """log2 C(n,k) = log2( 2 * sum_{i<k} binom(n-1,i) ).
       Numero de dicotomias linealmente separables (Cover, 1965): es la
       entropia de la fuente 'lowdim'."""
    logs = [math.lgamma(n) - math.lgamma(i + 1) - math.lgamma(n - i) for i in range(k)]
    mx = max(logs)
    return (mx + math.log(sum(math.exp(l - mx) for l in logs)) + math.log(2)) / math.log(2)


def ber(rec, x):
    return float((np.sign(rec) != np.sign(x)).mean())


# ----------------------------- fuentes ------------------------------------ #
def genera(regime, seed=SEED):
    r = np.random.default_rng(seed)
    if regime == "random":
        f = lambda n: 2.0 * r.integers(0, 2, (n, DIM)).astype(np.float32) - 1
        return f(N_TR), f(N_TE), {}, float(DIM)
    if regime == "oversamp":
        k = DIM // M_OVER
        f = lambda n: np.repeat(2.0 * r.integers(0, 2, (n, k)).astype(np.float32) - 1,
                                M_OVER, axis=1)
        return f(N_TR), f(N_TE), {}, float(k)
    if regime == "markov":
        def f(n):
            fl = (r.random((n, DIM)) < P_MARKOV).astype(np.int8)
            fl[:, 0] = r.integers(0, 2, n).astype(np.int8)
            return (2.0 * (np.cumsum(fl, 1) % 2) - 1).astype(np.float32)
        return f(N_TR), f(N_TE), {}, float(1 + (DIM - 1) * Hb(P_MARKOV))
    if regime == "lowdim":
        W = r.standard_normal((DIM, K_LOWDIM)).astype(np.float32) / np.sqrt(K_LOWDIM)
        def f(n):
            z = r.standard_normal((n, K_LOWDIM)).astype(np.float32)
            x = np.sign(z @ W.T)
            x[x == 0] = 1
            return x.astype(np.float32)
        return f(N_TR), f(N_TE), {"W": W}, log2_regiones(DIM, K_LOWDIM)
    if regime == "code":
        k = DIM // 2
        P = np.zeros((k, k), np.int32)
        for j in range(k):
            P[r.choice(k, 3, replace=False), j] = 1
        def f(n):
            u = r.integers(0, 2, (n, k)).astype(np.int8)
            par = (u.astype(np.int32) @ P) % 2
            return 2.0 * np.concatenate([u, par.astype(np.int8)], 1).astype(np.float32) - 1
        return f(N_TR), f(N_TE), {"P": P}, float(k)
    raise ValueError(regime)


# --------------------------- baselines ------------------------------------ #
def pca_curva(Xtr, Xte, presupuestos):
    """PCA + cuantizacion uniforme. La eigendescomposicion se calcula UNA vez."""
    mu = Xtr.mean(0, keepdims=True)
    C = (Xtr - mu).T @ (Xtr - mu) / (Xtr.shape[0] - 1)
    _, V = np.linalg.eigh(C)
    V = V[:, ::-1]
    best = {L: (1.0, "-") for L in presupuestos}
    for b in (1, 2, 4):
        for L in presupuestos:
            d = L // b
            if d < 1:
                continue
            Vd = V[:, :d]
            ztr = (Xtr - mu) @ Vd
            zte = (Xte - mu) @ Vd
            lo, hi = ztr.min(0), ztr.max(0)
            lev = 2 ** b
            zq = np.clip(np.round((zte - lo) / np.maximum(hi - lo, 1e-9) * (lev - 1)),
                         0, lev - 1)
            rec = (zq / (lev - 1) * (hi - lo) + lo) @ Vd.T + mu
            e = ber(rec, Xte)
            if e < best[L][0]:
                best[L] = (e, f"PCA d={d} b={b}")
    return best


def decimacion(X, m, modo):
    """Envia 1 de cada m simbolos. 'hold' = sample-and-hold, 'vecino' = mas cercano.
       Usar solo 'vecino' SUBESTIMA la vara en fuentes sobremuestreadas."""
    keep = np.arange(0, DIM, m)
    if modo == "hold":
        idx = keep[np.clip(np.arange(DIM) // m, 0, len(keep) - 1)]
    else:
        idx = keep[np.argmin(np.abs(np.arange(DIM)[:, None] - keep[None, :]), 1)]
    return len(keep), ber(X[:, idx], X)


def oraculo_code(X, P, n_send):
    """Oraculo: transmite n_send bits sistematicos y RECALCULA las paridades.
       Con n_send = k da BER exactamente 0 usando la mitad de los bits."""
    k = DIM // 2
    b = ((X + 1) / 2).astype(np.int32)
    u = np.zeros((X.shape[0], k), np.int32)
    u[:, :n_send] = b[:, :n_send]
    return ber(2.0 * np.concatenate([u, (u @ P) % 2], 1) - 1, X)


# ------------------------------ main -------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    regs = ["random", "oversamp", "markov", "lowdim", "code"]
    filas_cotas, filas_bl, filas_pca = [], [], []

    for reg in regs:
        Xtr, Xte, side, H = genera(reg)
        for L in LADDER:
            filas_cotas.append(dict(fuente=reg, entropia_bits=round(H, 1),
                                    H_sobre_n=round(H / DIM, 4), latente_bits=L,
                                    tasa=L / DIM,
                                    ber_minimo=round(slb(L / DIM, H / DIM), 4),
                                    sin_perdida_posible=bool(L >= H)))
        best = pca_curva(Xtr, Xte, LADDER)
        for L in LADDER:
            filas_pca.append(dict(fuente=reg, latente_bits=L,
                                  ber_pca=round(best[L][0], 4), config=best[L][1]))
        if reg in ("markov", "oversamp"):
            for m in range(2, 33):
                for modo in ("hold", "vecino"):
                    nb, e = decimacion(Xte, m, modo)
                    if nb > max(LADDER):
                        continue
                    for L in LADDER:
                        if nb <= L and e < best[L][0]:
                            best[L] = (e, f"decimacion m={m} ({modo})")
        if reg == "code":
            for L in LADDER:
                e = oraculo_code(Xte, side["P"], min(DIM // 2, L))
                if e < best[L][0]:
                    best[L] = (e, f"oraculo {min(DIM//2, L)}/{DIM//2}")
        for L in LADDER:
            filas_bl.append(dict(fuente=reg, latente_bits=L, tasa=L / DIM,
                                 compresion=round(DIM / L, 2),
                                 ber_baseline=round(best[L][0], 4), metodo=best[L][1]))

    for nombre, filas in (("cotas_teoricas", filas_cotas),
                          ("baselines_clasicos", filas_bl),
                          ("pca_vs_random", filas_pca)):
        df = pd.DataFrame(filas)
        p = os.path.join(OUT, nombre + ".csv")
        df.to_csv(p, index=False)
        print(f"[guardado] {p}  ({len(df)} filas)")

    pca = pd.DataFrame(filas_pca).set_index(["fuente", "latente_bits"]).ber_pca
    print("\nPCA sobre 'code' vs 'random' (si son iguales, PCA es ciego al codigo):")
    print(f"  {'L':>5s} {'code':>8s} {'random':>8s} {'dif':>8s}")
    for L in LADDER:
        c, r = pca[("code", L)], pca[("random", L)]
        print(f"  {L:5d} {c:8.4f} {r:8.4f} {c-r:+8.4f}")


if __name__ == "__main__":
    main()
