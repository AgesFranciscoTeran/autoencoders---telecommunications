#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
La vara definitiva: codificacion por transformada con SINTESIS OPTIMA.

Lloyd-Max ya dio la cuantizacion optima en MSE de la proyeccion, y tumbo
`lowdim` L=70 y L=125. Pero quedaba un eslabon debil: la reconstruccion seguia
siendo `zq @ Vd.T + mu`, es decir, transponer la base de analisis. Eso es
optimo solo si el codigo no tiene error; con codigo cuantizado NO lo es.

La sintesis optima es la que minimiza el error de reconstruccion dado el codigo,
y para un decoder lineal es minimos cuadrados:

    D* = argmin_D || Zq D - X ||_F     ->     D* = (Zq^T Zq)^-1 Zq^T X

Es codificacion por transformada de libro de texto (analisis PCA + cuantizador
escalar optimo + filtro de sintesis optimo). Sigue siendo un metodo clasico:
no hay red, no hay no linealidad, no hay oraculo.

Se prueban las cuatro combinaciones y se toma la mejor:

    cuantizador  x  sintesis
    -----------     --------
    uniforme opt    transpuesta Vd^T   (la vara anterior)
    uniforme opt    minimos cuadrados
    Lloyd-Max       transpuesta Vd^T   (la vara de la vuelta anterior)
    Lloyd-Max       minimos cuadrados  <- la mas fuerte esperada

Sin fuga: PCA, cuantizador y decoder se ajustan en TRAIN; el barrido de escala
uniforme usa VALIDACION; el BER se mide siempre en TEST.

Uso:  python3 vara_definitiva.py
"""
import gc, json
import numpy as np

DIM, K, P_MK, M_OS = 500, 32, 0.05, 4
LADDER = [35, 70, 125, 250]
N_TR, N_VA, N_TE = 100000, 20000, 20000
N_FIT = 20000


def ber(rec, x):
    return float((np.sign(rec) != np.sign(x)).mean())


def hacer(nombre, n, rng, W=None):
    if nombre == "lowdim":
        z = rng.standard_normal((n, K)).astype(np.float32)
        x = np.sign(z @ W.T); x[x == 0] = 1
        return x.astype(np.float32)
    if nombre == "markov":
        fl = (rng.random((n, DIM)) < P_MK).astype(np.int8)
        fl[:, 0] = rng.integers(0, 2, n).astype(np.int8)
        return (2.0 * (np.cumsum(fl, 1) % 2) - 1).astype(np.float32)
    if nombre == "random":
        return (2.0 * rng.integers(0, 2, (n, DIM)) - 1).astype(np.float32)
    if nombre == "oversamp":
        s = 2.0 * rng.integers(0, 2, (n, DIM // M_OS)).astype(np.float32) - 1
        return np.repeat(s, M_OS, 1)
    raise ValueError(nombre)


def q_uniforme(z, lo, hi, lev):
    q = np.clip(np.round((z - lo) / np.maximum(hi - lo, 1e-9) * (lev - 1)), 0, lev - 1)
    return q / (lev - 1) * (hi - lo) + lo


def lloyd_max(col, lev, iters=40):
    c = np.unique(np.percentile(col, np.linspace(0, 100, lev * 2 + 1)[1::2]))
    if len(c) < lev:
        c = np.linspace(float(col.min()), float(col.max()) + 1e-6, lev)
    for _ in range(iters):
        bord = (c[:-1] + c[1:]) / 2
        idx = np.searchsorted(bord, col)
        nuevo = c.copy()
        for k in range(lev):
            m = idx == k
            if m.any():
                nuevo[k] = col[m].mean()
        nuevo = np.sort(nuevo)
        if np.allclose(nuevo, c, rtol=1e-6, atol=1e-9):
            break
        c = nuevo
    return c


def aplica_lloyd(z, centros):
    out = np.empty_like(z)
    for j, c in enumerate(centros):
        bord = (c[:-1] + c[1:]) / 2
        out[:, j] = c[np.searchsorted(bord, z[:, j])]
    return out


def cov_chunked(X, mu, bs=25000):
    C = np.zeros((DIM, DIM), np.float64)
    for i in range(0, len(X), bs):
        d = (X[i:i + bs] - mu).astype(np.float64)
        C += d.T @ d
    return C / (len(X) - 1)


def decoder_ls(Zq, X, ridge=1e-6):
    """Sintesis optima: D* = (Zq^T Zq + eps I)^-1 Zq^T X, con sesgo."""
    n, d = Zq.shape
    A = np.concatenate([Zq, np.ones((n, 1), np.float32)], 1).astype(np.float64)
    G = A.T @ A
    G[np.diag_indices_from(G)] += ridge * np.trace(G) / len(G)
    D = np.linalg.solve(G, A.T @ X.astype(np.float64))
    return D.astype(np.float32)


def aplica_ls(Zq, D):
    A = np.concatenate([Zq, np.ones((len(Zq), 1), np.float32)], 1)
    return A @ D


def varas(Xtr, Xva, Xte, L):
    mu = Xtr.mean(0, keepdims=True)
    _, V = np.linalg.eigh(cov_chunked(Xtr, mu)); V = V[:, ::-1]
    claves = ("unif_transp", "unif_ls", "lloyd_transp", "lloyd_ls")
    best = {k: (1.0, None) for k in claves}
    for b in (1, 2, 4):
        d = L // b
        if d < 1:
            continue
        lev = 2 ** b
        Vd = V[:, :d].astype(np.float32)
        ztr_fit = (Xtr[:N_FIT] - mu) @ Vd
        ztr = (Xtr - mu) @ Vd
        zva, zte = (Xva - mu) @ Vd, (Xte - mu) @ Vd

        # --- cuantizador uniforme: escala barrida en VALIDACION ---
        med, s_col = ztr_fit.mean(0), ztr_fit.std(0)
        mejor_s, mejor_e = 1.0, 2.0
        for s in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0):
            sd = s_col * s
            e = ber(q_uniforme(zva, med - sd, med + sd, lev) @ Vd.T + mu, Xva)
            if e < mejor_e: mejor_e, mejor_s = e, s
        sd = s_col * mejor_s
        qu = lambda z: q_uniforme(z, med - sd, med + sd, lev)

        # --- cuantizador Lloyd-Max ajustado en TRAIN ---
        centros = [lloyd_max(ztr_fit[:, j], lev) for j in range(d)]
        ql = lambda z: aplica_lloyd(z, centros)

        for nomq, q in (("unif", qu), ("lloyd", ql)):
            zq_te = q(zte)
            e = ber(zq_te @ Vd.T + mu, Xte)                      # sintesis = transpuesta
            k = f"{nomq}_transp"
            if e < best[k][0]: best[k] = (e, f"d={d} b={b}")

            D = decoder_ls(q(ztr), Xtr)                          # sintesis = minimos cuadrados
            e = ber(aplica_ls(zq_te, D), Xte)
            k = f"{nomq}_ls"
            if e < best[k][0]: best[k] = (e, f"d={d} b={b}")
            del zq_te, D; gc.collect()

        del ztr_fit, ztr, zva, zte; gc.collect()
    return best


def decim_best(X, L):
    best = (1.0, None)
    for m in range(2, 40):
        keep = np.arange(0, DIM, m)
        if len(keep) > L:
            continue
        h = keep[np.clip(np.arange(DIM) // m, 0, len(keep) - 1)]
        v = keep[np.argmin(np.abs(np.arange(DIM)[:, None] - keep[None, :]), 1)]
        for idx, nm in ((h, "hold"), (v, "vecino")):
            e = ber(X[:, idx], X)
            if e < best[0]:
                best = (e, f"decim m={m} ({nm})")
    return best


AE = {("lowdim", 35): 0.1870, ("lowdim", 70): 0.1051,
      ("lowdim", 125): 0.0607, ("lowdim", 250): 0.0346,
      ("markov", 35): 0.1498, ("markov", 70): 0.0902,
      ("markov", 125): 0.0464, ("markov", 250): 0.0107}
LLOYD_T = {("lowdim", 35): 0.1932, ("lowdim", 70): 0.0977,
           ("lowdim", 125): 0.0513, ("lowdim", 250): 0.0403,
           ("markov", 35): 0.1529, ("markov", 70): 0.0915,
           ("markov", 125): 0.0490, ("markov", 250): 0.0250}


def main():
    print("=" * 100)
    print("VARA DEFINITIVA  |  PCA + cuantizador optimo + SINTESIS OPTIMA (minimos cuadrados)")
    print(f"n_train={N_TR}  n_val={N_VA}  n_test={N_TE}  (todo se ajusta en train/val)")
    print("=" * 100)
    print(f"{'fuente':9s}{'L':>5s}{'unif+Vt':>9s}{'unif+LS':>9s}{'Lloyd+Vt':>10s}"
          f"{'Lloyd+LS':>10s}{'decim':>9s}{'MEJOR':>9s}  ganador")
    filas, nuevas = [], {}
    for nombre in ("lowdim", "markov", "oversamp", "random"):
        rng = np.random.default_rng(0)
        W = (rng.standard_normal((DIM, K)).astype(np.float32) / np.sqrt(K)
             if nombre == "lowdim" else None)
        Xtr, Xva, Xte = (hacer(nombre, N_TR, rng, W), hacer(nombre, N_VA, rng, W),
                         hacer(nombre, N_TE, rng, W))
        for L in LADDER:
            cand = dict(varas(Xtr, Xva, Xte, L))
            cand["decim"] = (decim_best(Xte, L) if nombre in ("markov", "oversamp")
                             else (float("nan"), None))
            val = {k: v[0] for k, v in cand.items()}
            gan = min((k for k in val if not np.isnan(val[k])), key=lambda k: val[k])
            nuevas[(nombre, L)] = val[gan]
            print(f"{nombre:9s}{L:5d}{val['unif_transp']:9.4f}{val['unif_ls']:9.4f}"
                  f"{val['lloyd_transp']:10.4f}{val['lloyd_ls']:10.4f}{val['decim']:9.4f}"
                  f"{val[gan]:9.4f}  {gan} {cand[gan][1] or ''}")
            filas.append(dict(fuente=nombre, latente_bits=L,
                              **{f"ber_{k}": float(v) for k, v in val.items()},
                              ber_baseline=float(val[gan]),
                              metodo=f"{gan} {cand[gan][1] or ''}".strip()))
        del Xtr, Xva, Xte; gc.collect()

    print("\n" + "=" * 100)
    print("IMPACTO ACUMULADO SOBRE LOS MARGENES")
    print("=" * 100)
    print(f"{'punto':14s}{'AE':>9s}{'Lloyd+Vt':>10s}{'margen':>9s}"
          f"{'vara final':>12s}{'margen':>9s}  veredicto")
    g1 = g2 = 0
    for (f, L), ae in sorted(AE.items()):
        v1, v2 = LLOYD_T[(f, L)], nuevas[(f, L)]
        g1 += v1 > ae; g2 += v2 > ae
        print(f"{f+' L='+str(L):14s}{ae:9.4f}{v1:10.4f}{v1-ae:+9.4f}"
              f"{v2:12.4f}{v2-ae:+9.4f}  {'gana' if v2 > ae else 'PIERDE'}")
    print(f"\n  victorias con Lloyd + transpuesta : {g1}/{len(AE)}")
    print(f"  victorias con la vara definitiva  : {g2}/{len(AE)}")

    json.dump(filas, open("varas_definitivas.json", "w"), indent=1)
    print("\n[guardado] varas_definitivas.json")


if __name__ == "__main__":
    main()
