#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vara_tapados.py — la vara que le faltaba a `ber_tapados`.

POR QUE EXISTE
--------------
El proyecto entero se apoya en que un BER suelto no significa nada sin un piso
y una vara. `ber_tapados` —el BER medido solo en las posiciones enmascaradas—
se venia reportando contra 0.5, que es el piso trivial (adivinar), y contra
NADA por arriba. Sin techo no se puede decir si 0.1103 en `lowdim` es mucho o
poco, ni normalizar "recupera un X% del camino": el denominador que se estaba
usando implicitamente era la inferencia perfecta (0.0), y la inferencia
perfecta NO es alcanzable. Con enmascarado a tasa p, ni un oraculo infiere un
bit cuyos vecinos informativos tambien quedaron tapados.

Este script calcula, para cada fuente y cada p, el mejor `ber_tapados`
alcanzable por alguien que CONOCE la estructura de la fuente. Con eso:

    fraccion del camino = (0.5 - ber_medido) / (0.5 - ber_oraculo)

que si es comparable entre fuentes.

QUE ES CADA ORACULO (la distincion importa al citarlo)
------------------------------------------------------
PISO EXACTO (optimo de Bayes; nadie puede bajar de ahi):

  `random`    0.5 analitico. Un bit i.i.d. tapado es impredecible.
  `oversamp`  Un simbolo se repite m=4 veces. Una posicion tapada esta
              determinada si y solo si sobrevive visible alguna de sus 3
              hermanas; si no, es media exacta. Combinatoria, sin modelo.
  `markov`    Forward-backward sobre la cadena binaria con p_flip=0.05 y
              observaciones exactas en las visibles. La marginal a posteriori
              por posicion es exacta, y el error de Bayes por bit es
              min(q, 1-q). Es el optimo por bit, que es justo lo que mide BER.
  `code`      Rango sobre GF(2). Cada coordenada observada impone una
              restriccion lineal sobre los 250 bits sistematicos: una
              sistematica visible fija u_i, una paridad visible fija el XOR de
              sus tres sistematicas. La posterior es uniforme sobre el espacio
              afin de soluciones, asi que una coordenada tapada esta
              determinada si y solo si su funcional esta en el espacio
              generado por los funcionales visibles, y si no lo esta vale
              media exacta. Se resuelve con eliminacion gaussiana en GF(2)
              (bitsets), y captura las cadenas de dependencias, no solo el
              caso "las tres sistematicas visibles".

VARA ALCANZABLE (no es un piso demostrado):

  `lowdim`    x = sign(Wz). Las visibles definen un cono {z : s_i w_i·z > 0} y
              la posterior es la gaussiana restringida a ese cono. La
              prediccion de Bayes exigiria integrar sobre el cono (MCMC). En
              su lugar se usa el estimador de margen maximo —el de medicion de
              un bit, Boufounos y Baraniuk (2008)— resuelto con Pegasos:
                  min (lam/2)||z||^2 + (1/|S|) sum_i max(0, 1 - s_i w_i·z)
              y se predice sign(w_m·z*) en las tapadas. Conoce W, no usa red
              y no usa las tapadas. Es una vara en el sentido del glosario
              (metodo concreto con conocimiento perfecto de la estructura), no
              un piso: la prediccion de Bayes puede ser algo mejor. Al citarlo
              hay que decirlo asi.

La mascara es Bernoulli(p) i.i.d., igual que en `denoising.py`. No se
reproduce el flujo de ruido exacto de la GPU —no hace falta: el oraculo es una
esperanza sobre la mascara, y cualquier realizacion i.i.d. la estima igual.

Las fuentes se regeneran con el mismo orden de llamadas al RNG que
`ae_telecom_v4.build_dataset`, asi que para una semilla dada W y P son
identicas a las de las corridas medidas.

USO
---
    python3 code/vara_tapados.py                      # todo, 4 semillas
    python3 code/vara_tapados.py --quick              # humo, 1 semilla
    python3 code/vara_tapados.py --fuentes code,markov --p 0.1,0.25
    python3 code/vara_tapados.py --sin-comparar       # solo la vara

Escribe data/varas_tapados.json. Si existen data/denoising/denoising_resumen.csv
o data/combinado/combinado.csv, imprime ademas la comparacion medido vs vara.
Las dos corridas se muestran por separado y con su origen etiquetado: son
experimentos distintos y mezclarlos en una misma fila ya costo una correccion.

Solo NumPy. Sin GPU. El caso lento es `code` (eliminacion en GF(2) en Python).
"""
import argparse, json, os, time
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")

DIM, K, P_MK, M_OS = 500, 32, 0.05, 4

# n por fuente: `code` es el caro; `markov` y `oversamp` son casi gratis.
N_POR_FUENTE = {"code": 200, "lowdim": 400, "markov": 2000,
                "oversamp": 5000, "random": 5000}

TIPO_VARA = {"random": "piso exacto", "oversamp": "piso exacto",
             "markov": "piso exacto", "code": "piso exacto",
             "lowdim": "vara alcanzable"}


# =========================================================================== #
# 1. FUENTES  (mismo orden de RNG que ae_telecom_v4.build_dataset)
# =========================================================================== #
def fuente(nombre, n, seed):
    """Devuelve (X en +/-1, estructura). W y P se sortean ANTES de los simbolos."""
    rng = np.random.default_rng(seed)

    if nombre == "random":
        return (2.0 * rng.integers(0, 2, (n, DIM)) - 1.0).astype(np.float32), {}

    if nombre == "oversamp":
        k = DIM // M_OS
        s = 2.0 * rng.integers(0, 2, (n, k)).astype(np.float32) - 1.0
        return np.repeat(s, M_OS, axis=1), {}

    if nombre == "markov":
        fl = (rng.random((n, DIM)) < P_MK).astype(np.int8)
        fl[:, 0] = rng.integers(0, 2, n).astype(np.int8)
        return (2.0 * (np.cumsum(fl, 1) % 2) - 1.0).astype(np.float32), {}

    if nombre == "lowdim":
        W = rng.standard_normal((DIM, K)).astype(np.float32) / np.sqrt(K)
        z = rng.standard_normal((n, K)).astype(np.float32)
        x = np.sign(z @ W.T).astype(np.float32)
        x[x == 0.0] = 1.0
        return x, {"W": W}

    if nombre == "code":
        k = DIM // 2
        P = np.zeros((k, k), dtype=np.int32)
        for j in range(k):
            P[rng.choice(k, size=3, replace=False), j] = 1
        u = rng.integers(0, 2, (n, k)).astype(np.int8)
        par = (u.astype(np.int32) @ P) % 2
        x = 2.0 * np.concatenate([u, par.astype(np.int8)], 1).astype(np.float32) - 1.0
        return x, {"P": P}

    raise ValueError(nombre)


def mascara(n, p, rng):
    """True donde se tapa. Bernoulli(p) i.i.d., como en denoising.py."""
    return rng.random((n, DIM)) < p


# =========================================================================== #
# 2. ORACULOS
# =========================================================================== #
def vara_random(X, tap, extra):
    """Piso exacto y analitico: un bit i.i.d. tapado no se puede predecir."""
    return 0.5


def vara_oversamp(X, tap, extra):
    """
    Piso exacto. La posicion t pertenece al bloque t//m; esta determinada si
    alguna otra posicion del bloque quedo visible. Si no, media exacta.
    """
    n = X.shape[0]
    bloques = tap.reshape(n, DIM // M_OS, M_OS)
    tapadas_por_bloque = bloques.sum(2, keepdims=True)          # (n, k, 1)
    # una posicion tapada esta indeterminada si TODO su bloque esta tapado
    indet = bloques & (tapadas_por_bloque == M_OS)
    n_tap = tap.sum()
    return 0.5 * float(indet.sum()) / max(int(n_tap), 1)


def vara_markov(X, tap, extra):
    """
    Piso exacto por forward-backward. Estado b_t in {0,1} con x_t = 2 b_t - 1,
    transicion [[1-p, p], [p, 1-p]], prior uniforme, observacion exacta en las
    visibles y verosimilitud plana en las tapadas. El error de Bayes por bit es
    min(q, 1-q) con q = P(b_t = 1 | observaciones).
    """
    n = X.shape[0]
    b = (X > 0).astype(np.int8)
    p = P_MK
    T = np.array([[1 - p, p], [p, 1 - p]], dtype=np.float64)

    # verosimilitud por posicion: (n, 2)
    def lik(t):
        l = np.ones((n, 2), dtype=np.float64)
        v = ~tap[:, t]
        if v.any():
            bt = b[v, t]
            l[v, 0] = (bt == 0)
            l[v, 1] = (bt == 1)
        return l

    alphas = np.empty((DIM, n, 2), dtype=np.float64)
    a = np.full((n, 2), 0.5) * lik(0)
    a /= np.maximum(a.sum(1, keepdims=True), 1e-300)
    alphas[0] = a
    for t in range(1, DIM):
        a = (a @ T) * lik(t)
        a /= np.maximum(a.sum(1, keepdims=True), 1e-300)
        alphas[t] = a

    beta = np.ones((n, 2), dtype=np.float64)
    err, n_tap = 0.0, 0
    for t in range(DIM - 1, -1, -1):
        post = alphas[t] * beta
        post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
        m = tap[:, t]
        if m.any():
            q = post[m, 1]
            err += float(np.minimum(q, 1.0 - q).sum())
            n_tap += int(m.sum())
        if t > 0:
            beta = (lik(t) * beta) @ T.T
            beta /= np.maximum(beta.sum(1, keepdims=True), 1e-300)
    return err / max(n_tap, 1)


def _funcionales(P):
    """
    Funcional lineal sobre GF(2) de cada una de las 500 coordenadas, como
    bitset de 250 bits sobre los sistematicos:
      - coordenada i < 250 (sistematica): e_i
      - coordenada 250+j (paridad j)    : XOR de las 3 sistematicas del check j
    """
    k = P.shape[0]
    f = [1 << i for i in range(k)]
    for j in range(k):
        v = 0
        for i in np.nonzero(P[:, j])[0]:
            v ^= 1 << int(i)
        f.append(v)
    return f


def vara_code(X, tap, extra):
    """
    Piso exacto por rango sobre GF(2). La posterior sobre u es uniforme en el
    espacio afin compatible con lo observado, asi que una coordenada tapada
    esta determinada si y solo si su funcional cae en el espacio generado por
    los funcionales visibles; si no, vale media exacta.
    """
    F = _funcionales(extra["P"])
    n = X.shape[0]
    err, n_tap = 0.0, 0
    for r in range(n):
        base = {}                                   # bit pivote -> fila
        vis = np.nonzero(~tap[r])[0]
        for c in vis:
            v = F[c]
            while v:
                h = v.bit_length() - 1
                if h in base:
                    v ^= base[h]
                else:
                    base[h] = v
                    v = 0
                    break
        tapadas = np.nonzero(tap[r])[0]
        indet = 0
        for c in tapadas:
            v = F[c]
            while v:
                h = v.bit_length() - 1
                if h not in base:
                    indet += 1
                    break
                v ^= base[h]
        err += 0.5 * indet
        n_tap += len(tapadas)
    return err / max(n_tap, 1)


def vara_lowdim(X, tap, extra, iters=600, lam=1e-3):
    """
    Vara alcanzable (NO piso demostrado). Margen maximo sobre las visibles:
        min (lam/2)||z||^2 + (1/|S|) sum_{i visible} max(0, 1 - s_i w_i·z)
    resuelto con Pegasos, y prediccion sign(w_m·z*) en las tapadas. Conoce W.
    La prediccion de Bayes exigiria integrar la gaussiana sobre el cono.
    """
    W = extra["W"]
    n = X.shape[0]
    vis = ~tap
    S = (X * vis).astype(np.float32)                # signo en visibles, 0 fuera
    nv = np.maximum(vis.sum(1, keepdims=True).astype(np.float32), 1.0)
    z = np.zeros((n, K), dtype=np.float32)
    for t in range(1, iters + 1):
        eta = 1.0 / (lam * t)
        M = z @ W.T                                 # (n, DIM)
        act = ((S * M) < 1.0) & vis
        g = ((act * S).astype(np.float32) @ W) / nv
        z = (1.0 - eta * lam) * z + eta * g
    pred = np.sign(z @ W.T)
    pred[pred == 0] = 1.0
    n_tap = int(tap.sum())
    return float(((pred != X) & tap).sum()) / max(n_tap, 1)


ORACULO = {"random": vara_random, "oversamp": vara_oversamp,
           "markov": vara_markov, "code": vara_code, "lowdim": vara_lowdim}


# =========================================================================== #
# 3. COMPARACION CON LO MEDIDO
# =========================================================================== #
def lee_medidos():
    """
    Filas (origen, fuente, arm, L, p, ber_tapados). Las dos corridas se
    devuelven etiquetadas y NO se fusionan: son experimentos distintos.
    """
    import csv
    filas = []
    rutas = [("denoising", os.path.join(DATA, "denoising", "denoising_resumen.csv")),
             ("factorial", os.path.join(DATA, "combinado", "combinado.csv"))]
    for origen, ruta in rutas:
        if not os.path.exists(ruta):
            continue
        for r in csv.DictReader(open(ruta)):
            v = r.get("ber_tapados", "")
            if v in ("", "nan", "NaN"):
                continue
            filas.append(dict(origen=origen, fuente=r["fuente"],
                              arm=r.get("arm", "ladder_random"),
                              L=int(r["L"]), p=float(r["p"]),
                              ber_tapados=float(v)))
    return filas


def compara(varas, medidos):
    """Imprime medido vs vara y la fraccion del camino realmente alcanzable."""
    if not medidos:
        print("\n(no hay CSV de corridas medidas; solo se escribio la vara)")
        return []
    agr = {}
    for f in medidos:
        agr.setdefault((f["origen"], f["fuente"], f["arm"], f["L"], f["p"]),
                       []).append(f["ber_tapados"])

    print("\n" + "=" * 100)
    print("MEDIDO CONTRA LA VARA   fraccion = (0.5 - medido) / (0.5 - vara)")
    print("=" * 100)
    print(f"{'origen':11s}{'fuente':9s}{'arm':15s}{'L':>5s}{'p':>6s}"
          f"{'medido':>9s}{'vara':>9s}{'fraccion':>10s}  tipo de vara")
    salida = []
    for k in sorted(agr, key=lambda x: (x[0], x[1], x[2], x[4], x[3])):
        origen, fte, arm, L, p = k
        v = varas.get((fte, p))
        if v is None:
            continue
        med = float(np.mean(agr[k]))
        margen = 0.5 - v["vara"]
        frac = "n/a" if margen <= 1e-9 else f"{(0.5 - med) / margen * 100:8.1f}%"
        print(f"{origen:11s}{fte:9s}{arm:15s}{L:5d}{p:6.2f}"
              f"{med:9.4f}{v['vara']:9.4f}{frac:>10s}  {TIPO_VARA[fte]}")
        salida.append(dict(origen=origen, fuente=fte, arm=arm, L=L, p=p,
                           ber_tapados=med, vara=v["vara"],
                           fraccion=None if margen <= 1e-9 else (0.5 - med) / margen,
                           tipo_vara=TIPO_VARA[fte]))
    print("\n  fraccion 0% = no infiere nada (adivinar).  100% = iguala al oraculo.")
    print("  sobre `lowdim` la vara es alcanzable, no un piso: pasar del 100% es")
    print("  posible en principio y significaria que el margen maximo no es optimo.")
    return salida


# =========================================================================== #
# 4. MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fuentes", default="code,lowdim,markov,oversamp,random")
    ap.add_argument("--p", default="0.1,0.25,0.5")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--n", type=int, default=0, help="muestras por celda; 0 = por fuente")
    ap.add_argument("--iters", type=int, default=600, help="pasos de Pegasos (lowdim)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sin-comparar", action="store_true")
    a = ap.parse_args()
    fuentes = a.fuentes.split(",")
    ps = [float(x) for x in a.p.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.quick:
        seeds, a.iters = seeds[:1], 150

    print("=" * 100)
    print("VARA DE ber_tapados  |  mejor inferencia posible de los bits enmascarados")
    print(f"semillas={seeds}  p={ps}  (mascara Bernoulli i.i.d., como en denoising.py)")
    print("=" * 100)
    print(f"{'fuente':10s}{'p':>6s}{'vara':>9s}{'sd':>9s}{'n':>7s}{'seg':>7s}  tipo")

    varas, filas = {}, []
    for fte in fuentes:
        n = a.n or N_POR_FUENTE[fte]
        if a.quick:
            n = max(20, n // 10)
        for p in ps:
            t0 = time.time()
            vals = []
            for seed in seeds:
                X, extra = fuente(fte, n, seed)
                tap = mascara(n, p, np.random.default_rng(10_000 + seed))
                f = ORACULO[fte]
                vals.append(f(X, tap, extra, iters=a.iters)
                            if fte == "lowdim" else f(X, tap, extra))
            m, s = float(np.mean(vals)), float(np.std(vals))
            varas[(fte, p)] = dict(vara=m, sd=s, n=n, tipo=TIPO_VARA[fte])
            filas.append(dict(fuente=fte, p=p, vara=m, sd=s, n=n,
                              seeds=seeds, tipo=TIPO_VARA[fte]))
            print(f"{fte:10s}{p:6.2f}{m:9.4f}{s:9.4f}{n:7d}"
                  f"{time.time() - t0:7.1f}  {TIPO_VARA[fte]}", flush=True)

    comparacion = [] if a.sin_comparar else compara(varas, lee_medidos())

    os.makedirs(DATA, exist_ok=True)
    destino = os.path.join(DATA, "varas_tapados.json")
    json.dump(dict(script="code/vara_tapados.py",
                   fecha_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   numpy=np.__version__, config=vars(a),
                   varas=filas, comparacion=comparacion),
              open(destino, "w"), indent=1)
    print(f"\n[guardado] {destino}")


if __name__ == "__main__":
    main()
