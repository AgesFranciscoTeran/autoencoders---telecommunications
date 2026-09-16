#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combinado.py — ¿se componen el preentrenamiento RBM y el denoising?

Los dos ayudan por separado, pero actuan sobre cosas distintas:
  - el preentrenamiento RBM cambia DONDE EMPIEZA la optimizacion
  - el enmascarado cambia QUE OBJETIVO se optimiza
Nada garantiza que se sumen. Podrian ser redundantes (ambos regularizan) o
incluso estorbarse (el ruido destruye la inicializacion cuidadosa).

DISENO: factorial 2x3 completo — inicializacion x nivel de ruido.

    init in {aleatoria, RBM}   x   p in {0, 0.1, 0.25}

Eso permite medir no solo la combinacion sino la INTERACCION: si el efecto del
ruido es distinto partiendo de RBM que partiendo de aleatoria.

CONFIRMATORIO vs EXPLORATORIO
-----------------------------
Con 13 celdas x 6 condiciones x 4 semillas, buscar sorpresas encontraria algunas
por azar. Por eso se separan los dos roles, como en un ensayo clinico:

  CONFIRMATORIO — una sola hipotesis primaria, registrada, sobre un punto
    elegido de antemano: `markov` L=250, el mejor del proyecto y el unico que
    roza el umbral FEC. Lo que salga ahi cuenta como resultado.

  EXPLORATORIO — todo lo demas. Sirve para generar hipotesis, no para
    confirmarlas. Cualquier hallazgo aqui se marca como tal y EXIGE replica en
    semillas frescas (4,5,6,7) antes de entrar en el cuaderno como resultado.
    El script imprime automaticamente el comando de replica.

PREDICCIONES REGISTRADAS (antes de correr)
------------------------------------------
Primaria (confirmatoria), `markov` L=250:
  H1. RBM + denoising NO mejora sobre RBM solo (0.0107). El denoising ya
      empeoraba ahi con init aleatoria (0.0135 -> 0.0140), y la interaccion no
      deberia cambiar el signo.
  H2. Ninguna condicion cruza 1e-2 con BER TOTAL. Con salida blanda, todas las
      que esten cerca lo cruzan, como ya ocurria.

Exploratorias (generan hipotesis, no las confirman):
  H3. `lowdim` L=250: RBM + p=0.10 mejora sobre ambos por separado (0.0369 y
      0.0346). Es el unico sitio donde espero composicion positiva.
  H4. Interaccion negativa en general: donde los dos ayudan por separado, juntos
      dan menos que la suma. Ambos actuan como regularizadores.
  H5. `code`: nada se mueve y `ber_tapados` sigue en 0.5 en las seis condiciones.
      Ni cambiando el punto de partida se encuentra la estructura algebraica.

USO
---
    python3 code/combinado.py                      # rejilla completa, ~5 h
    python3 code/combinado.py --solo-primaria      # solo markov L=250, ~15 min
    python3 code/combinado.py --seeds 4,5,6,7      # replica de un hallazgo
    python3 code/combinado.py --quick              # humo
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbm_stack as R
import denoising as D
A = R.A

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "data", "combinado")

PRIMARIA = ("markov", 250)
L_POR_FUENTE = {"code": [250, 125, 70, 35], "lowdim": [250, 125, 70, 35],
                "markov": [250, 125, 70, 35], "random": [250]}
ARMS = ["ladder_random", "stacked_rbm"]

# referencias conocidas (mejor valor previo por punto y metodo)
PREV = {("markov", 250): {"rbm": 0.0107, "denoise": 0.0140, "ninguno": 0.0135},
        ("lowdim", 250): {"rbm": 0.0369, "denoise": 0.0346, "ninguno": 0.0386}}
UMBRAL = 1e-2


def corre_celda(fuente, L, arm, p, seed, Xtr, Xva, Xte, etapas_rbm, cfg):
    dev = cfg["device"]
    depth = R.LADDER.index(L) + 1
    torch.manual_seed(seed)
    m = R.unroll(arm, depth, etapas_rbm, None, dev)
    gen_datos = torch.Generator(device=dev); gen_datos.manual_seed(seed)
    gen_ruido = torch.Generator(device=dev)
    gen_ruido.manual_seed(seed * 7919 + int(round(p * 1000)))
    r = D.finetune_denoising(m, Xtr, Xva, Xte, cfg, p, "mask", gen_datos, gen_ruido)
    del m
    if dev == "cuda": torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuentes", default="markov,lowdim,code,random")
    ap.add_argument("--p", default="0,0.1,0.25")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--solo-primaria", action="store_true")
    ap.add_argument("--n_train", type=int, default=400_000)
    ap.add_argument("--ft_steps", type=int, default=30_000)
    ap.add_argument("--rbm_steps", type=int, default=10_000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.fuentes = a.fuentes.split(",")
    a.p = [float(x) for x in a.p.split(",")]
    a.seeds = [int(x) for x in a.seeds.split(",")]
    if 0.0 not in a.p:
        sys.exit("--p debe incluir 0: es el control del eje de ruido")
    if a.solo_primaria:
        a.fuentes = [PRIMARIA[0]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               n_train=a.n_train, n_test=20000, batch=4096, lr=1e-3, wd=1e-4,
               ft_steps=a.ft_steps, bn_batches=100,
               rbm_batch=256, rbm_lr=0.05, rbm_wd=1e-4,
               rbm_mom0=0.5, rbm_mom1=0.9, rbm_steps=a.rbm_steps)
    if a.quick:
        cfg.update(n_train=40000, n_test=5000, ft_steps=600, rbm_steps=400)
        a.seeds = a.seeds[:1]
    os.makedirs(SALIDA, exist_ok=True)
    JSONL = os.path.join(SALIDA, "combinado.jsonl")

    def clave(f, L, arm, p, seed):
        return f"{f}|{L}|{arm}|{p}|{seed}"

    hechas, filas = set(), []
    if os.path.exists(JSONL):
        for ln in open(JSONL):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            filas.append(r)
            hechas.add(clave(r["fuente"], r["L"], r["arm"], r["p"], r["seed"]))
        if hechas:
            print(f"[reanudar] {len(hechas)} celdas ya hechas en {JSONL}")

    plan = [(f, L) for f in a.fuentes
            for L in (L_POR_FUENTE[f] if not a.solo_primaria else [PRIMARIA[1]])]
    n_cel = len(plan) * len(ARMS) * len(a.p) * len(a.seeds)
    print("=" * 92)
    print("COMBINADO | factorial 2x3: inicializacion x ruido de enmascaramiento")
    print(f"puntos={plan}")
    print(f"arms={ARMS}  p={a.p}  seeds={a.seeds}  -> {n_cel} celdas  (~{n_cel*60/3600:.1f} h)")
    print(f"PRIMARIA (confirmatoria): {PRIMARIA[0]} L={PRIMARIA[1]}  |  el resto es EXPLORATORIO")
    print("=" * 92)

    for fuente in a.fuentes:
        Ls = [PRIMARIA[1]] if a.solo_primaria else L_POR_FUENTE[fuente]
        for seed in a.seeds:
            pend = [(L, arm, p) for L in Ls for arm in ARMS for p in a.p
                    if clave(fuente, L, arm, p, seed) not in hechas]
            if not pend:
                print(f"[skip] {fuente} seed={seed}: completo")
                continue
            Xtr, Xrest, meta, _ = A.build_dataset(fuente, cfg["n_train"],
                                                  cfg["n_test"] * 2, 500, cfg, seed)
            Xva, Xte = Xrest[:cfg["n_test"]], Xrest[cfg["n_test"]:]
            Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)
            gen = torch.Generator(device=dev); gen.manual_seed(seed)
            print(f"\n[{fuente} seed={seed}]  preentrenando RBM...", flush=True)
            etapas_rbm, _ = R.pretrain_rbm_stack(Xtr, cfg, gen)
            print(f"    {'L':>4s} {'init':>14s} {'p':>5s} {'BER':>9s} {'top90':>8s} "
                  f"{'tapados':>8s} {'@step':>6s}")
            for (L, arm, p) in pend:
                t0 = time.time()
                r = corre_celda(fuente, L, arm, p, seed,
                                Xtr, Xva, Xte, etapas_rbm, cfg)
                print(f"    {L:4d} {arm:>14s} {p:5.2f} {r['ber_test']:9.4f} "
                      f"{r['ber_top90']:8.4f} {r['ber_tapados']:8.4f} "
                      f"{r['mejor_step']:6d}  ({time.time()-t0:.0f}s)", flush=True)
                fila = dict(fuente=fuente, L=L, arm=arm, p=p, seed=seed,
                            primaria=((fuente, L) == PRIMARIA), **r)
                filas.append(fila)
                with open(JSONL, "a") as fh:          # volcado inmediato: reanudable
                    fh.write(json.dumps(fila) + "\n")
                hechas.add(clave(fuente, L, arm, p, seed))
            del Xtr, Xva, Xte
            if dev == "cuda": torch.cuda.empty_cache()

    if not filas:
        sys.exit("[!] no hay filas: nada que resumir")
    import csv
    with open(os.path.join(SALIDA, "combinado.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)

    # ------------------------------------------------------------------ #
    def cel(f, L, arm, p):
        v = [r["ber_test"] for r in filas
             if r["fuente"] == f and r["L"] == L and r["arm"] == arm and r["p"] == p]
        return np.array(v) if v else None

    print("\n" + "=" * 92)
    print("CONFIRMATORIO — hipotesis primaria registrada")
    print("=" * 92)
    f, L = PRIMARIA
    base = cel(f, L, "stacked_rbm", 0.0)
    if base is not None:
        print(f"  {f} L={L}, referencia RBM sin ruido: {base.mean():.4f}")
        mejor = (base.mean(), "stacked_rbm", 0.0)
        for arm in ARMS:
            for p in a.p:
                c = cel(f, L, arm, p)
                if c is None: continue
                gana = int((c < base).sum()) if len(c) == len(base) else 0
                cruza = int((c < UMBRAL).sum())
                if c.mean() < mejor[0]: mejor = (c.mean(), arm, p)
                print(f"    {arm:>14s} p={p:<5} {c.mean():.4f} ± {c.std():.4f}  "
                      f"mejora sobre RBM en {gana}/{len(c)}  cruza 1e-2 en {cruza}/{len(c)}")
        c = cel(f, L, "stacked_rbm", 0.1)
        h1 = "CONFIRMADA" if c is None or c.mean() >= base.mean() else "REFUTADA"
        todos = np.concatenate([cel(f, L, ar, pp) for ar in ARMS for pp in a.p
                                if cel(f, L, ar, pp) is not None])
        h2 = "CONFIRMADA" if (todos >= UMBRAL).all() else "REFUTADA"
        print(f"\n  H1 (RBM+denoising no mejora sobre RBM solo): {h1}")
        print(f"  H2 (nada cruza 1e-2 con BER total): {h2}")
        print(f"  mejor condicion: {mejor[1]} p={mejor[2]} -> {mejor[0]:.4f}")

    print("\n" + "=" * 92)
    print("EXPLORATORIO — genera hipotesis, NO las confirma")
    print("=" * 92)
    print(f"{'fuente':8s}{'L':>5s}{'solo RBM':>10s}{'solo ruido':>12s}"
          f"{'ambos':>9s}{'suma esp.':>11s}{'interacc.':>11s}  nota")
    hallazgos = []
    for (f, L) in plan:
        if (f, L) == PRIMARIA: continue
        b00 = cel(f, L, "ladder_random", 0.0)
        if b00 is None: continue
        b10 = cel(f, L, "stacked_rbm", 0.0)
        pbest = min([p for p in a.p if p > 0],
                    key=lambda p: (cel(f, L, "ladder_random", p) or np.array([9])).mean())
        b01 = cel(f, L, "ladder_random", pbest)
        b11 = cel(f, L, "stacked_rbm", pbest)
        if any(x is None for x in (b10, b01, b11)): continue
        e_rbm, e_ruido = b10.mean() - b00.mean(), b01.mean() - b00.mean()
        suma = b00.mean() + e_rbm + e_ruido
        inter = b11.mean() - suma
        mejor_prev = min(b00.mean(), b10.mean(), b01.mean())
        nota = ""
        if b11.mean() < mejor_prev - 2 * b11.std() and (b11 < b10).all() and (b11 < b01).all():
            nota = "SE COMPONEN (replicar)"
            hallazgos.append((f, L, pbest, b11.mean()))
        print(f"{f:8s}{L:5d}{b10.mean():10.4f}{b01.mean():12.4f}{b11.mean():9.4f}"
              f"{suma:11.4f}{inter:+11.4f}  {nota}")

    tap = [r["ber_tapados"] for r in filas
           if r["fuente"] == "code" and r["p"] > 0 and not np.isnan(r["ber_tapados"])]
    if tap:
        print(f"\n  H5 (`code`: ber_tapados sigue en 0.5 con cualquier init): "
              f"{'CONFIRMADA' if abs(np.mean(tap) - 0.5) < 0.02 else 'REFUTADA'}"
              f"  ({np.mean(tap):.4f})")

    if hallazgos:
        frescas = ",".join(str(s + max(a.seeds) + 1) for s in range(len(a.seeds)))
        print(f"\n  {len(hallazgos)} hallazgo(s) exploratorio(s). NO entran en el cuaderno")
        print("  sin replica en semillas frescas:")
        for f, L, p, b in hallazgos:
            print(f"    python3 code/combinado.py --fuentes {f} --p 0,{p} "
                  f"--seeds {frescas}    # {f} L={L}, esperado ~{b:.4f}")
    else:
        print("\n  Sin hallazgos exploratorios que superen el umbral de replica.")

    json.dump(dict(filas=filas, primaria=PRIMARIA, args=vars(a)),
              open(os.path.join(SALIDA, "combinado.json"), "w"), indent=1)
    print(f"\nescrito en {SALIDA}/")


if __name__ == "__main__":
    main()
