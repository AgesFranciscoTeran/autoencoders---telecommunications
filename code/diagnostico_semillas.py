#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostico_semillas.py — ¿que semillas divergieron?

Contexto: el contenedor murio a mitad de la rejilla, se perdieron 41 celdas no
sincronizadas a disco, y al reanudar se re-ejecutaron. Los valores finales son
sistematicamente PEORES que los del parcial de 119 celdas, y la desviacion de
la hipotesis primaria paso de 0.0001 a 0.0191. Eso no es ruido entre semillas:
apunta a divergencias en las celdas re-ejecutadas.

Este script no vuelve a entrenar nada. Lee el JSONL y aplica tres controles que
localizan celdas no fiables:

  1. SEMILLA ATIPICA. Por cada (fuente, L, arm, p) se compara cada semilla con
     la mediana de las demas. Una semilla que se desvia mucho mas que el resto
     en muchas celdas es una semilla divergida, no una muestra legitima.

  2. MONOTONIA. Mas bits SIEMPRE debe dar menos BER. Una violacion dentro de una
     misma (fuente, arm, p, seed) delata una corrida que no convergio. Es el
     mismo control que ya detecto problemas en el barrido v4.

  3. CHECKPOINT AL FINAL. Si el mejor checkpoint esta en los primeros pasos, el
     entrenamiento se degrado despues: senal de inestabilidad.

Salida: lista de celdas a re-ejecutar, con el comando concreto.

USO
    python3 code/diagnostico_semillas.py
    python3 code/diagnostico_semillas.py --excluir 3    # analisis sin esa semilla
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL = os.path.join(RAIZ, "data", "combinado", "combinado.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=JSONL)
    ap.add_argument("--excluir", default="", help="semillas a excluir, p.ej. 3")
    a = ap.parse_args()
    excl = {int(x) for x in a.excluir.split(",") if x.strip()}

    filas = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    seeds = sorted({r["seed"] for r in filas})
    print(f"celdas: {len(filas)}  semillas: {seeds}"
          + (f"  (excluyendo {sorted(excl)})" if excl else ""))

    # ---------------- 1. semilla atipica ----------------
    print("\n" + "=" * 78)
    print("1. SEMILLA ATIPICA  (desviacion respecto a la mediana de las demas)")
    print("=" * 78)
    celdas = defaultdict(dict)
    for r in filas:
        celdas[(r["fuente"], r["L"], r["arm"], r["p"])][r["seed"]] = r
    desv = defaultdict(list)
    for k, porsemilla in celdas.items():
        if len(porsemilla) < 3:
            continue
        for s, r in porsemilla.items():
            otras = [q["ber_test"] for t, q in porsemilla.items() if t != s]
            med = np.median(otras)
            rel = (r["ber_test"] - med) / max(med, 1e-9)
            desv[s].append(rel)
    print(f"  {'semilla':>8s}{'desv. mediana':>15s}{'desv. p90':>12s}"
          f"{'celdas >20 % peor':>20s}")
    sospechosas = []
    for s in seeds:
        v = np.array(desv[s])
        peor = int((v > 0.20).sum())
        print(f"  {s:>8d}{np.median(v)*100:>14.1f}%{np.percentile(v,90)*100:>11.1f}%"
              f"{peor:>13d}/{len(v)}")
        if peor > 0.25 * len(v):
            sospechosas.append(s)
    if sospechosas:
        print(f"\n  [!] semilla(s) sospechosa(s): {sospechosas}")
        print("      mas de un cuarto de sus celdas estan >20 % por encima de las demas")
    else:
        print("\n  ninguna semilla se desvia de forma sistematica")

    # ---------------- 2. monotonia ----------------
    print("\n" + "=" * 78)
    print("2. MONOTONIA  (mas bits debe dar menos BER)")
    print("=" * 78)
    grupos = defaultdict(dict)
    for r in filas:
        if r["seed"] in excl:
            continue
        grupos[(r["fuente"], r["arm"], r["p"], r["seed"])][r["L"]] = r["ber_test"]
    viol = []
    for k, porL in grupos.items():
        Ls = sorted(porL)
        if len(Ls) < 2:
            continue
        for i in range(len(Ls) - 1):
            if porL[Ls[i + 1]] > porL[Ls[i]] + 1e-4:      # mas bits, peor BER
                viol.append((k, Ls[i], Ls[i + 1], porL[Ls[i]], porL[Ls[i + 1]]))
    if viol:
        print(f"  {len(viol)} violaciones:")
        porsemilla = defaultdict(int)
        for (f, arm, p, s), La, Lb, ba, bb in viol:
            porsemilla[s] += 1
            print(f"    {f:8s} {arm:>14s} p={p:<5} seed={s}  "
                  f"L{La}={ba:.4f} -> L{Lb}={bb:.4f}")
        print(f"\n  por semilla: {dict(porsemilla)}")
    else:
        print("  ninguna: todas las curvas son monotonas")

    # ---------------- 3. checkpoint temprano ----------------
    print("\n" + "=" * 78)
    print("3. CHECKPOINT TEMPRANO  (el mejor modelo llego en los primeros pasos)")
    print("=" * 78)
    pasos = defaultdict(list)
    for r in filas:
        pasos[r["seed"]].append(r.get("mejor_step", np.nan))
    print(f"  {'semilla':>8s}{'mediana @step':>15s}{'celdas <5000':>15s}")
    for s in seeds:
        v = np.array([x for x in pasos[s] if not np.isnan(x)])
        print(f"  {s:>8d}{np.median(v):>15.0f}{int((v<5000).sum()):>10d}/{len(v)}")

    # ---------------- resumen ----------------
    print("\n" + "=" * 78)
    if sospechosas:
        print("QUE HACER")
        print("=" * 78)
        ss = ",".join(str(s) for s in sospechosas)
        print(f"  Opcion A — re-ejecutar la(s) semilla(s) sospechosa(s):")
        print(f"    python3 - <<'EOF'")
        print(f"import json")
        print(f"P='data/combinado/combinado.jsonl'")
        print(f"ls=[l for l in open(P) if l.strip() and json.loads(l)['seed'] not in {set(sospechosas)}]")
        print(f"open(P,'w').writelines(ls)")
        print(f"print(f'quedan {{len(ls)}} celdas')")
        print(f"EOF")
        print(f"    nohup python3 -u code/combinado.py > logs/rerun.log 2>&1 &")
        print(f"\n  Opcion B — analizar sin ellas, declarandolo:")
        print(f"    python3 code/analizar_combinado.py --excluir {ss}")
        print(f"\n  A es preferible: excluir semillas a posteriori por su resultado")
        print(f"  es seleccion, aunque el criterio sea tecnico. Solo vale si se")
        print(f"  declara y se justifica con los controles de arriba.")
    else:
        print("Sin semillas sospechosas: los numeros se pueden usar tal cual.")


if __name__ == "__main__":
    main()
