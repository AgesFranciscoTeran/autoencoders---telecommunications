#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analizar_combinado.py — analisis del factorial 2x3, separado de la corrida.

Se lee `data/combinado/combinado.jsonl` (una linea por celda, escrita en el
momento) y se rehace la estadistica. Separar el analisis de la ejecucion
permite corregir el criterio sin repetir 5 horas de GPU, y permite mirar
resultados parciales mientras corre.

CORRIGE EL DETECTOR DEL SCRIPT ORIGINAL
---------------------------------------
El resumen de `combinado.py` marcaba "SE COMPONEN" con esta condicion:

    b11.mean() < mejor_prev - 2*b11.std()  and  (b11 < b10).all() ...

Tres defectos:
  (1) con una sola semilla, std() = 0 y la condicion se reduce a "cualquier
      mejora": en el humo disparo en 8 de 12 puntos;
  (2) usa la desviacion de b11, no la de la DIFERENCIA pareada, que es la que
      corresponde cuando las condiciones comparten semilla;
  (3) no exige tamano de efecto, asi que una mejora de 0.0003 cuenta igual que
      una de 0.03.

Criterio de aqui, los tres a la vez:
  - gana en TODAS las semillas frente a las dos condiciones simples
  - diferencia pareada media por debajo de -3 sd de esa diferencia
  - mejora relativa de al menos 1 %

Sobre la correccion por comparaciones multiples: con 12 puntos exploratorios y
4 semillas, la probabilidad de un 4/4 espurio en una direccion dada es 1/16 por
punto, asi que cabe esperar ~0.75 falsos positivos solo con el criterio de
signo. De ahi las dos exigencias adicionales.

USO
    python3 code/analizar_combinado.py                  # lee el jsonl
    python3 code/analizar_combinado.py --parcial        # permite celdas faltantes
"""
import argparse, json, os, sys
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL = os.path.join(RAIZ, "data", "combinado", "combinado.jsonl")
PRIMARIA = ("markov", 250)
UMBRAL = 1e-2
MIN_REL = 0.01          # mejora relativa minima
MIN_SIGMA = 3.0         # sd de la diferencia pareada
SD_MIN = 1e-4           # suelo de ruido: sd pareada por debajo de esto no se
                        # penaliza. Una diferencia identica en las 4 semillas es
                        # la evidencia MAS fuerte, no la mas debil.


def cargar(path):
    filas = [json.loads(l) for l in open(path) if l.strip()]
    if not filas:
        sys.exit(f"[!] {path} vacio")
    return filas


def sd(v):
    """Desviacion muestral; 0.0 con una sola observacion (en vez de NaN)."""
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pareado(filas, f, L, arm, p, seeds):
    """BER por semilla, en el orden de `seeds`. None si falta alguna."""
    d = {r["seed"]: r["ber_test"] for r in filas
         if r["fuente"] == f and r["L"] == L and r["arm"] == arm and r["p"] == p}
    if not all(s in d for s in seeds):
        return None
    return np.array([d[s] for s in seeds])


def contrasta(mejor_simple, ambos):
    """Diferencia pareada: negativa = `ambos` es mejor."""
    dif = ambos - mejor_simple
    sd_dif = sd(dif)
    rel = dif.mean() / max(mejor_simple.mean(), 1e-9)
    gana_todas = bool((dif < 0).all())
    sigma_ok = dif.mean() < -MIN_SIGMA * max(sd_dif, SD_MIN)
    return dict(dif=float(dif.mean()), sd=float(sd_dif), rel=float(rel),
                gana=int((dif < 0).sum()), n=len(dif),
                significativo=bool(gana_todas and sigma_ok and rel < -MIN_REL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=JSONL)
    ap.add_argument("--parcial", action="store_true")
    a = ap.parse_args()
    filas = cargar(a.jsonl)
    seeds = sorted({r["seed"] for r in filas})
    ps = sorted({r["p"] for r in filas})
    puntos = sorted({(r["fuente"], r["L"]) for r in filas})
    print(f"celdas: {len(filas)}  semillas: {seeds}  p: {ps}  puntos: {len(puntos)}")
    if len(seeds) < 4:
        print(f"[aviso] solo {len(seeds)} semilla(s): los veredictos son provisionales.")
        print("        El criterio de composicion exige 4 semillas para tener sentido.")
    if len(filas) < 312 and not a.parcial:
        print(f"[aviso] faltan {312 - len(filas)} celdas. Usa --parcial para analizar igual.")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 90)
    print("CONFIRMATORIO — hipotesis primaria")
    print("=" * 90)
    f, L = PRIMARIA
    base = pareado(filas, f, L, "stacked_rbm", 0.0, seeds)
    if base is None:
        print("  faltan celdas de la primaria")
    else:
        print(f"  {f} L={L}. Referencia: RBM sin ruido = {base.mean():.4f} ± {sd(base):.4f}")
        print(f"  {'init':>14s} {'p':>5s} {'BER':>18s} {'vs RBM':>9s} {'gana':>6s} {'cruza 1e-2':>11s}")
        mejor = (base.mean(), "stacked_rbm", 0.0)
        for arm in ("ladder_random", "stacked_rbm"):
            for p in ps:
                c = pareado(filas, f, L, arm, p, seeds)
                if c is None: continue
                dif = c - base
                if c.mean() < mejor[0]: mejor = (c.mean(), arm, p)
                print(f"  {arm:>14s} {p:5.2f} {c.mean():8.4f} ± {sd(c):6.4f} "
                      f"{dif.mean():+9.4f} {int((dif<0).sum())}/{len(dif):<4d} "
                      f"{int((c<UMBRAL).sum())}/{len(c):>9d}")
        c = pareado(filas, f, L, "stacked_rbm", 0.1, seeds)
        h1 = "CONFIRMADA" if c is None or c.mean() >= base.mean() else "REFUTADA"
        todo = [pareado(filas, f, L, ar, pp, seeds) for ar in ("ladder_random", "stacked_rbm")
                for pp in ps]
        todo = np.concatenate([t for t in todo if t is not None])
        h2 = "CONFIRMADA" if (todo >= UMBRAL).all() else "REFUTADA"
        print(f"\n  H1 (RBM+ruido no mejora sobre RBM solo): {h1}")
        print(f"  H2 (nada cruza 1e-2 con BER total):      {h2}")
        print(f"  mejor condicion: {mejor[1]} p={mejor[2]} -> {mejor[0]:.4f}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 90)
    print("EXPLORATORIO — criterio corregido (pareado, 3 sd, 1 % relativo)")
    print("=" * 90)
    print(f"{'punto':14s}{'solo RBM':>10s}{'solo ruido':>11s}{'ambos':>9s}"
          f"{'interacc.':>10s}{'dif pareada':>13s}{'gana':>6s}  veredicto")
    hall = []
    for (fu, Lp) in puntos:
        if (fu, Lp) == PRIMARIA: continue
        b00 = pareado(filas, fu, Lp, "ladder_random", 0.0, seeds)
        b10 = pareado(filas, fu, Lp, "stacked_rbm", 0.0, seeds)
        if b00 is None or b10 is None: continue
        cand = [(p, pareado(filas, fu, Lp, "ladder_random", p, seeds)) for p in ps if p > 0]
        cand = [(p, v) for p, v in cand if v is not None]
        if not cand: continue
        pbest, b01 = min(cand, key=lambda t: t[1].mean())
        b11 = pareado(filas, fu, Lp, "stacked_rbm", pbest, seeds)
        if b11 is None: continue
        suma = b00.mean() + (b10.mean() - b00.mean()) + (b01.mean() - b00.mean())
        inter = b11.mean() - suma
        mejor_simple = np.minimum(b10, b01)
        r = contrasta(mejor_simple, b11)
        ver = "COMPONEN (replicar)" if r["significativo"] else \
              ("mejora dentro del ruido" if r["gana"] == r["n"] else "no componen")
        print(f"{fu+' L='+str(Lp):14s}{b10.mean():10.4f}{b01.mean():11.4f}{b11.mean():9.4f}"
              f"{inter:+10.4f}{r['dif']:+9.4f}±{r['sd']:.4f}{r['gana']}/{r['n']:<4d}  {ver}")
        if r["significativo"]:
            hall.append((fu, Lp, pbest, b11.mean(), r))

    tap = [r["ber_tapados"] for r in filas
           if r["fuente"] == "code" and r["p"] > 0 and not np.isnan(r.get("ber_tapados", np.nan))]
    if tap:
        print(f"\n  H5 (`code`: ber_tapados en 0.5 con cualquier init): "
              f"{'CONFIRMADA' if abs(np.mean(tap)-0.5) < 0.02 else 'REFUTADA'}  "
              f"({np.mean(tap):.4f}, n={len(tap)})")

    inters = []
    for (fu, Lp) in puntos:
        b00 = pareado(filas, fu, Lp, "ladder_random", 0.0, seeds)
        b10 = pareado(filas, fu, Lp, "stacked_rbm", 0.0, seeds)
        cand = [(p, pareado(filas, fu, Lp, "ladder_random", p, seeds)) for p in ps if p > 0]
        cand = [(p, v) for p, v in cand if v is not None]
        if b00 is None or b10 is None or not cand: continue
        pbest, b01 = min(cand, key=lambda t: t[1].mean())
        b11 = pareado(filas, fu, Lp, "stacked_rbm", pbest, seeds)
        if b11 is None: continue
        inters.append(b11.mean() - (b10.mean() + b01.mean() - b00.mean()))
    if inters:
        pos = sum(1 for i in inters if i > 0)
        print(f"\n  H4 (interaccion negativa: juntos dan menos que la suma): "
              f"{'REFUTADA' if pos > len(inters)/2 else 'CONFIRMADA'}"
              f"  ({pos}/{len(inters)} interacciones positivas, mediana {np.median(inters):+.4f})")

    if hall:
        frescas = ",".join(str(max(seeds) + 1 + i) for i in range(len(seeds)))
        print(f"\n  {len(hall)} hallazgo(s) que superan el criterio. Replica obligatoria:")
        for fu, Lp, p, b, r in hall:
            print(f"    python3 code/combinado.py --fuentes {fu} --L {Lp} --p 0,{p} "
                  f"--seeds {frescas}")
            print(f"      {fu} L={Lp}: esperado ~{b:.4f}, diferencia {r['dif']:+.4f} "
                  f"({r['rel']*100:+.1f} %)")
    else:
        print("\n  Ningun punto supera el criterio de composicion. Nada que replicar.")


if __name__ == "__main__":
    main()
