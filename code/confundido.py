#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONFUNDIDO -- separar "mejor checkpoint" de "arquitectura".

rbm_stack mostro que una escalera estrecha (500-250-125-70-35, sigmoides) con
inicializacion aleatoria da 0.0135 en markov L=250, mientras v4 direct (MLP
ancho 1536x4, GELU+LayerNorm) daba 0.0345. Dos explicaciones posibles:

  (1) CHECKPOINT: v4 evaluaba solo al final y era inestable en L=250. El MLP
      ancho alcanza ~0.013 en algun momento y despues se degrada.
  (2) ARQUITECTURA: la escalera estrecha es genuinamente mejor a tasas altas.

Diseno: dos modelos, PROTOCOLO IDENTICO (se importa finetune() de rbm_stack,
asi que pasos, lr, batch, validacion y seleccion de checkpoint son los mismos).

  ancho     = A.PlainAE(500, L, 500, 1536, 4)   <- el de v4
  escalera  = rbm_stack.LadderAE                <- el de rbm_stack, init aleatoria

Para cada uno se reporta el BER al MEJOR checkpoint y al FINAL. La diferencia
final - mejor es la inestabilidad del modelo.

Lectura:
  ancho_mejor ~ escalera_mejor   -> era el checkpoint: v4 estaba degradado
  ancho_mejor ~ v4 (0.0345)      -> era la arquitectura
  entre medias                   -> las dos cosas

Uso:
    python3 confundido.py --device cuda
"""
import argparse, json, os, sys, time
import numpy as np
import torch

try:
    import rbm_stack as R
except ModuleNotFoundError:
    sys.exit("[!] no encuentro rbm_stack.py: pon este script en la misma carpeta")
A = R.A          # el arnes (ae2 o ae_telecom_v4) que rbm_stack ya resolvio

# referencia v4 direct (media de 4 semillas, sin mejor checkpoint) y varas
V4 = {("markov", 35): 0.1498, ("markov", 125): 0.0501, ("markov", 250): 0.0345,
      ("lowdim", 250): 0.0653}
VARA = {("markov", 35): 0.1531, ("markov", 125): 0.0488, ("markov", 250): 0.0250,
        ("lowdim", 250): 0.0492}
CASOS = [("markov", 250), ("markov", 125), ("lowdim", 250), ("markov", 35)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="./confundido")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--ft_steps", type=int, default=30000)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.outdir, exist_ok=True)

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               n_train=400000, n_test=20000,
               batch=4096, lr=1e-3, wd=1e-4, ft_steps=a.ft_steps)
    print("=" * 78)
    print("CONFUNDIDO | mismo protocolo que rbm_stack.finetune, solo cambia el modelo")
    print(f"ft_steps={cfg['ft_steps']}  semillas={a.seeds}  device={dev}")
    print("=" * 78)

    filas = []
    for fuente, L in CASOS:
        depth = R.LADDER.index(L) + 1
        print(f"\n[{fuente} L={L}]  v4 direct={V4[(fuente, L)]:.4f}  vara={VARA[(fuente, L)]:.4f}")
        print(f"    {'seed':>4s} {'modelo':10s} {'mejor':>8s} {'final':>8s} {'inestab.':>9s} {'@step':>6s}")
        for seed in a.seeds:
            Xtr, Xrest, meta, _ = A.build_dataset(fuente, cfg["n_train"], cfg["n_test"] * 2,
                                                  500, cfg, seed)
            Xva, Xte = Xrest[:cfg["n_test"]], Xrest[cfg["n_test"]:]
            Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)
            for nombre, mk in (("ancho", lambda: A.PlainAE(500, L, 500, 1536, 4)),
                               ("escalera", lambda: R.LadderAE(R.build_ladder(depth)))):
                torch.manual_seed(seed)
                m = mk().to(dev)
                t0 = time.time()
                r = R.finetune(m, Xtr, Xva, Xte, cfg)
                ines = r["ber_test_final"] - r["ber_test"]
                print(f"    {seed:4d} {nombre:10s} {r['ber_test']:8.4f} {r['ber_test_final']:8.4f} "
                      f"{ines:+9.4f} {r['mejor_step']:6d}  ({time.time()-t0:.0f}s)", flush=True)
                filas.append(dict(fuente=fuente, L=L, seed=seed, modelo=nombre,
                                  mejor=r["ber_test"], final=r["ber_test_final"],
                                  inestabilidad=ines, step=r["mejor_step"]))
                del m
                if dev == "cuda": torch.cuda.empty_cache()
            del Xtr, Xva, Xte

    json.dump(filas, open(os.path.join(a.outdir, "confundido.json"), "w"), indent=1)

    print("\n" + "=" * 78)
    print("VEREDICTO  (media sobre semillas)")
    print("=" * 78)
    print(f"{'caso':14s}{'v4':>8s}{'ancho+ckpt':>12s}{'ancho final':>13s}{'escalera':>10s}{'vara':>8s}  explicacion")
    for fuente, L in CASOS:
        s = [f for f in filas if f["fuente"] == fuente and f["L"] == L]
        an = np.mean([f["mejor"] for f in s if f["modelo"] == "ancho"])
        af = np.mean([f["final"] for f in s if f["modelo"] == "ancho"])
        es = np.mean([f["mejor"] for f in s if f["modelo"] == "escalera"])
        v4, va = V4[(fuente, L)], VARA[(fuente, L)]
        # a que se parece mas el MLP ancho con checkpoint?
        d_esc, d_v4 = abs(an - es), abs(an - v4)
        if d_esc < 0.003 and d_v4 > 0.006:
            ex = "CHECKPOINT (v4 estaba degradado)"
        elif d_v4 < 0.003 and d_esc > 0.006:
            ex = "ARQUITECTURA (la escalera es mejor)"
        elif an < es - 0.003:
            ex = "el ancho GANA a la escalera con checkpoint"
        else:
            ex = "mixto: parte checkpoint, parte arquitectura"
        print(f"{fuente+' L='+str(L):14s}{v4:8.4f}{an:12.4f}{af:13.4f}{es:10.4f}{va:8.4f}  {ex}")

    print("\n  inestabilidad = final - mejor. Si es grande en 'ancho' y ~0 en")
    print("  'escalera', el MLP ancho se degrada tras su pico y v4 lo medía degradado.")


if __name__ == "__main__":
    main()
