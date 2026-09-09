#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMBRAL -- la ultima pregunta del proyecto.

markov L=250 con preentrenamiento RBM da BER 0.0107. El umbral corregible por
FEC es 0.0100. Dos vias para cruzarlo, ambas baratas:

  (7) ajuste fino con lr pequeno: los preentrenados alcanzan su pico antes del
      paso 6000 y luego se degradan con lr=1e-3. Con 1e-4 podrian conservar
      mas de lo ganado.
  (8) salida blanda: si los errores se concentran en los bits de baja
      confianza, el BER del 90% mas confiable puede quedar bajo 1e-2 aunque el
      BER total no. Es lo que aprovecha un FEC de decision blanda.

Se reporta, por semilla y por lr: BER total, BER del 90% y del 50% mas
confiables, y el veredicto contra 1e-2.

Reutiliza rbm_stack (pretrain_rbm_stack, unroll, finetune) sin cambios.

Uso:
    python3 umbral.py --device cuda
"""
import argparse, json, os, time
import numpy as np
import torch

try:
    import rbm_stack as R
except ModuleNotFoundError:
    raise SystemExit("[!] pon este script junto a rbm_stack.py")
A = R.A
if not hasattr(A, "ber_by_confidence"):
    raise SystemExit("[!] el arnes no expone ber_by_confidence (esta en ae_telecom_v4 / ae2)")

UMBRAL = 1e-2


@torch.no_grad()
def evalua(m, X):
    m.eval()
    lg = m(X).float()
    out = {"ber": A.ber_hard(lg, X)}
    out.update(A.ber_by_confidence(lg, X, fracs=(0.5, 0.9, 1.0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="./umbral")
    ap.add_argument("--fuente", default="markov")
    ap.add_argument("--L", type=int, default=250)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--lrs", nargs="*", type=float, default=[1e-3, 1e-4])
    ap.add_argument("--ft_steps", type=int, default=30000)
    ap.add_argument("--rbm_steps", type=int, default=10000)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    depth = R.LADDER.index(a.L) + 1

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               n_train=400000, n_test=20000, batch=4096, wd=1e-4,
               ft_steps=a.ft_steps, rbm_batch=256, rbm_lr=0.05, rbm_wd=1e-4,
               rbm_mom0=0.5, rbm_mom1=0.9, rbm_steps=a.rbm_steps, ae_lr=1e-3)

    print("=" * 76)
    print(f"UMBRAL | {a.fuente} L={a.L} | stacked_rbm | umbral FEC = {UMBRAL}")
    print(f"semillas={a.seeds}  lrs={a.lrs}  ft_steps={a.ft_steps}  device={dev}")
    print("=" * 76)
    print(f"{'seed':>4s} {'lr':>7s} {'BER':>8s} {'top90':>8s} {'top50':>8s} {'@step':>6s}  veredicto")

    filas = []
    for seed in a.seeds:
        Xtr, Xrest, meta, _ = A.build_dataset(a.fuente, cfg["n_train"], cfg["n_test"] * 2,
                                              500, cfg, seed)
        Xva, Xte = Xrest[:cfg["n_test"]], Xrest[cfg["n_test"]:]
        Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        etapas_rbm, _ = R.pretrain_rbm_stack(Xtr, cfg, gen)
        for lr in a.lrs:
            c = dict(cfg, lr=lr)
            torch.manual_seed(seed)
            m = R.unroll("stacked_rbm", depth, etapas_rbm, None, dev)
            t0 = time.time()
            r = R.finetune(m, Xtr, Xva, Xte, c)          # deja el mejor checkpoint cargado
            e = evalua(m, Xte)
            cruza_total = e["ber"] < UMBRAL
            cruza_blando = e["ber_top90"] < UMBRAL
            v = ("CRUZA con BER total" if cruza_total else
                 "cruza con salida blanda (top90)" if cruza_blando else "no cruza")
            print(f"{seed:4d} {lr:7.0e} {e['ber']:8.4f} {e['ber_top90']:8.4f} "
                  f"{e['ber_top50']:8.4f} {r['mejor_step']:6d}  {v}   ({time.time()-t0:.0f}s)",
                  flush=True)
            filas.append(dict(fuente=a.fuente, L=a.L, seed=seed, lr=lr, step=r["mejor_step"],
                              **e))
            del m
            if dev == "cuda": torch.cuda.empty_cache()
        del Xtr, Xva, Xte

    json.dump(filas, open(os.path.join(a.outdir, "umbral.json"), "w"), indent=1)

    print("\n" + "=" * 76)
    print("RESUMEN  (media +/- sd sobre semillas)")
    print("=" * 76)
    for lr in a.lrs:
        s = [f for f in filas if f["lr"] == lr]
        b = np.array([f["ber"] for f in s]); t9 = np.array([f["ber_top90"] for f in s])
        t5 = np.array([f["ber_top50"] for f in s])
        n_tot = int((b < UMBRAL).sum()); n_bl = int((t9 < UMBRAL).sum())
        print(f"  lr={lr:.0e}: BER {b.mean():.4f}+/-{b.std():.4f} | "
              f"top90 {t9.mean():.4f}+/-{t9.std():.4f} | top50 {t5.mean():.4f}")
        print(f"           cruza 1e-2 con BER total en {n_tot}/{len(s)} semillas, "
              f"con salida blanda en {n_bl}/{len(s)}")

    mejor = min(filas, key=lambda f: f["ber"])
    print(f"\n  Mejor corrida: seed={mejor['seed']} lr={mejor['lr']:.0e} "
          f"BER={mejor['ber']:.4f} top90={mejor['ber_top90']:.4f}")
    todos_total = all(f["ber"] < UMBRAL for f in filas if f["lr"] == min(a.lrs))
    todos_blando = all(f["ber_top90"] < UMBRAL for f in filas)
    if todos_total:
        print("  -> CRUZA el umbral con BER total en todas las semillas.")
        print("     Titular: viable en una franja estrecha, sin necesitar salida blanda.")
    elif todos_blando:
        print("  -> Cruza el umbral con salida blanda en todas las semillas.")
        print("     Titular: viable con FEC de decision blanda, en una franja estrecha.")
    else:
        print("  -> No cruza de forma consistente. El titular de inviabilidad se mantiene,")
        print("     con este punto como el mas cercano al umbral.")


if __name__ == "__main__":
    main()
