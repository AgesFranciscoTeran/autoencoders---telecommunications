#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GATE TEST 3 -- el modelo alcanza la solucion y luego la destruye.

Hallazgo del gate 2: tanh con lr=1e-3 llego a BER 0.0003 en la epoca 60 y
despues se autodestruyo (epoca 150: BER 0.4725, 47% de bits muertos). La
capacidad no esta en duda; el optimizador no se queda en la solucion.

Mecanismo: con sign() en el forward, escalar las preactivaciones no cambia la
salida, asi que la perdida es CIEGA a la escala. Pero el backward no lo es:
preactivaciones grandes anulan el gradiente. Existe una direccion degenerada por
la que los pesos crecen sin penalizacion mientras matan el gradiente propio.

Este script cambia tres cosas frente al gate 2:

 1. VALIDACION + MEJOR CHECKPOINT. Se separa validacion de test. Se guarda el
    mejor modelo segun validacion y se reporta su BER de TEST. Seleccionar por
    test seria hacer trampa.
 2. DIAGNOSTICO CORREGIDO. Se mide |u| (lo que entra al binarizador) ademas de
    |pre| (la preactivacion cruda). En las variantes normalizadas solo |u| es
    interpretable; el gate 2 medía |pre| y por eso daba lecturas confusas.
 3. VARIANTES SIN AFIN. LayerNorm y BatchNorm con elementwise_affine=False /
    affine=False: la ganancia aprendible puede crecer sin limite y reproducir
    la explosion, asi que se prueba con y sin.

Uso:
    python3 gate_test3.py --device cuda --epochs 600 --lr 1e-3
"""
import argparse, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ae2 as A

VARIANTES = ["tanh", "layernorm", "layernorm_sinafin", "batchnorm_sinafin"]


class Modelo(nn.Module):
    def __init__(self, dim, latent, hidden, depth, variante):
        super().__init__()
        self.variante = variante
        self.enc = A.mlp([dim] + [hidden] * depth + [latent])
        self.dec = A.mlp([latent] + [hidden] * depth + [dim])
        if variante == "layernorm":
            self.norm = nn.LayerNorm(latent)
        elif variante == "layernorm_sinafin":
            self.norm = nn.LayerNorm(latent, elementwise_affine=False)
        elif variante == "batchnorm_sinafin":
            self.norm = nn.BatchNorm1d(latent, affine=False)
        else:
            self.norm = None

    def forward(self, x):
        pre = self.enc(x)
        u = torch.tanh(pre) if self.variante == "tanh" else (
            self.norm(pre) if self.norm is not None else pre)
        z = A.BinarizeSTE.apply(u)
        return self.dec(z), pre, u, z


@torch.no_grad()
def evalua(m, X):
    m.eval()
    lg, pre, u, z = m(X)
    return dict(
        ber=A.ber_hard(lg.float(), X),
        # |u| es lo que ve el binarizador: el umbral que importa para el STE
        u_gt1=(u.abs().float() > 1).float().mean().item(),
        # |pre| crudo: detecta la explosion de escala aunque la norma la oculte
        pre_gt3=(pre.abs().float() > 3).float().mean().item(),
        pre_max=pre.abs().float().max().item(),
        vivos=(z.float().std(0) > 1e-3).float().mean().item(),
    )


def corre(variante, Xtr, Xva, Xte, cfg):
    dev = cfg["device"]
    torch.manual_seed(cfg["seed"])
    m = Modelo(Xtr.shape[1], cfg["latent"], cfg["hidden"], cfg["depth"], variante).to(dev)
    n, bs = Xtr.shape[0], cfg["batch"]
    spe = max(1, n // bs)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                              total_steps=cfg["epochs"] * spe,
                                              pct_start=0.1)
    amp = (dev == "cuda")
    mejor = {"val": 1.0, "ep": 0, "estado": None}
    cada = max(1, cfg["epochs"] // 20)

    print(f"\n  --- {variante} ---")
    print(f"    {'ep':>5s} {'BER_val':>8s} {'|u|>1':>7s} {'|pre|>3':>8s} "
          f"{'pre_max':>8s} {'vivos':>7s}")
    for ep in range(1, cfg["epochs"] + 1):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            xb = Xtr[perm[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg, _, _, _ = m(xb)
                loss = F.binary_cross_entropy_with_logits(lg, (xb + 1) * 0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()

        if ep % cada == 0 or ep == 1 or ep == cfg["epochs"]:
            d = evalua(m, Xva)
            if d["ber"] < mejor["val"]:
                mejor = {"val": d["ber"], "ep": ep,
                         "estado": copy.deepcopy(m.state_dict())}
            print(f"    {ep:5d} {d['ber']:8.4f} {d['u_gt1']*100:6.1f}% "
                  f"{d['pre_gt3']*100:7.1f}% {d['pre_max']:8.1f} {d['vivos']*100:6.1f}%")

    final_te = evalua(m, Xte)["ber"]
    m.load_state_dict(mejor["estado"])
    mejor_te = evalua(m, Xte)["ber"]
    return dict(variante=variante, final_test=final_te, mejor_test=mejor_te,
                mejor_val=mejor["val"], mejor_ep=mejor["ep"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--latent", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variantes", nargs="*", default=VARIANTES)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               hidden=1536, depth=4, batch=4096, lr=a.lr, wd=a.wd,
               epochs=a.epochs, latent=a.latent, seed=a.seed)

    # validacion SEPARADA de test: seleccionar el checkpoint por test seria trampa
    Xtr, Xrest, meta, _ = A.build_dataset("oversamp", 200000, 40000, 500, cfg, a.seed)
    Xva, Xte = Xrest[:20000], Xrest[20000:]
    Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)

    print("=" * 76)
    print(f"GATE 3  |  oversamp  H={meta['entropy_bits']:.0f} bits  "
          f"latente={a.latent}  ->  BER correcto ~ 0")
    print(f"epochs={a.epochs}  lr={a.lr}  wd={a.wd}  seed={a.seed}  device={dev}")
    print(f"train={Xtr.shape[0]}  val={Xva.shape[0]}  test={Xte.shape[0]}")
    print("=" * 76)

    res = []
    for v in a.variantes:
        t0 = time.time()
        r = corre(v, Xtr, Xva, Xte, cfg)
        res.append(r)
        print(f"    -> final(test)={r['final_test']:.4f}  "
              f"mejor(test)={r['mejor_test']:.4f} @ep{r['mejor_ep']}  "
              f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 76)
    print(f"{'variante':>20s} {'final':>9s} {'mejor':>9s} {'@ep':>6s} "
          f"{'destruccion':>12s} {'veredicto':>10s}")
    for r in res:
        destr = r["final_test"] - r["mejor_test"]
        ver = "APRUEBA" if r["mejor_test"] < 0.005 else "reprueba"
        print(f"{r['variante']:>20s} {r['final_test']:9.4f} {r['mejor_test']:9.4f} "
              f"{r['mejor_ep']:6d} {destr:+12.4f} {ver:>10s}")

    ok = [r for r in res if r["mejor_test"] < 0.005]
    print()
    if ok:
        b = min(ok, key=lambda r: r["mejor_test"])
        print(f"  {len(ok)}/{len(res)} variantes APRUEBAN con mejor checkpoint.")
        print(f"  Mejor: {b['variante']} -> BER test {b['mejor_test']:.4f} "
              f"(epoca {b['mejor_ep']} de {a.epochs})")
        estables = [r for r in ok if r["final_test"] - r["mejor_test"] < 0.01]
        if estables:
            print(f"  Estables (no se destruyen): "
                  f"{', '.join(r['variante'] for r in estables)}")
            print("  -> Usar una de estas en el barrido. Añadir validacion +")
            print("     mejor checkpoint a ae2.py y relanzar.")
        else:
            print("  Ninguna es estable: todas se destruyen tras alcanzar el optimo.")
            print("  -> El mejor checkpoint es OBLIGATORIO en el barrido, no opcional.")
    else:
        print("  Ninguna variante aprueba ni con el mejor checkpoint.")
        print("  -> Probar wd mayor (1e-2) para frenar el crecimiento de escala,")
        print("     o un encoder convolucional 1D que explote la repeticion.")


if __name__ == "__main__":
    main()
