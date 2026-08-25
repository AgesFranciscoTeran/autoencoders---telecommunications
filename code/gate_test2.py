#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GATE TEST 2 -- ablacion en vez de conjetura.

El gate test 1 refuto la hipotesis de saturacion: |pre|>3 nunca paso del 5%
y no hubo bits muertos. Este script mide lo correcto y compara cuatro variantes
del binarizador a la vez, con entrenamiento suficientemente largo para
distinguir "subentrenado" de "roto".

Variantes del encoder (todas terminan en BinarizeSTE):
  tanh       BinarizeSTE(tanh(pre))        <- lo que hay hoy
  layernorm  BinarizeSTE(LayerNorm(pre))   <- normaliza POR MUESTRA (reprobo)
  batchnorm  BinarizeSTE(BatchNorm(pre))   <- normaliza POR DIMENSION
  raw        BinarizeSTE(pre)              <- STE puro, sin nada

Metricas corregidas:
  |pre|>1   fraccion donde el STE RECORTA el gradiente a cero (el umbral real)
  |pre|>2   fraccion donde tanh ya mata el 93% del gradiente
  g_efec    gradiente efectivo medio que sobrevive: E[tanh'(pre)] o E[1_{|pre|<=1}]
  pendiente cambio de BER en el ultimo tramo: si sigue bajando, esta subentrenado

Uso:
    python3 gate_test2.py --device cuda --epochs 250
    python3 gate_test2.py --device cuda --epochs 250 --lr 1e-3
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ae2 as A          # reutiliza build_dataset, mlp, BinarizeSTE, ber_hard

VARIANTES = ["tanh", "layernorm", "batchnorm", "raw"]


class Modelo(nn.Module):
    def __init__(self, dim, latent, hidden, depth, variante):
        super().__init__()
        self.variante = variante
        self.enc = A.mlp([dim] + [hidden] * depth + [latent])
        self.dec = A.mlp([latent] + [hidden] * depth + [dim])
        self.ln = nn.LayerNorm(latent)
        self.bn = nn.BatchNorm1d(latent)

    def binariza(self, pre):
        if self.variante == "tanh":
            u = torch.tanh(pre)
        elif self.variante == "layernorm":
            u = self.ln(pre)
        elif self.variante == "batchnorm":
            u = self.bn(pre)
        else:
            u = pre
        return A.BinarizeSTE.apply(u), u

    def forward(self, x):
        pre = self.enc(x)
        z, u = self.binariza(pre)
        return self.dec(z), pre, u, z


@torch.no_grad()
def diagnostico(m, Xte):
    m.eval()
    lg, pre, u, z = m(Xte)
    ber = A.ber_hard(lg.float(), Xte)
    a = pre.abs().float()
    # gradiente efectivo: lo que realmente sobrevive al binarizador
    if m.variante == "tanh":
        g_efec = (1 - torch.tanh(pre.float()) ** 2).mean().item()
    else:
        g_efec = (u.abs().float() <= 1.0).float().mean().item()
    return dict(ber=ber,
                pre_gt1=(a > 1).float().mean().item(),
                pre_gt2=(a > 2).float().mean().item(),
                vivos=(z.float().std(0) > 1e-3).float().mean().item())


def corre(variante, Xtr, Xte, cfg):
    dev = cfg["device"]
    torch.manual_seed(0)
    m = Modelo(Xtr.shape[1], cfg["latent"], cfg["hidden"], cfg["depth"], variante).to(dev)
    n, bs = Xtr.shape[0], cfg["batch"]
    spe = max(1, n // bs)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                              total_steps=cfg["epochs"] * spe,
                                              pct_start=0.1)
    amp = (dev == "cuda")
    hist, marcas = [], sorted(set(
        [1] + [int(cfg["epochs"] * f) for f in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]))
    print(f"\n  --- {variante} ---")
    print(f"    {'ep':>5s} {'BER':>8s} {'|pre|>1':>8s} {'|pre|>2':>8s} {'vivos':>7s}")
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
        if ep in marcas:
            d = diagnostico(m, Xte); d["ep"] = ep; hist.append(d)
            print(f"    {ep:5d} {d['ber']:8.4f} {d['pre_gt1']*100:7.1f}% "
                  f"{d['pre_gt2']*100:7.1f}% {d['vivos']*100:6.1f}%")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--latent", type=int, default=250)
    ap.add_argument("--variantes", nargs="*", default=VARIANTES)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               hidden=1536, depth=4, batch=4096,
               lr=a.lr, epochs=a.epochs, latent=a.latent)
    Xtr, Xte, meta, _ = A.build_dataset("oversamp", 200000, 20000, 500, cfg, 0)
    Xtr, Xte = Xtr.to(dev), Xte.to(dev)

    print("=" * 72)
    print(f"ABLACION  |  oversamp  H={meta['entropy_bits']:.0f} bits  "
          f"latente={a.latent} bits  ->  BER correcto ~ 0")
    print(f"epochs={a.epochs}  lr={a.lr}  device={dev}")
    print("=" * 72)

    res = {}
    for v in a.variantes:
        t0 = time.time()
        res[v] = corre(v, Xtr, Xte, cfg)
        print(f"    -> final {res[v][-1]['ber']:.4f}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 72)
    print(f"{'variante':>10s} {'BER final':>10s} {'pendiente':>11s} {'veredicto':>26s}")
    for v, h in res.items():
        fin = h[-1]["ber"]
        pend = h[-1]["ber"] - h[-2]["ber"] if len(h) > 1 else 0.0
        if fin < 0.005:
            ver = "APRUEBA"
        elif pend < -0.002:
            ver = "SUBENTRENADO (sigue bajando)"
        else:
            ver = "ESTANCADO"
        print(f"{v:>10s} {fin:10.4f} {pend:+11.4f} {ver:>26s}")

    mejor = min(res, key=lambda v: res[v][-1]["ber"])
    fin, h = res[mejor][-1]["ber"], res[mejor]
    pend = h[-1]["ber"] - h[-2]["ber"] if len(h) > 1 else 0.0
    print(f"\n  Mejor variante: {mejor} (BER {fin:.4f})")
    if fin < 0.005:
        print("  -> El arnes APRUEBA la calibracion. Relanzar el barrido con esta variante.")
    elif pend < -0.002:
        print("  -> Sigue bajando: el problema es TIEMPO DE ENTRENAMIENTO, no la")
        print("     arquitectura. Subir epochs (probar el doble) antes de tocar nada mas.")
    else:
        print("  -> Estancado sin llegar a ~0. El binarizador no es el cuello de botella:")
        print("     revisar tasa de aprendizaje, y probar un encoder convolucional que")
        print("     explote la estructura de repeticion de la fuente.")


if __name__ == "__main__":
    main()
