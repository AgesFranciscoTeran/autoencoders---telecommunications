#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GATE TEST -- el arnés debe aprobar esto antes de creerle cualquier resultado.

Caso: fuente 'oversamp' (125 simbolos x4 = 500 bits, entropia 125 bits) con
latente de 250 bits. El AE tiene EL DOBLE de bits de los necesarios; la
respuesta correcta es BER ~ 0. Si no lo logra, el problema es optimizacion,
no "los autoencoders no sirven".

Compara dos encoders:
  A) tanh + STE          (lo que hay hoy en ae2.py)
  B) LayerNorm + STE     (propuesta: evita la saturacion de la tanh)

Uso:  python3 gate_test.py --device cuda
"""
import argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ae2 as A          # reutiliza BinarizeSTE, mlp y el generador de fuentes


class Enc(nn.Module):
    def __init__(self, dim, latent, hidden, depth, modo):
        super().__init__()
        self.net = A.mlp([dim] + [hidden] * depth + [latent])
        self.modo = modo
        self.norm = nn.LayerNorm(latent)

    def forward(self, x):
        pre = self.net(x)
        u = torch.tanh(pre) if self.modo == "tanh" else self.norm(pre)
        return A.BinarizeSTE.apply(u), pre


class Modelo(nn.Module):
    def __init__(self, dim, latent, hidden, depth, modo):
        super().__init__()
        self.e = Enc(dim, latent, hidden, depth, modo)
        self.d = A.mlp([latent] + [hidden] * depth + [dim])

    def forward(self, x):
        z, pre = self.e(x)
        return self.d(z), z, pre


def corre(modo, Xtr, Xte, latent, cfg):
    dev = cfg["device"]
    m = Modelo(Xtr.shape[1], latent, cfg["hidden"], cfg["depth"], modo).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    n, bs = Xtr.shape[0], cfg["batch"]
    steps = cfg["epochs"] * max(1, n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                              total_steps=steps, pct_start=0.1)
    amp = (dev == "cuda")
    print(f"\n  --- encoder = {modo} ---")
    for ep in range(cfg["epochs"]):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            xb = Xtr[perm[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg, _, _ = m(xb)
                loss = F.binary_cross_entropy_with_logits(lg, (xb + 1) * 0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
            opt.step(); sch.step()
        if (ep + 1) % max(1, cfg["epochs"] // 6) == 0 or ep == 0:
            m.eval()
            with torch.no_grad():
                lg, z, pre = m(Xte)
                lg = lg.float()
                ber = A.ber_hard(lg, Xte)
                # diagnosticos de colapso:
                sat = (pre.abs() > 3).float().mean().item()   # preactivaciones saturadas
                viv = (z.std(0) > 1e-3).float().mean().item() # bits latentes que varian
            print(f"    ep {ep+1:4d} | BER={ber:.4f} | |pre|>3: {sat*100:5.1f}% "
                  f"| bits vivos: {viv*100:5.1f}% | |grad|={gnorm:.3f}")
    m.eval()
    with torch.no_grad():
        return A.ber_hard(m(Xte)[0].float(), Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--latent", type=int, default=250)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               hidden=1536, depth=4, batch=4096, lr=a.lr, epochs=a.epochs)
    Xtr, Xte, meta, _ = A.build_dataset("oversamp", 200000, 20000, 500, cfg, 0)
    Xtr, Xte = Xtr.to(dev), Xte.to(dev)

    print("=" * 70)
    print(f"GATE TEST  |  oversamp  H={meta['entropy_bits']:.0f} bits  "
          f"latente={a.latent} bits  ->  BER correcto ~ 0")
    print(f"device={dev}  lr={a.lr}  epochs={a.epochs}")
    print("=" * 70)

    out = {}
    for modo in ["tanh", "layernorm"]:
        torch.manual_seed(0)
        t0 = time.time()
        out[modo] = corre(modo, Xtr, Xte, a.latent, cfg)
        print(f"    -> BER final = {out[modo]:.4f}   ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 70)
    for k, v in out.items():
        print(f"  {k:10s}: BER={v:.4f}  {'APRUEBA' if v < 0.005 else 'REPRUEBA'}")
    if out["layernorm"] < 0.005 <= out["tanh"]:
        print("\n  -> Diagnostico CONFIRMADO: la saturacion de tanh mataba el gradiente.")
        print("     Aplica el cambio a ae2.py y relanza la corrida completa.")
    elif out["tanh"] < 0.005:
        print("\n  -> La tanh NO es el problema: aprueba el caso facil.")
        print("     Entonces el fallo de la corrida esta en otro lado (lr, epochs,")
        print("     o el modo 'direct' especificamente). Revisar por separado.")
    else:
        print("\n  -> NINGUNO aprueba. El problema es mas profundo que el encoder:")
        print("     baja lr a 1e-3, sube epochs, y revisa el decoder.")


if __name__ == "__main__":
    main()
