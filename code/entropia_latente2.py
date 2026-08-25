#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTROPIA DEL LATENTE v2 -- estimador correcto.

CORRIGE un error de la v1: la aproximacion H_conjunta ~ H_marginal - sum I(i;j)
solo vale para dependencias en arbol. Con 250 bits (31125 pares) sobrecuenta la
informacion compartida y produce resultados imposibles (0 bits para un latente
que reconstruye una fuente de 125 bits).

Metodo correcto (el de la compresion aprendida, Balle et al. 2017): entrenar un
modelo AUTOREGRESIVO sobre los bits del latente. Su entropia cruzada de test es
la longitud de codigo realmente alcanzable, y es una cota SUPERIOR valida de
H(z) que se aprieta segun mejore el modelo.

Se reportan:
  H_marginal   suma de entropias por bit (cota superior burda)
  H_lineal     modelo autoregresivo LINEAL causal (captura dependencias lineales)
  H_gru        modelo autoregresivo GRU (captura no lineales)
  cota inferior de Fano:  H(z) >= H(x) - n*Hb(BER)

La cota inferior es la verificacion de cordura: cualquier estimacion por debajo
de ella es imposible y el script lo marca.

Uso:
    python3 entropia_latente2.py --device cuda --fuente oversamp --latent 250
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ae2 as A


def Hb(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class ARLineal(nn.Module):
    """Predice el bit i a partir de los bits < i con una matriz triangular."""
    def __init__(self, L):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(L, L))
        self.b = nn.Parameter(torch.zeros(L))
        self.register_buffer("mask", torch.tril(torch.ones(L, L), diagonal=-1))

    def forward(self, B):                      # B en {0,1}, (n, L)
        x = 2.0 * B - 1.0
        return x @ (self.W * self.mask).t() + self.b


class ARGru(nn.Module):
    """GRU causal: el estado tras el bit i-1 predice el bit i."""
    def __init__(self, L, h=256):
        super().__init__()
        self.emb = nn.Linear(1, 32)
        self.gru = nn.GRU(32, h, batch_first=True)
        self.out = nn.Linear(h, 1)
        self.h0 = nn.Parameter(torch.zeros(1, 1, h))

    def forward(self, B):
        n, L = B.shape
        x = (2.0 * B - 1.0).unsqueeze(-1)
        e = self.emb(x[:, :-1])                       # entradas: bits 0..L-2
        h0 = self.h0.expand(-1, n, -1).contiguous()
        y, _ = self.gru(e, h0)
        lg = self.out(y).squeeze(-1)                  # predice bits 1..L-1
        p0 = torch.zeros(n, 1, device=B.device)       # bit 0: sin contexto
        return torch.cat([p0, lg], 1)


def entropia_ar(modelo, Btr, Bte, dev, epochs=60, lr=3e-3, bs=1024):
    modelo = modelo.to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-5)
    n = Btr.shape[0]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        modelo.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            b = Btr[perm[i:i + bs]]
            loss = F.binary_cross_entropy_with_logits(modelo(b), b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    modelo.eval()
    with torch.no_grad():
        # entropia cruzada en BITS por vector = longitud de codigo alcanzable
        ce = F.binary_cross_entropy_with_logits(modelo(Bte), Bte, reduction="sum")
        return float(ce.item() / Bte.shape[0] / np.log(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fuente", default="oversamp")
    ap.add_argument("--latent", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--ar-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               hidden=1536, depth=4, batch=4096, lr=1e-3, epochs=a.epochs)
    Xtr, Xte, meta, _ = A.build_dataset(a.fuente, 200000, 40000, 500, cfg, a.seed)
    Xtr, Xte = Xtr.to(dev), Xte.to(dev)

    print("=" * 74)
    print(f"ENTROPIA DEL LATENTE v2 | {a.fuente} (H fuente = {meta['entropy_bits']:.1f} bits)")
    print(f"latente = {a.latent} bits | epochs AE = {a.epochs}")
    print("=" * 74)

    # --- entrenar el AE ---
    torch.manual_seed(a.seed)
    m = A.PlainAE(500, a.latent, 500, cfg["hidden"], cfg["depth"]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    n, bs = Xtr.shape[0], cfg["batch"]
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=a.epochs * max(1, n // bs), pct_start=0.1)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            xb = Xtr[perm[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = F.binary_cross_entropy_with_logits(m(xb), (xb + 1) * 0.5)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        ber = A.ber_hard(m(Xte).float(), Xte)
        Z = m.encode(Xte).float()
    print(f"\n  BER de test: {ber:.6f}   (AE en {time.time()-t0:.0f}s)")

    B = (Z > 0).float()
    Btr, Bte = B[:20000], B[20000:]
    p = B.mean(0).cpu().numpy()
    Hm = float(Hb(p).sum())

    # --- cota inferior de Fano (verificacion de cordura) ---
    Hx = meta["entropy_bits"]
    piso = max(0.0, Hx - 500 * float(Hb(max(ber, 1e-12))))

    print(f"  bits muertos: {int(((p < 0.01) | (p > 0.99)).sum())} de {a.latent}")
    print(f"\n  estimando modelos autoregresivos ({a.ar_epochs} epocas)...", flush=True)
    torch.manual_seed(0)
    Hl = entropia_ar(ARLineal(a.latent), Btr, Bte, dev, a.ar_epochs)
    torch.manual_seed(0)
    Hg = entropia_ar(ARGru(a.latent), Btr, Bte, dev, a.ar_epochs)
    Hbest = min(Hl, Hg)

    print(f"\n  {'medida':<34s} {'bits':>8s} {'tasa':>8s}")
    print(f"  {'-'*52}")
    print(f"  {'nominal (L)':<34s} {a.latent:8d} {a.latent/500:8.3f}")
    print(f"  {'H marginal (cota sup. burda)':<34s} {Hm:8.1f} {Hm/500:8.3f}")
    print(f"  {'H con modelo AR lineal':<34s} {Hl:8.1f} {Hl/500:8.3f}")
    print(f"  {'H con modelo AR GRU':<34s} {Hg:8.1f} {Hg/500:8.3f}")
    print(f"  {'cota INFERIOR de Fano':<34s} {piso:8.1f} {piso/500:8.3f}")
    print(f"  {'entropia de la fuente':<34s} {Hx:8.1f} {Hx/500:8.3f}")

    print()
    if Hbest < piso - 0.5:
        print(f"  [!!] IMPOSIBLE: la estimacion ({Hbest:.1f}) esta por debajo de la")
        print(f"       cota de Fano ({piso:.1f}). El modelo AR esta sobreajustando")
        print("       o hay un error. NO usar este numero.")
    else:
        ahorro = (a.latent - Hbest) / a.latent * 100
        print(f"  Mejor cota superior valida: {Hbest:.1f} bits (tasa {Hbest/500:.3f})")
        print(f"  Ahorro por codificacion entropica: {ahorro:.1f}% ({a.latent-Hbest:.0f} bits)")
        if ahorro > 10:
            print("  -> La tasa nominal SOBREESTIMA el coste. Vale la pena regraficar")
            print("     el mapa de viabilidad con la tasa efectiva.")
        else:
            print("  -> El latente ya es casi incompresible. La tasa nominal es honesta.")


if __name__ == "__main__":
    main()
