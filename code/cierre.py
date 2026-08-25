#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIERRE DEL PROYECTO -- tres experimentos.

  --stage paridad   Convierte en demostracion la hipotesis de por que falla 'code'.
                    Separa DOS dificultades que hasta ahora estaban mezcladas:
                    (a) aprender XOR de grado d, y (b) encontrar el subconjunto
                    correcto entre n_in bits. Variando n_in se aisla cual manda.

  --stage conv      Encoder convolucional 1D frente al MLP. Prediccion:
                    debe ganar en 'oversamp' (periodo 4) y 'markov' (correlacion
                    local), y NO cambiar nada en 'lowdim' ni 'code' (W y P son
                    matrices aleatorias: no hay localidad que explotar). Esas dos
                    ultimas son el CONTROL del experimento.

  --stage datos     Escalado de datos en lowdim L=70. Con PASOS FIJOS, no epocas
                    fijas: si se fijan epocas, mas datos = mas pasos y no se
                    distingue el efecto de los datos del de la optimizacion.

Config validada del proyecto: BatchNorm1d(affine=False) + lr=1e-3.

Uso:
    python3 cierre.py --stage paridad --device cuda
    python3 cierre.py --stage conv    --device cuda
    python3 cierre.py --stage datos   --device cuda
    python3 cierre.py --stage all     --device cuda --outdir ./cierre
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ae2 as A


# =========================================================================== #
# ETAPA 1: DIAGNOSTICO DE PARIDAD
# =========================================================================== #
def stage_paridad(cfg, out):
    """
    Entrena un MLP para predecir el XOR (producto de signos) de un subconjunto
    de d bits, entre n_in bits de entrada.

    La clave del diseno: variar n_in separa dos cosas.
      - Si falla en d=3 incluso con n_in=10, el problema es la PARIDAD.
      - Si aprende con n_in=10 pero falla con n_in=500, el problema es la
        BUSQUEDA del subconjunto.
    El AE sobre 'code' enfrenta las dos a la vez, con n_in=500 y d=3.
    """
    dev = cfg["device"]
    print(f"\n{'='*74}\nETAPA 1 - PARIDAD: que impide aprender XOR de grado d?\n{'='*74}")
    print(f"{'n_in':>6s} {'d':>3s} {'acc_train':>10s} {'acc_test':>9s}  veredicto")
    filas = []
    for n_in in cfg["par_nin"]:
        for d in cfg["par_grados"]:
            rng = np.random.default_rng(cfg["seed"])
            sub = rng.choice(n_in, size=d, replace=False)
            def gen(n):
                B = 2.0 * rng.integers(0, 2, (n, n_in)).astype(np.float32) - 1.0
                y = (B[:, sub].prod(1) > 0).astype(np.float32)
                return torch.from_numpy(B), torch.from_numpy(y)
            Xtr, ytr = gen(cfg["par_train"]); Xte, yte = gen(20000)
            Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)

            torch.manual_seed(cfg["seed"])
            net = nn.Sequential(nn.Linear(n_in, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512, 1)).to(dev)
            n, bs = Xtr.shape[0], 1024
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=1e-3, total_steps=cfg["par_epochs"] * max(1, n // bs),
                pct_start=0.1)
            for ep in range(cfg["par_epochs"]):
                perm = torch.randperm(n, device=dev)
                for i in range(0, n - bs + 1, bs):
                    idx = perm[i:i + bs]
                    loss = F.binary_cross_entropy_with_logits(
                        net(Xtr[idx]).squeeze(1), ytr[idx])
                    opt.zero_grad(set_to_none=True); loss.backward()
                    opt.step(); sch.step()
            with torch.no_grad():
                atr = ((net(Xtr).squeeze(1) > 0).float() == ytr).float().mean().item()
                ate = ((net(Xte).squeeze(1) > 0).float() == yte).float().mean().item()
            v = "APRENDE" if ate > 0.95 else ("parcial" if ate > 0.6 else "FALLA (azar)")
            memo = " [memoriza]" if atr - ate > 0.15 else ""
            print(f"{n_in:6d} {d:3d} {atr:10.4f} {ate:9.4f}  {v}{memo}")
            filas.append(dict(n_in=n_in, grado=d, acc_train=atr, acc_test=ate))
            del Xtr, ytr, Xte, yte
            if dev == "cuda": torch.cuda.empty_cache()

    json.dump(filas, open(os.path.join(out, "paridad.json"), "w"), indent=1)

    print("\n  INTERPRETACION:")
    for d in cfg["par_grados"]:
        acc = {r["n_in"]: r["acc_test"] for r in filas if r["grado"] == d}
        chico = acc.get(min(cfg["par_nin"]), 0); grande = acc.get(max(cfg["par_nin"]), 0)
        if d == 1: continue
        if chico > 0.95 and grande < 0.6:
            print(f"    d={d}: aprende con n_in chico, falla con n_in grande "
                  f"-> el cuello es la BUSQUEDA del subconjunto")
        elif chico < 0.6:
            print(f"    d={d}: falla incluso con n_in chico -> el cuello es la PARIDAD misma")
        elif grande > 0.95:
            print(f"    d={d}: aprende siempre -> este grado NO es una barrera")
    return filas


# =========================================================================== #
# ETAPA 2: ENCODER CONVOLUCIONAL
# =========================================================================== #
class ConvAE(nn.Module):
    """500 -> 250 -> 125 -> 63 (canales 64/128/256) -> Linear -> latente binario."""
    def __init__(self, dim, latent, ch=(64, 128, 256)):
        super().__init__()
        c1, c2, c3 = ch
        self.enc = nn.Sequential(
            nn.Conv1d(1, c1, 7, 2, 3), nn.BatchNorm1d(c1), nn.GELU(),
            nn.Conv1d(c1, c2, 5, 2, 2), nn.BatchNorm1d(c2), nn.GELU(),
            nn.Conv1d(c2, c3, 5, 2, 2), nn.BatchNorm1d(c3), nn.GELU())
        self.enc_fc = nn.Linear(63 * c3, latent)
        self.enc_norm = nn.BatchNorm1d(latent, affine=False)   # config validada
        self.dec_fc = nn.Linear(latent, 63 * c3)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(c3, c2, 5, 2, 2, output_padding=0),
            nn.BatchNorm1d(c2), nn.GELU(),
            nn.ConvTranspose1d(c2, c1, 5, 2, 2, output_padding=1),
            nn.BatchNorm1d(c1), nn.GELU(),
            nn.ConvTranspose1d(c1, 1, 7, 2, 3, output_padding=1))
        self.c3 = c3

    def encode(self, x):
        h = self.enc(x.unsqueeze(1)).flatten(1)
        return A.BinarizeSTE.apply(self.enc_norm(self.enc_fc(h)))

    def decode(self, z):
        h = self.dec_fc(z).view(-1, self.c3, 63)
        return self.dec(h).squeeze(1)

    def forward(self, x):
        return self.decode(self.encode(x))


def entrena(model, Xtr, Xte, cfg, pasos=None):
    dev = cfg["device"]
    model = model.to(dev)
    n, bs = Xtr.shape[0], cfg["batch"]
    spe = max(1, n // bs)
    total = pasos if pasos else cfg["epochs"] * spe
    epochs = max(1, int(np.ceil(total / spe)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                              total_steps=total, pct_start=0.1)
    hechos = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            if hechos >= total: break
            xb = Xtr[perm[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = F.binary_cross_entropy_with_logits(model(xb), (xb + 1) * 0.5)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); hechos += 1
    model.eval()
    with torch.no_grad():
        return A.ber_hard(model(Xte).float(), Xte), hechos


def stage_conv(cfg, out):
    dev = cfg["device"]
    print(f"\n{'='*74}\nETAPA 2 - ENCODER CONVOLUCIONAL vs MLP\n{'='*74}")
    print("  Prediccion: gana en oversamp (periodo 4) y markov (correlacion local).")
    print("  CONTROL: no debe cambiar nada en lowdim ni code (W y P son aleatorias,")
    print("  no hay localidad que explotar). Si cambia ahi, algo esta mal.\n")
    print(f"{'fuente':9s} {'L':>4s} {'MLP':>9s} {'Conv':>9s} {'vara':>9s} {'mejora':>9s}  gana?")
    filas = []
    for fuente in cfg["conv_fuentes"]:
        Xtr, Xte, meta, _ = A.build_dataset(fuente, cfg["n_train"], cfg["n_test"],
                                            500, cfg, cfg["seed"])
        Xtr, Xte = Xtr.to(dev), Xte.to(dev)
        for L in cfg["conv_latentes"]:
            torch.manual_seed(cfg["seed"])
            m1 = A.PlainAE(500, L, 500, cfg["hidden"], cfg["depth"])
            b1, _ = entrena(m1, Xtr, Xte, cfg)
            del m1
            if dev == "cuda": torch.cuda.empty_cache()
            torch.manual_seed(cfg["seed"])
            m2 = ConvAE(500, L)
            b2, _ = entrena(m2, Xtr, Xte, cfg)
            del m2
            if dev == "cuda": torch.cuda.empty_cache()
            vara = cfg["varas"].get((fuente, L), float("nan"))
            mej = (b1 - b2) / max(b1, 1e-12) * 100
            g = "SI" if b2 < vara else "no"
            print(f"{fuente:9s} {L:4d} {b1:9.4f} {b2:9.4f} {vara:9.4f} {mej:8.1f}%  {g}")
            filas.append(dict(fuente=fuente, L=L, ber_mlp=b1, ber_conv=b2,
                              vara=vara, mejora_pct=mej))
        del Xtr, Xte
        if dev == "cuda": torch.cuda.empty_cache()
    json.dump(filas, open(os.path.join(out, "conv.json"), "w"), indent=1)

    print("\n  CONTROL:")
    for f in ("lowdim", "code"):
        r = [x for x in filas if x["fuente"] == f]
        if r:
            m = np.mean([x["mejora_pct"] for x in r])
            print(f"    {f:8s}: mejora media {m:+.1f}%  "
                  f"{'(esperado ~0: control OK)' if abs(m) < 10 else '(INESPERADO: revisar)'}")
    return filas


# =========================================================================== #
# ETAPA 3: ESCALADO DE DATOS
# =========================================================================== #
def stage_datos(cfg, out):
    """
    PASOS FIJOS, datos variables. Si se fijaran epocas, mas datos daria mas
    pasos y no se podria separar el efecto de los datos del de la optimizacion.
    """
    dev = cfg["device"]
    print(f"\n{'='*74}\nETAPA 3 - ESCALADO DE DATOS (lowdim L=70), pasos FIJOS\n{'='*74}")
    print(f"  pasos fijos = {cfg['pasos_fijos']} en todas las corridas\n")
    print(f"{'n_train':>10s} {'BER':>9s} {'vs 200k':>9s} {'pasos':>8s} {'epocas eq':>10s}")
    filas = []
    base = None
    for n_tr in cfg["datos_n"]:
        Xtr, Xte, meta, _ = A.build_dataset("lowdim", n_tr, cfg["n_test"],
                                            500, cfg, cfg["seed"])
        Xtr, Xte = Xtr.to(dev), Xte.to(dev)
        torch.manual_seed(cfg["seed"])
        m = A.PlainAE(500, 70, 500, cfg["hidden"], cfg["depth"])
        b, pasos = entrena(m, Xtr, Xte, cfg, pasos=cfg["pasos_fijos"])
        if base is None: base = b
        eq = pasos / max(1, n_tr // cfg["batch"])
        print(f"{n_tr:10d} {b:9.4f} {(b-base)/base*100:8.1f}% {pasos:8d} {eq:10.1f}")
        filas.append(dict(n_train=n_tr, ber=b, pasos=pasos, epocas_equiv=eq))
        del Xtr, Xte, m
        if dev == "cuda": torch.cuda.empty_cache()
    json.dump(filas, open(os.path.join(out, "datos.json"), "w"), indent=1)

    b = [f["ber"] for f in filas]
    if len(b) > 1:
        ult = (b[-2] - b[-1]) / max(b[-2], 1e-12) * 100
        print(f"\n  Mejora del ultimo salto: {ult:.1f}%")
        print("  -> " + ("NO ha saturado: mas datos siguen ayudando."
                         if ult > 2 else
                         "SATURADO: mas datos ya no aportan; el limite es otro."))
    return filas


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "paridad", "conv", "datos"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="./cierre")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = dict(
        dim=500, k=32, markov_p=0.05, oversample=4, device=dev, seed=a.seed,
        hidden=1536, depth=4, batch=4096, lr=1e-3, epochs=300,
        n_train=400000, n_test=40000,
        # etapa 1
        par_nin=[10, 100, 500], par_grados=[1, 2, 3, 4],
        par_train=400000, par_epochs=60,
        # etapa 2
        conv_fuentes=["oversamp", "markov", "lowdim", "code"],
        conv_latentes=[70, 125],
        varas={("oversamp", 70): 0.2330, ("oversamp", 125): 0.0000,
               ("markov", 70): 0.0914, ("markov", 125): 0.0488,
               ("lowdim", 70): 0.1861, ("lowdim", 125): 0.0613,
               ("code", 70): 0.3787, ("code", 125): 0.3339},
        # etapa 3
        datos_n=[200000, 400000, 800000, 1600000], pasos_fijos=30000,
    )
    print(f"[device] {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    t0 = time.time()
    if a.stage in ("all", "paridad"): stage_paridad(cfg, a.outdir)
    if a.stage in ("all", "conv"):    stage_conv(cfg, a.outdir)
    if a.stage in ("all", "datos"):   stage_datos(cfg, a.outdir)
    print(f"\nTiempo total: {time.time()-t0:.0f}s  (salidas en {a.outdir})")


if __name__ == "__main__":
    main()
