#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
denoising.py — ¿Corromper la entrada durante el entrenamiento (Vincent et al.,
2008) abre el camino de gradiente que a `code` le falta?

MOTIVACION
----------
El diagnostico de paridad mostro que un MLP aprende XOR de grado 3 con
supervision directa, pero que la perdida de reconstruccion no genera camino de
gradiente hacia 250 paridades simultaneas: ninguna paridad individual reduce el
error hasta que estan casi todas bien. El resultado es que sobre `code` a
L=250, donde la compresion sin perdida es posible, el autoencoder se queda en
BER ~0.198 (copiar la mitad y adivinar da 0.250; el oraculo da 0).

El ruido de enmascaramiento cambia el objetivo: si se tapan al azar una
fraccion p de los 500 bits de entrada y se exige reconstruir los 500, la unica
forma de acertar un bit tapado es usar los demas. En `code`, cada bit de
paridad depende de tres sistematicos, asi que predecir un bit oculto es
exactamente aprender la estructura del codigo, y acertar UN bit oculto ya
reduce la perdida desde el primer paso. Es la hipotesis del "objetivo
auxiliar" de 07-hacia-paper.md, con nombre concreto.

Densidad de senal por p (una paridad es inferible si sus 3 sistematicos quedan
visibles, con probabilidad (1-p)^3):

    p=0.10 -> 182 de 250 paridades inferibles,  50 posiciones tapadas
    p=0.25 -> 105 de 250 paridades inferibles, 125 posiciones tapadas
    p=0.50 ->  31 de 250 paridades inferibles, 250 posiciones tapadas

PROTOCOLO
---------
- Corrupcion SOLO en entrenamiento. Validacion y test siempre sobre entrada
  limpia, como en Vincent: el ruido es un regularizador, no parte del problema.
- Objetivo: reconstruir la entrada LIMPIA a partir de la corrompida.
- Dos tipos de corrupcion: `mask` (poner a 0 una fraccion p; para datos ±1 el 0
  es "desconocido") y `flip` (cambiar de signo una fraccion p; es un canal BSC
  en la entrada). El principal es `mask`.
- p = 0 es el control y tiene que reproducir `ladder_random` de rbm_stack.py.
- `random` es el control negativo: sobre bits i.i.d. un bit tapado es
  impredecible, asi que el enmascarado NO puede ayudar. Si aparenta ayudar,
  hay una fuga en el montaje.
- Arquitectura y ajuste fino identicos a rbm_stack.py: escalera estrecha,
  BatchNorm(affine=False) + STE en el cuello, AdamW + OneCycle, 30 000 pasos,
  mejor checkpoint sobre validacion, BER reportado de test. 4 semillas.

TRES CORRECCIONES SOBRE LA PRIMERA VERSION
-------------------------------------------
(a) RECALIBRACION DE BATCHNORM. El BN del cuello acumula sus estadisticas en el
    regimen corrompido y en evaluacion recibe entrada limpia. No es un simple
    reescalado: con mask, E[W·x] se encoge por (1-p) pero el sesgo b no, asi que
    el centrado queda descolocado por un desplazamiento que NO escala con los
    datos. sign() es invariante a escala pero no a ese desplazamiento, de modo
    que algunos bits del latente pueden cambiar de signo. Se reporta el BER
    antes y despues de refrescar las running stats con entrada limpia
    (`ber_test` y `ber_test_bnfix`): si difieren, el desajuste era real.
    Reemplaza a `--anneal`, que trataba el sintoma.
(b) COMPARACION PAREADA. Dos generadores separados: `gen_datos` con la misma
    semilla para toda p (mismo orden de minilotes) y `gen_ruido` distinto por p.
    Antes compartian generador y con p=0 no se consumia, de modo que p=0 y
    p=0.25 veian ordenes de datos distintos: un confundido evitable.
(c) BER EN POSICIONES TAPADAS. El BER de test se mide sobre entrada limpia, asi
    que nunca observa si el modelo realmente infiere bits ocultos — que es el
    mecanismo que la hipotesis afirma. `ber_tapados` lo mide directamente.
    Sobre `random` debe dar 0.5 exacto; sobre `code`, un valor bajo significa
    que el modelo SI aprendio la estructura del codigo aunque el BER limpio no
    mejore, y entonces el cuello de botella es la compresion y no el objetivo.

PREDICCIONES REGISTRADAS (2026-09-11, antes de correr)
-------------------------------------------------------
P1. `code` L=250: mask con alguna p in {0.10, 0.25} baja el BER frente a p=0 en
    4/4 semillas, y lo baja por debajo de 0.15.
P2. `code` L=125: la mejora existe pero es menor; el cuello es el limite.
P3. `lowdim` y `markov`: mask con p=0.25 NO ayuda (dentro de 1 sd o peor),
    porque ahi la reconstruccion ya da gradiente y el ruido solo destruye
    informacion.
P4. p=0.50 empeora en todas las fuentes.
P5. Si P1 se cumple, el cociente BER/BER_top90 sobre `code` sube por encima
    de 1.2: los errores dejan de estar repartidos uniformemente.
P6. `random`: mask no ayuda a ninguna p, y `ber_tapados` se queda en 0.5.

USO
---
    python3 code/denoising.py                       # grilla completa, 4 semillas
    python3 code/denoising.py --quick               # humo: 1 semilla, pasos cortos
    python3 code/denoising.py --fuentes code --L 250 --p 0,0.1,0.25
    python3 code/denoising.py --modo flip

Salida en data/denoising/: un JSON por (fuente, semilla), denoising_resumen.csv
con todas las filas, resumen impreso con media ± sd por celda, y
procedencia.json con hash de git, versiones y configuracion.
"""
import argparse, copy, json, os, sys, time, subprocess, platform
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbm_stack as R          # LadderAE, build_ladder, bce, ber, VARAS, LADDER
A = R.A                        # arnes: build_dataset, ber_hard, ber_by_confidence

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "data", "denoising")

L_POR_FUENTE = {"code": [250, 125], "lowdim": [250, 70],
                "markov": [250, 70], "random": [250]}


def cargar_varas():
    """Vara por (fuente, L) desde data/baselines_clasicos.csv; cae a R.VARAS."""
    varas = dict(R.VARAS)
    ruta = os.path.join(RAIZ, "data", "baselines_clasicos.csv")
    if os.path.exists(ruta):
        import csv
        for row in csv.DictReader(open(ruta)):
            varas[(row["fuente"], int(row["latente_bits"]))] = float(row["ber_baseline"])
    return varas


VARAS = cargar_varas()
# Referencias de `code` a L=250, ademas de la vara (que ahi es el oraculo, 0):
REF_CODE_250 = {"trivial (copiar mitad, adivinar)": 0.2500,
                "AE previo, MLP ancho v4": 0.1982, "oraculo": 0.0}


# =========================================================================== #
# 1. CORRUPCION
# =========================================================================== #
def mascara(shape, p, device, gen):
    """Booleana: True donde se corrompe."""
    return torch.rand(shape, device=device, generator=gen) < p


def aplicar(x, tap, modo):
    """Aplica la corrupcion indicada por la mascara booleana `tap`."""
    if modo == "mask":
        return x * (~tap).to(x.dtype)                 # 0 donde se tapa
    if modo == "flip":
        return x * (1.0 - 2.0 * tap.to(x.dtype))      # -x donde se voltea
    raise ValueError(modo)


def corromper(x, p, modo, gen):
    """Devuelve una copia corrompida de x (valores ±1). p=0 -> x sin tocar."""
    if p <= 0.0:
        return x
    return aplicar(x, mascara(x.shape, p, x.device, gen), modo)


# =========================================================================== #
# 2. DIAGNOSTICOS
# =========================================================================== #
@torch.no_grad()
def recalibra_bn(m, Xtr, cfg, gen, n_batches=100):
    """
    Refresca las running stats de BatchNorm con entrada LIMPIA.

    Necesario porque el BN del cuello las acumulo bajo corrupcion: con mask,
    E[W·x] se encoge por (1-p) pero el sesgo b no, de modo que el centrado
    queda descolocado por un desplazamiento que no escala con los datos.
    """
    hubo = False
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); hubo = True
    if not hubo:
        return False
    m.train()
    n, bs = Xtr.shape[0], cfg["batch"]
    for _ in range(n_batches):
        idx = torch.randint(0, n, (bs,), device=Xtr.device, generator=gen)
        m(Xtr[idx])
    m.eval()
    return True


@torch.no_grad()
def ber_en_tapados(m, X, p, modo, gen):
    """
    BER medido SOLO en las posiciones corrompidas: mide si el modelo infiere
    bits ocultos a partir de los visibles, que es el mecanismo que la hipotesis
    afirma. Sobre `random` debe dar 0.5 exacto (un bit i.i.d. tapado es
    impredecible). Sobre `code`, un valor bajo significa que el modelo aprendio
    la estructura del codigo aunque el BER con entrada limpia no mejore.
    """
    if p <= 0.0:
        return float("nan")
    m.eval()
    tap = mascara(X.shape, p, X.device, gen)
    xin = aplicar(X, tap, modo)
    h = torch.sign(R.apply_chunks(lambda z: m(z).float(), xin))
    h[h == 0] = -1.0
    n_tap = tap.sum().item()
    return ((h != X) & tap).float().sum().item() / max(n_tap, 1)


# =========================================================================== #
# 3. AJUSTE FINO CON ENTRADA CORROMPIDA
# =========================================================================== #
def finetune_denoising(m, Xtr, Xva, Xte, cfg, p, modo, gen_datos, gen_ruido):
    """
    gen_datos: misma semilla para toda p -> el orden de minilotes es identico
               entre celdas y la comparacion queda pareada.
    gen_ruido: distinto por p -> la corrupcion si varia.
    """
    dev = cfg["device"]
    n, bs = Xtr.shape[0], cfg["batch"]
    steps = cfg["ft_steps"]
    amp = (dev == "cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                              total_steps=steps, pct_start=0.1)
    best_va, best_state, best_step = 1.0, copy.deepcopy(m.state_dict()), 0
    eval_every = max(1, steps // 40)
    for step in range(steps):
        m.train()
        idx = torch.randint(0, n, (bs,), device=dev, generator=gen_datos)
        xb = Xtr[idx]
        xin = corromper(xb, p, modo, gen_ruido)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = R.bce(m(xin), xb)                  # objetivo: la entrada LIMPIA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            va = R.ber(m, Xva)                        # validacion LIMPIA
            if va < best_va:
                best_va, best_step = va, step + 1
                best_state = copy.deepcopy(m.state_dict())
    final_te = R.ber(m, Xte)
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        logits = R.apply_chunks(lambda z: m(z).float(), Xte)
    conf = A.ber_by_confidence(logits, Xte, fracs=(0.5, 0.9, 1.0))

    # mecanismo: ¿infiere bits ocultos?  (con el modelo tal cual quedo)
    tap_ber = ber_en_tapados(m, Xte, p, modo, gen_ruido)

    # (a) ¿el BN estaba descalibrado por entrenar con entrada corrompida?
    ber_bnfix = float("nan")
    if p > 0.0:
        m_fix = copy.deepcopy(m)
        if recalibra_bn(m_fix, Xtr, cfg, gen_datos, cfg.get("bn_batches", 100)):
            ber_bnfix = R.ber(m_fix, Xte)
        del m_fix

    return dict(ber_test=conf["ber_top100"], ber_test_final=final_te,
                ber_test_bnfix=ber_bnfix, ber_val=best_va, mejor_step=best_step,
                ber_top90=conf["ber_top90"], ber_top50=conf["ber_top50"],
                ber_tapados=tap_ber)


# =========================================================================== #
# 4. CORRIDA POR (FUENTE, SEMILLA)
# =========================================================================== #
def corre_fuente_semilla(fuente, seed, cfg, a):
    dev = cfg["device"]
    Xtr, Xrest, meta, _ = A.build_dataset(fuente, cfg["n_train"], cfg["n_test"] * 2,
                                          500, cfg, seed)
    Xva, Xte = Xrest[:cfg["n_test"]], Xrest[cfg["n_test"]:]
    Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)
    print(f"\n[{fuente} seed={seed}]  H={meta['entropy_bits']:.1f} bits  "
          f"train={Xtr.shape[0]} val={Xva.shape[0]} test={Xte.shape[0]}  modo={a.modo}")
    print(f"    {'L':>4s} {'p':>5s} {'BER test':>9s} {'bn-fix':>8s} {'top90':>8s} "
          f"{'tapados':>8s} {'@step':>6s} {'vara':>7s}")
    filas = []
    Ls = a.L if a.L else L_POR_FUENTE[fuente]
    for L in Ls:
        depth = R.LADDER.index(L) + 1
        for p in a.p:
            torch.manual_seed(seed)                   # misma init para todas las p
            m = R.LadderAE(R.build_ladder(depth)).to(dev)
            # (b) comparacion pareada: datos iguales entre celdas, ruido distinto
            gen_datos = torch.Generator(device=dev); gen_datos.manual_seed(seed)
            gen_ruido = torch.Generator(device=dev)
            gen_ruido.manual_seed(seed * 7919 + int(round(p * 1000)))
            t0 = time.time()
            r = finetune_denoising(m, Xtr, Xva, Xte, cfg, p, a.modo,
                                   gen_datos, gen_ruido)
            vara = VARAS.get((fuente, L), float("nan"))
            print(f"    {L:4d} {p:5.2f} {r['ber_test']:9.4f} {r['ber_test_bnfix']:8.4f} "
                  f"{r['ber_top90']:8.4f} {r['ber_tapados']:8.4f} "
                  f"{r['mejor_step']:6d} {vara:7.4f}  ({time.time()-t0:.0f}s)", flush=True)
            filas.append(dict(fuente=fuente, seed=seed, L=L, rate=L / 500, p=p,
                              modo=a.modo, vara=vara, **r))
            del m
            if dev == "cuda": torch.cuda.empty_cache()
    json.dump(dict(fuente=fuente, seed=seed, modo=a.modo, filas=filas),
              open(os.path.join(SALIDA, f"denoising_{fuente}_s{seed}.json"), "w"),
              indent=1)
    del Xtr, Xva, Xte
    if dev == "cuda": torch.cuda.empty_cache()
    return filas


# =========================================================================== #
# 5. RESUMEN Y PROCEDENCIA
# =========================================================================== #
def resumen(filas, a):
    import csv
    campos = list(filas[0].keys())
    with open(os.path.join(SALIDA, "denoising_resumen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(filas)

    celdas = {}
    for r in filas:
        celdas.setdefault((r["fuente"], r["L"], r["p"]), []).append(r)

    def col(c, k): return np.array([r[k] for r in c], dtype=float)

    print("\n" + "=" * 86)
    print("RESUMEN  (media ± sd sobre semillas; p=0 es el control)")
    print("=" * 86)
    print(f"{'fuente':8s} {'L':>4s} {'p':>5s} {'BER test':>16s} {'bn-fix':>8s} "
          f"{'top90':>8s} {'tapados':>8s} {'vs p=0':>8s} {'gana':>5s} {'vara':>7s}")
    veredictos = []
    for (fuente, L, p), rs in sorted(celdas.items()):
        b, t = col(rs, "ber_test"), col(rs, "ber_top90")
        fix, tap = col(rs, "ber_test_bnfix"), col(rs, "ber_tapados")
        ctrl = celdas.get((fuente, L, 0.0), [])
        if ctrl and p > 0 and len(ctrl) == len(rs):
            b0 = col(ctrl, "ber_test")
            delta = f"{(b.mean()-b0.mean())/b0.mean()*100:+7.1f}%"
            ganatxt = f"{int((b < b0).sum())}/{len(b)}"
        else:
            delta, ganatxt = "", ""
        print(f"{fuente:8s} {L:4d} {p:5.2f} {b.mean():8.4f} ± {b.std():6.4f} "
              f"{np.nanmean(fix):8.4f} {t.mean():8.4f} {np.nanmean(tap):8.4f} "
              f"{delta:>8s} {ganatxt:>5s} {rs[0]['vara']:7.4f}")
        veredictos.append(dict(fuente=fuente, L=L, p=p, ber=float(b.mean()),
                               sd=float(b.std()), top90=float(t.mean()),
                               bnfix=float(np.nanmean(fix)),
                               tapados=float(np.nanmean(tap)), gana_vs_p0=ganatxt))

    # (a) ¿importo el desajuste de BatchNorm?
    d = [abs(r["ber_test"] - r["ber_test_bnfix"]) for r in filas
         if r["p"] > 0 and not np.isnan(r["ber_test_bnfix"])]
    if d:
        print(f"\nRECALIBRACION DE BATCHNORM: |Δ| mediana {np.median(d):.4f}, "
              f"máxima {max(d):.4f}")
        print("  " + ("el desajuste NO era relevante; el BER limpio es interpretable tal cual"
                      if max(d) < 0.003 else
                      "el desajuste SI mueve el resultado: usar la columna bn-fix al concluir"))

    if any(r["fuente"] == "code" and r["L"] == 250 for r in filas):
        print("\nreferencias code L=250: " +
              "  ".join(f"{k} {v:.4f}" for k, v in REF_CODE_250.items()))

    # ---- predicciones registradas ----
    print("\nPREDICCIONES REGISTRADAS")
    def celda(f, L, p): return celdas.get((f, L, p))

    # P1: basta con que ALGUNA p gane 4/4 y baje de 0.15 (antes se exigian ambas)
    c0 = celda("code", 250, 0.0)
    if c0:
        b0 = col(c0, "ber_test"); gano_alguna = False; mejor = (1.0, None)
        for p in [pp for pp in a.p if pp > 0]:
            c = celda("code", 250, p)
            if not c or len(c) != len(c0):
                continue
            b = col(c, "ber_test"); n4 = int((b < b0).sum())
            if b.mean() < mejor[0]: mejor = (b.mean(), p)
            marca = "gana 4/4" if n4 == len(b) else f"gana {n4}/{len(b)}"
            print(f"     code L=250 p={p:.2f}: {b.mean():.4f}  ({marca})")
            if n4 == len(b) and b.mean() < 0.15:
                gano_alguna = True
        print(f"  P1 (alguna p gana 4/4 y baja de 0.15): "
              f"{'CONFIRMADA' if gano_alguna else 'REFUTADA'}  "
              f"(control {b0.mean():.4f}, mejor {mejor[0]:.4f} en p={mejor[1]})")
        cbest = celda("code", 250, mejor[1]) if mejor[1] is not None else None
        if cbest:
            q = float(np.mean([r["ber_test"] / max(r["ber_top90"], 1e-6) for r in cbest]))
            print(f"  P5 (BER/top90 en code sube de 1.2): "
                  f"{'CONFIRMADA' if q > 1.2 else 'REFUTADA'}  (cociente {q:.2f})")

    for f in ("lowdim", "markov"):
        for L in L_POR_FUENTE.get(f, []):
            c0, cp = celda(f, L, 0.0), celda(f, L, 0.25)
            if c0 and cp:
                b0, b = col(c0, "ber_test"), col(cp, "ber_test")
                print(f"  P3 ({f} L={L}, mask p=0.25 no ayuda): "
                      f"{'CONFIRMADA' if b.mean() >= b0.mean() - b0.std() else 'REFUTADA'}"
                      f"  ({b0.mean():.4f} -> {b.mean():.4f})")

    for (fuente, L) in dict.fromkeys((r["fuente"], r["L"]) for r in filas):
        c0, c5 = celda(fuente, L, 0.0), celda(fuente, L, 0.50)
        if c0 and c5:
            b0, b5 = col(c0, "ber_test").mean(), col(c5, "ber_test").mean()
            print(f"  P4 ({fuente} L={L}, p=0.5 empeora): "
                  f"{'CONFIRMADA' if b5 > b0 else 'REFUTADA'}  ({b0:.4f} -> {b5:.4f})")

    # P6: control negativo
    c0 = celda("random", 250, 0.0)
    if c0:
        b0 = col(c0, "ber_test"); ayuda = []
        for p in [pp for pp in a.p if pp > 0]:
            c = celda("random", 250, p)
            if c and len(c) == len(c0) and (col(c, "ber_test") < b0).all():
                ayuda.append(p)
        taps = [r["ber_tapados"] for r in filas
                if r["fuente"] == "random" and r["p"] > 0 and not np.isnan(r["ber_tapados"])]
        tm = float(np.mean(taps)) if taps else float("nan")
        print(f"  P6 (random: mask no ayuda y tapados ~0.5): "
              f"{'CONFIRMADA' if not ayuda and abs(tm - 0.5) < 0.02 else 'REFUTADA'}"
              f"  (ayuda en p={ayuda or 'ninguna'}, tapados {tm:.4f})")
        if ayuda:
            print("     [!] el enmascarado NO puede ayudar sobre bits i.i.d.:")
            print("         revisar el montaje antes de creer cualquier otro resultado")
    return veredictos


def procedencia(cfg, a, veredictos):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=RAIZ,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    info = dict(script="code/denoising.py",
                fecha_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                git_commit=git("rev-parse", "HEAD"), git_sucio=bool(git("status", "--porcelain")),
                torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                cuda=torch.version.cuda if torch.cuda.is_available() else None,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                config={k: v for k, v in cfg.items() if k != "device"},
                args=vars(a), veredictos=veredictos)
    json.dump(info, open(os.path.join(SALIDA, "procedencia.json"), "w"), indent=1)


# =========================================================================== #
# 6. MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fuentes", default="code,lowdim,markov,random")
    ap.add_argument("--L", default="", help="lista de L; vacio = la de L_POR_FUENTE")
    ap.add_argument("--p", default="0,0.1,0.25,0.5")
    ap.add_argument("--modo", default="mask", choices=["mask", "flip"])
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--n_train", type=int, default=400_000)
    ap.add_argument("--ft_steps", type=int, default=30_000)
    ap.add_argument("--bn_batches", type=int, default=100,
                    help="minilotes limpios para recalibrar BatchNorm")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.fuentes = a.fuentes.split(",")
    a.L = [int(x) for x in a.L.split(",")] if a.L else []
    a.p = [float(x) for x in a.p.split(",")]
    a.seeds = [int(x) for x in a.seeds.split(",")]
    if 0.0 not in a.p:
        sys.exit("--p debe incluir 0: es el control contra el que se compara todo")
    for f in a.fuentes:
        if f not in L_POR_FUENTE and not a.L:
            sys.exit(f"fuente sin L por defecto: {f}; pasa --L")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = dict(dim=500, k=32, markov_p=0.05, oversample=4, device=dev,
               n_train=a.n_train, n_test=20000,
               batch=4096, lr=1e-3, wd=1e-4, ft_steps=a.ft_steps,
               bn_batches=a.bn_batches)
    if a.quick:
        cfg.update(n_train=40000, n_test=5000, ft_steps=600, bn_batches=10)
        a.seeds = a.seeds[:1]
    os.makedirs(SALIDA, exist_ok=True)

    print(f"denoising.py  dev={dev}  fuentes={a.fuentes}  p={a.p}  modo={a.modo}  "
          f"seeds={a.seeds}  pasos={cfg['ft_steps']}")
    filas = []
    for fuente in a.fuentes:
        for seed in a.seeds:
            filas += corre_fuente_semilla(fuente, seed, cfg, a)
    veredictos = resumen(filas, a)
    procedencia(cfg, a, veredictos)
    print(f"\nescrito en {SALIDA}/")


if __name__ == "__main__":
    main()
