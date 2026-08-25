#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificacion post-corrida de ae2.py (v4).
    python3 check_v4.py ./res
Revisa, en orden: completitud, controles de sanidad (lo que invalidaria todo),
resultado principal, costo de la anidacion y viabilidad con FEC.
"""
import json, os, sys
import numpy as np
import pandas as pd

LADDER = [35, 70, 125, 250]
REGIMES = ["random", "oversamp", "markov", "lowdim", "code"]
TRAINERS = ["nested", "direct", "stacked"]
OK, BAD, WARN = "  OK  ", " FALLA", " AVISO"


def load(outdir):
    rp = os.path.join(outdir, "v4_resultados.jsonl")
    bp = os.path.join(outdir, "v4_baselines.jsonl")
    for p in (rp, bp):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            sys.exit(f"[!] falta o esta vacio: {p}")
    with open(rp) as f:
        res = pd.read_json(f, lines=True)
    with open(bp) as f:
        bl = pd.read_json(f, lines=True)
    m = res.merge(bl[["regime", "latent_bits", "baseline_ber", "baseline_metodo"]],
                  on=["regime", "latent_bits"], how="left")
    m["sobre_slb"] = m.test_ber - m.slb_ber
    m["gana_baseline"] = m.test_ber < m.baseline_ber
    m["gap"] = m.test_ber - m.train_ber
    return m


def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


def main(outdir):
    m = load(outdir)

    # ---------------------------------------------------------------- 1
    sec("1. COMPLETITUD")
    esperado = len(REGIMES) * len(TRAINERS) * len(LADDER)
    print(f"  filas: {len(m)} / {esperado} esperadas")
    faltan = [(r, t) for r in REGIMES for t in TRAINERS
              if len(m[(m.regime == r) & (m.train_mode == t)]) != len(LADDER)]
    if faltan:
        print(f"{BAD} combinaciones incompletas (revisa shard_*.log):")
        for r, t in faltan:
            n = len(m[(m.regime == r) & (m.train_mode == t)])
            print(f"         {r:10s} {t:8s} -> {n}/4 escalones")
    else:
        print(f"{OK} las {len(REGIMES)}x{len(TRAINERS)} combinaciones estan completas")

    nan = m[m.test_ber.isna() | m.train_ber.isna()]
    print(f"{BAD if len(nan) else OK} filas con NaN: {len(nan)}"
          + ("  <- entrenamiento divergio" if len(nan) else ""))

    # ---------------------------------------------------------------- 2
    sec("2. CONTROLES DE SANIDAD  (si algo falla aqui, el resto no vale)")

    viol = m[m.sobre_slb < -1e-4]
    if len(viol):
        print(f"{BAD} {len(viol)} filas VIOLAN la cota de Shannon (test_BER < SLB).")
        print("        Causa casi segura: fuga train/test o la fuente cambia entre splits.")
        print(viol[["regime", "train_mode", "latent_bits", "test_ber", "slb_ber"]]
              .to_string(index=False))
    else:
        print(f"{OK} ninguna fila viola la cota SLB (nadie hace trampa)")

    r = m[m.regime == "random"]
    peor = r.groupby("latent_bits").test_ber.min()
    mal = [L for L in LADDER if L in peor.index and peor[L] < 0.05]
    if mal:
        print(f"{BAD} 'random' comprime en L={mal}. Es IMPOSIBLE: bits i.i.d. no")
        print("        son comprimibles. Indica contaminacion train/test.")
    else:
        print(f"{OK} 'random' NO comprime (control negativo correcto)")
    print(f"         random: " + "  ".join(
        f"L{L}:{peor[L]:.4f}" for L in LADDER if L in peor.index))

    g = m[m.regime == "random"].gap.mean()
    print(f"{OK} brecha train/test en 'random': {g:+.4f} "
          f"({'memoriza, esperado' if g > 0.02 else 'baja'})")

    # ---------------------------------------------------------------- 3
    sec("3. RESULTADO PRINCIPAL: AE vs vara clasica vs cota")
    mejor = (m.loc[m.groupby(["regime", "latent_bits"]).test_ber.idxmin()]
             .sort_values(["regime", "latent_bits"]))
    print(f"{'fuente':10s} {'L':>4s} {'AE':>8s} {'modo':>8s} {'clasico':>8s} "
          f"{'SLB':>8s} {'sobreSLB':>9s}  gana?")
    for _, row in mejor.iterrows():
        print(f"{row.regime:10s} {row.latent_bits:4d} {row.test_ber:8.4f} "
              f"{row.train_mode:>8s} {row.baseline_ber:8.4f} {row.slb_ber:8.4f} "
              f"{row.sobre_slb:+9.4f}  {'SI' if row.gana_baseline else 'no'}")

    n_gana = int(mejor.gana_baseline.sum())
    print(f"\n  El AE gana en {n_gana}/{len(mejor)} puntos de operacion.")

    # ---------------------------------------------------------------- 4
    sec("4. LA PREGUNTA CLAVE: se ve la redundancia algebraica?")
    c = mejor[mejor.regime == "code"]
    rnd = mejor[mejor.regime == "random"].set_index("latent_bits").test_ber
    for _, row in c.iterrows():
        ref = rnd.get(row.latent_bits, np.nan)
        d = row.test_ber - ref
        # solo hay estructura detectada si 'code' sale MEJOR que ruido puro
        if d < -0.02:
            v = "SI ve estructura"
        elif abs(d) <= 0.02:
            v = "IGUAL a ruido"
        else:
            v = "PEOR que ruido (revisar)"
        print(f"  L={row.latent_bits:3d}: code={row.test_ber:.4f}  random={ref:.4f}  "
              f"dif={d:+.4f}  -> {v}")
    c250 = c[c.latent_bits == 250]
    if len(c250):
        v = float(c250.test_ber.iloc[0])
        print(f"\n  En L=250 el oraculo logra BER=0.0000 y el AE logra {v:.4f}.")
        print("  -> " + ("RESULTADO NEGATIVO confirmado: la redundancia existe "
                         "(2x, sin perdida)\n     pero el AE no la encuentra."
                         if v > 0.05 else
                         "OJO: el AE SI encontro estructura algebraica. Esto REFUTA\n"
                         "     la hipotesis de paridad y es un hallazgo mejor. Verificar."))

    # ---------------------------------------------------------------- 5
    sec("5. COSTO DE LA ANIDACION (nested vs direct)")
    piv = m.pivot_table(index=["regime", "latent_bits"], columns="train_mode",
                        values="test_ber")
    if {"nested", "direct"}.issubset(piv.columns):
        piv["costo"] = piv["nested"] - piv["direct"]
        print(piv.round(4).to_string())
        cm = piv["costo"].mean()
        print(f"\n  Costo medio de anidar: {cm:+.4f} BER")
        print("  -> " + ("esperado (brecha de successive refinement); repórtalo, no lo escondas."
                         if cm > 0 else
                         "la anidacion NO cuesta (o ayuda como regularizador). Vale la pena decirlo."))
    else:
        print(f"{WARN} faltan modos para comparar")

    # ---------------------------------------------------------------- 6
    sec("6. ARGUMENTO FEC (salida blanda)")
    if "ber_top90" in m.columns:
        f = mejor[(mejor.test_ber > 1e-4) & (mejor.regime != "random")]
        print(f"{'fuente':10s} {'L':>4s} {'BER':>8s} {'BER top90':>10s} {'factor':>8s}")
        for _, row in f.iterrows():
            fac = row.test_ber / max(row.ber_top90, 1e-9)
            print(f"{row.regime:10s} {row.latent_bits:4d} {row.test_ber:8.4f} "
                  f"{row.ber_top90:10.4f} {fac:8.1f}x")
        print("\n  factor >> 1 = los errores se concentran en bits poco confiables")
        print("  -> un FEC de decision blanda los limpia. Ese es el argumento de viabilidad.")
    else:
        print(f"{WARN} sin columnas de confianza")

    out = os.path.join(outdir, "v4_revision.csv")
    m.sort_values(["regime", "train_mode", "latent_bits"]).to_csv(out, index=False)
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./res")
