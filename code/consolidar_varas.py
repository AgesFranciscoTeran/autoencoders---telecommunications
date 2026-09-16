#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidar_varas.py -- convierte `data/varas_definitivas.json` en
`data/baselines_clasicos.csv`, que es el archivo que leen el resto de piezas
del proyecto (figuras, presentacion, denoising, combinado).

POR QUE EXISTE
--------------
El CSV de varas se venia manteniendo a mano. Eso tuvo dos consecuencias, las
dos detectadas en revision:

  (a) El archivo publicado no tenia generador en el repositorio: se podia leer
      pero no reproducir. Un artefacto que hay que conservar no es un resultado
      reproducible.
  (b) `generar_tablas.py` escribia un archivo con el MISMO nombre usando el
      cuantizador uniforme min/max, es decir, la vara retractada. Ejecutar el
      nivel 1 de reproduccion deshacia en silencio la correccion mas cara del
      proyecto.

Este script separa los dos papeles: `vara_definitiva.py` CALCULA (numeros), este
FORMATEA (sin ninguna decision cientifica dentro). Si el CSV y el JSON no
coinciden, el que manda es el JSON.

QUE ESCRIBE
-----------
Una fila por (fuente, escalon), con la vara vigente y las tres alternativas que
se probaron, para que el recalculo quede auditable desde el propio CSV:

  ber_baseline           el mejor metodo clasico a ese presupuesto de bits
  metodo                 cual fue
  ber_minmax_historico   cuantizacion uniforme + transpuesta (la vara RETRACTADA)
  ber_lloyd              Lloyd-Max + transpuesta  (la vigente en la mayoria)
  ber_lloyd_ls           Lloyd-Max + sintesis por minimos cuadrados

Uso:  python3 code/consolidar_varas.py
      python3 code/consolidar_varas.py --verificar   # no escribe; solo compara
"""
import argparse, csv, json, math, os, sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_IN = os.path.join(RAIZ, "data", "varas_definitivas.json")
CSV_OUT = os.path.join(RAIZ, "data", "baselines_clasicos.csv")

DIM = 500
FUENTES = ["random", "oversamp", "markov", "lowdim", "code"]
LADDER = [35, 70, 125, 250]
COLS = ["fuente", "latente_bits", "tasa", "compresion", "ber_baseline", "metodo",
        "ber_minmax_historico", "ber_lloyd", "ber_lloyd_ls"]


def r4(x):
    """Redondea a 4 decimales; NaN y ausentes salen como cadena vacia."""
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(x) else round(x, 4)


def construir(filas_json):
    filas = []
    for r in filas_json:
        L = int(r["latente_bits"])
        filas.append({
            "fuente": r["fuente"],
            "latente_bits": L,
            "tasa": round(L / DIM, 4),
            "compresion": round(DIM / L, 2),
            "ber_baseline": r4(r["ber_baseline"]),
            "metodo": r["metodo"],
            "ber_minmax_historico": r4(r.get("ber_unif_transp")),
            "ber_lloyd": r4(r.get("ber_lloyd_transp")),
            "ber_lloyd_ls": r4(r.get("ber_lloyd_ls")),
        })
    orden = {f: i for i, f in enumerate(FUENTES)}
    filas.sort(key=lambda d: (orden.get(d["fuente"], 99), d["latente_bits"]))
    return filas


def comprueba_cobertura(filas):
    """La omision de una fuente entera es el fallo que ya ocurrio una vez:
       el CSV vigente perdio las cuatro filas de `code` sin que nadie lo notara,
       y los consumidores se quedaron con la vara en blanco."""
    faltan = [(f, L) for f in FUENTES for L in LADDER
              if not any(d["fuente"] == f and d["latente_bits"] == L for d in filas)]
    if faltan:
        print("[!] faltan celdas en varas_definitivas.json:")
        for f, L in faltan:
            print(f"      {f} L={L}")
        print("    Vuelve a correr `python3 code/vara_definitiva.py` con todas "
              "las fuentes antes de consolidar.")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true",
                    help="no escribe: compara el CSV en disco con el JSON")
    a = ap.parse_args()

    if not os.path.exists(JSON_IN):
        sys.exit(f"[!] no existe {JSON_IN}. Corre antes: python3 code/vara_definitiva.py")

    filas = construir(json.load(open(JSON_IN)))
    completo = comprueba_cobertura(filas)

    if a.verificar:
        if not os.path.exists(CSV_OUT):
            sys.exit(f"[!] no existe {CSV_OUT}")
        crudo = list(csv.DictReader(open(CSV_OUT)))
        # se compara por clave, no por posicion: el orden de las filas no es
        # parte del resultado
        actual = {(d["fuente"], int(d["latente_bits"])):
                  {k: str(d.get(k, "")) for k in COLS} for d in crudo}
        esperado = {(d["fuente"], d["latente_bits"]):
                    {k: str(v) for k, v in d.items()} for d in filas}
        difs = [(esperado[k], actual[k]) for k in esperado
                if k in actual and esperado[k] != actual[k]]
        difs += [({"fuente": k[0], "latente_bits": k[1]}, {"(ausente)": ""})
                 for k in esperado if k not in actual]
        if len(actual) != len(esperado) or difs:
            print(f"[!] el CSV NO coincide con el JSON "
                  f"({len(actual)} filas en disco, {len(esperado)} esperadas, "
                  f"{len(difs)} discrepancias)")
            for e, b in difs[:5]:
                print(f"    esperado {e}\n    en disco {b}")
            sys.exit(1)
        print(f"[ok] {CSV_OUT} coincide con el JSON ({len(actual)} filas)")
        return

    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(filas)
    print(f"[guardado] {CSV_OUT}  ({len(filas)} filas)")

    print(f"\n{'fuente':10s}{'L':>5s}{'vara':>9s}  metodo")
    for d in filas:
        print(f"{d['fuente']:10s}{d['latente_bits']:5d}{str(d['ber_baseline']):>9s}"
              f"  {d['metodo']}")
    if not completo:
        sys.exit(1)


if __name__ == "__main__":
    main()
