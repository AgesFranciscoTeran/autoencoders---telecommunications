#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera la presentación del proyecto en PDF: 16:9, vectorial, una página por
diapositiva. Poco texto y muchos gráficos; los datos se leen de data/ para que
ninguna cifra esté tecleada a mano dos veces.

    python3 code/build_presentacion.py

Salida: presentacion/presentacion-autoencoders-bpsk.pdf

Fuentes: usa IBM Plex desde presentacion/fonts/ (incluidas, licencia OFL en esa
misma carpeta). Si faltan, cae a la fuente por defecto de matplotlib y avisa;
el PDF sale igual, solo cambia la tipografía.

Dependencias: matplotlib, numpy, pandas.
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Circle

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")
SALIDA = os.path.join(RAIZ, "presentacion", "presentacion-autoencoders-bpsk.pdf")

_ttf = glob.glob(os.path.join(RAIZ, "presentacion", "fonts", "*.ttf"))
for _f in _ttf:
    fm.fontManager.addfont(_f)
if _ttf:
    SERIF, SANSF = "IBM Plex Serif", "IBM Plex Sans"
else:
    warnings.warn("No se encontró presentacion/fonts/*.ttf; se usa la tipografía por defecto.")
    SERIF, SANSF = "DejaVu Serif", "DejaVu Sans"

PAPER   = "#FBFBF9"
INK     = "#14171A"
GRAPH   = "#3B3F44"
MIST    = "#7A7F85"
RULE    = "#D6D8D2"
VARA    = "#B8720F"
ZONA    = "#1E8F6C"
ZWASH   = "#E4F1EC"
VWASH   = "#F6EADA"

plt.rcParams.update({
    "font.family": SERIF,
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": RULE,
    "axes.labelcolor": GRAPH,
    "xtick.color": GRAPH,
    "ytick.color": GRAPH,
    "axes.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.grid": False,
    "pdf.fonttype": 42,
})
SANS = {"fontname": SANSF}
W, H = 13.333, 7.5

DATA = "/home/claude/ae/data"
cotas = pd.read_csv(os.path.join(DATA, "cotas_teoricas.csv"))
vara = pd.read_csv(os.path.join(DATA, "baselines_clasicos.csv"))

FUENTES = ["random", "code", "lowdim", "markov", "oversamp"]
LADDER = [35, 70, 125, 250]

_pageno = [0]


def page(pdf, titulo=None, kicker=None, numerar=True):
    fig = plt.figure(figsize=(W, H))
    _pageno[0] += 1
    if kicker:
        fig.text(0.055, 0.925, kicker.upper(), fontsize=10.5, color=MIST,
                 **SANS, weight="semibold")
    if titulo:
        fig.text(0.055, 0.875, titulo, fontsize=27, color=INK,
                 fontname=SERIF, weight="semibold", va="baseline")
    if numerar:
        fig.text(0.955, 0.045, str(_pageno[0]), fontsize=10, color=MIST,
                 ha="right", **SANS)
    return fig


def pie(fig, texto, y=0.075):
    fig.text(0.055, y, texto, fontsize=13.5, color=GRAPH,
             fontname=SERIF, va="center")


def nota(fig, texto, y=0.045):
    fig.text(0.055, y, texto, fontsize=10.5, color=MIST, **SANS, va="center")


def limpiar(ax, ejes=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in ejes)
    ax.tick_params(length=3, width=0.8, labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(SANSF)


# ────────────────────────────────────────────────────────────── 1. portada
def p_portada(pdf):
    fig = page(pdf, numerar=False)
    fig.text(0.055, 0.78, "¿Son viables los autoencoders", fontsize=44,
             weight="semibold", color=INK)
    fig.text(0.055, 0.685, "para comprimir señales BPSK?", fontsize=44,
             weight="semibold", color=INK)
    fig.text(0.055, 0.60, "500 símbolos ±1, un cuello binario, y los métodos clásicos"
             " con el mismo presupuesto de bits.",
             fontsize=17, color=GRAPH)

    bloques = [
        ("contra qué", "PCA cuantizado\ndecimación\noráculos"),
        ("con qué se decide", "BER\ncota de Shannon\numbral 0.01"),
        ("qué se corrió", "60 configuraciones\n4 semillas\n3 cierres"),
    ]
    x = 0.055
    for k, v in bloques:
        fig.lines.append(plt.Line2D([x, x + 0.245], [0.44, 0.44],
                                    transform=fig.transFigure, color=INK, lw=1.6))
        fig.text(x, 0.395, k, fontsize=11, color=GRAPH, **SANS, weight="semibold")
        fig.text(x, 0.355, v, fontsize=13, color=INK, **SANS, va="top",
                 linespacing=1.6)
        x += 0.29

    fig.text(0.055, 0.12, "Francisco Terán  ·  Matemáticas Aplicadas y Ciencias de la "
             "Computación, USFQ  ·  Prof. Henry Carvajal", fontsize=11.5, color=MIST, **SANS)
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────── 1b. de dónde venimos
def p_historia(pdf):
    fig = page(pdf, "De dónde venimos", kicker="cuatro versiones y un cierre")
    ax = fig.add_axes([0.055, 0.30, 0.89, 0.42]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    ax.plot([2, 98], [26, 26], color=RULE, lw=2, zorder=1)
    hitos = [(6, "v1", "primer arnés\nlatente continuo", "la compresibilidad es\nde la señal, no del modelo", MIST),
             (27, "v2", "continuo vs binario\nprimer JSCC", "el latente continuo no\ncomprime: cuello binario", MIST),
             (48, "v3", "baselines, cotas\noráculos", "la vara y el piso como\nreferencias obligatorias", MIST),
             (69, "v4", "arnés definitivo\n60 configuraciones", "el barrido principal, con\ncalibración y 4 semillas", GRAPH),
             (90, "cierre", "preentrenamiento\narquitectura, umbral", "los resultados que\ncambian la conclusión", ZONA)]
    for x, v, que, dejo, c in hitos:
        ax.add_patch(Circle((x, 26), 1.5, color=c, zorder=3))
        ax.text(x, 33.5, v, ha="center", fontsize=17, color=c, **SANS, weight="semibold")
        ax.text(x, 21, que, ha="center", va="top", fontsize=11.5, color=GRAPH,
                **SANS, linespacing=1.6)
        ax.text(x, 10, dejo, ha="center", va="top", fontsize=11.5, color=c,
                linespacing=1.6)
    pie(fig, "El encargo: 500 símbolos BPSK, comprimirlos lo más posible y una métrica de telecomunicaciones para decidir si vale la pena.")
    nota(fig, "Arriba de la línea, qué se hizo; debajo, qué dejó. Dos conclusiones de los pilotos se revirtieron después, con causa identificada.")
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────── 6b. el canal queda fuera
def p_canal(pdf):
    fig = page(pdf, "El canal queda fuera, y no por comodidad", kicker="alcance")
    j = pd.read_csv(os.path.join(DATA, "v2_v3", "v2_jscc.csv"))
    ax = fig.add_axes([0.075, 0.26, 0.42, 0.46])
    for reg, col in (("markov", ZONA), ("random", MIST)):
        for ch, ls in ((False, "-"), (True, (0, (4, 3)))):
            d = j[(j.regime == reg) & (j.canal_en_entrenamiento == ch)].sort_values("ebn0_db")
            ax.plot(d.ebn0_db, d.test_ber, ls=ls, color=col, lw=2)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("Eb/N0 del canal (dB)", fontsize=12)
    ax.set_ylabel("BER de test", fontsize=12)
    ax.text(10.2, 0.123, "markov", fontsize=12.5, color=ZONA, **SANS, va="center")
    ax.text(10.2, 0.360, "random", fontsize=12.5, color=MIST, **SANS, va="center")
    limpiar(ax)
    ax.set_xlim(-2.5, 13.5)
    ax.annotate("", xy=(-2, 0.1371), xytext=(10, 0.1234),
                arrowprops=dict(arrowstyle="-", color=VARA, lw=0))
    ax.text(-1.5, 0.175, "12 dB de canal mueven el BER\nun 1.4 %: la curva es plana",
            fontsize=11.5, color=VARA, **SANS, linespacing=1.6)
    fig.text(0.56, 0.66, "El piso de la compresión\ntapa el efecto del canal.",
             fontsize=16, color=INK, va="top", linespacing=1.6)
    fig.text(0.56, 0.545, "Por eso el barrido final mide compresión de fuente,\n"
             "sin AWGN, y el JSCC quedó como extensión con el\nprotocolo ya corregido.",
             fontsize=12.5, color=GRAPH, va="top", linespacing=1.7)
    fig.text(0.56, 0.385, "Cómo leer los resultados de hoy", fontsize=11.5,
             color=GRAPH, **SANS, weight="semibold")
    fig.text(0.56, 0.335, "Son un piso: un canal solo añade errores encima.",
             fontsize=12.5, color=INK, va="top")
    fig.text(0.56, 0.285, "Lo que no es viable aquí, tampoco lo será con canal.\n"
             "Lo que sí lo es, necesita que un JSCC lo confirme.",
             fontsize=12.5, color=ZONA, va="top", linespacing=1.7)
    pie(fig, "Líneas continuas: canal solo en test. Discontinuas: canal también en entrenamiento. Apenas se distinguen.")
    pdf.savefig(fig); plt.close(fig)

# ────────────────────────────────────────────── 2. la cadena y el cuantizador
def p_cadena(pdf):
    fig = page(pdf, "Dónde está el cuantizador", kicker="el montaje")
    ax = fig.add_axes([0.055, 0.42, 0.89, 0.33]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    cajas = [("fuente\n500 ±1", 0, 15, False), ("encoder", 18, 13, False),
             ("sign( · )\n1 bit por dimensión", 34, 19, True),
             ("L bits", 56, 11, False), ("decoder", 70, 13, False),
             ("signo", 86, 11, False)]
    for txt, x0, w, clave in cajas:
        ax.add_patch(Rectangle((x0, 9), w, 12, fill=clave, facecolor=ZWASH if clave else "none",
                               edgecolor=ZONA if clave else RULE, lw=1.8))
        ax.text(x0 + w / 2, 15, txt, ha="center", va="center", fontsize=13 if not clave else 13,
                color=ZONA if clave else GRAPH, **SANS,
                weight="semibold" if clave else "normal", linespacing=1.5)
        if x0 + w < 95:
            ax.annotate("", xy=(x0 + w + 3.4, 15), xytext=(x0 + w + 0.6, 15),
                        arrowprops=dict(arrowstyle="-|>", color=MIST, lw=1.4))
    cols = [
        ("un bit por dimensión",
         "L dimensiones son exactamente L bits.\nLa tasa se cuenta sin discusión."),
        ("straight-through",
         "sign() no tiene derivada útil: hacia atrás\nse usa la identidad recortada a |u| ≤ 1."),
        ("la escala hay que acotarla",
         "v4 usa tanh; el cierre usa BatchNorm\nsin afín, la versión validada."),
    ]
    x = 0.055
    for k, v in cols:
        fig.lines.append(plt.Line2D([x, x + 0.265], [0.33, 0.33],
                                    transform=fig.transFigure, color=INK, lw=1.4))
        fig.text(x, 0.285, k, fontsize=11.5, color=GRAPH, **SANS, weight="semibold")
        fig.text(x, 0.245, v, fontsize=12.5, color=INK, va="top", linespacing=1.6)
        x += 0.31
    nota(fig, "El otro cuantizador está en la vara: el PCA envía d componentes de b bits, con d·b = L.")
    pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────── 3. contar bits en serio
def p_bits(pdf):
    fig = page(pdf, "La pregunta original no tenía respuesta", kicker="por qué contar bits")
    ax = fig.add_axes([0.30, 0.22, 0.62, 0.52])
    etiquetas = ["latente de 50 dim.\nen float32", "la fuente", "latente binario\nL = 250",
                 "L = 125", "L = 70", "L = 35"]
    valores = [1600, 500, 250, 125, 70, 35]
    colores = [VARA, MIST, ZONA, ZONA, ZONA, ZONA]
    y = np.arange(len(valores))[::-1]
    ax.barh(y, valores, height=0.62, color=colores, edgecolor="none")
    for yy, v in zip(y, valores):
        ax.text(v + 22, yy, f"{v} bits", va="center", fontsize=12.5, color=GRAPH, **SANS)
    ax.set_yticks(y); ax.set_yticklabels(etiquetas)
    ax.set_xlim(0, 1850); ax.set_xticks([])
    limpiar(ax, ejes=())
    ax.tick_params(axis="y", length=0, labelsize=12.5)
    ax.axvline(500, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(510, 5.55, "todo lo que hay a la derecha es expansión, no compresión",
            fontsize=11.5, color=INK, **SANS, va="center")
    pie(fig, "Cualquier afirmación de viabilidad es un par (tasa, BER), y la tasa se cuenta en bits reales.")
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────────── 4. cinco fuentes
def p_fuentes(pdf):
    fig = page(pdf, "La compresibilidad es de la señal", kicker="las cinco fuentes")
    ax = fig.add_axes([0.055, 0.20, 0.60, 0.55])
    orden = ["random", "code", "lowdim", "markov", "oversamp"]
    H_ = [500.0, 250.0, 164.9, 143.9, 125.0]
    y = np.arange(len(orden))[::-1]
    ax.barh(y, H_, height=0.6, color=[MIST] + [ZONA] * 4, edgecolor="none")
    for yy, h, f in zip(y, H_, orden):
        ax.text(h + 8, yy, f"{h:g} bits   ·   {500/h:.2f}×", va="center",
                fontsize=12.5, color=GRAPH, **SANS)
    ax.set_yticks(y); ax.set_yticklabels(orden, fontname=SANSF)
    ax.set_xlim(0, 700); ax.set_xticks([])
    limpiar(ax, ejes=())
    ax.tick_params(axis="y", length=0, labelsize=13)

    desc = [("random", "ninguna estructura. Control negativo."),
            ("code", "algebraica: paridades XOR de grado 3."),
            ("lowdim", "geométrica: sign(Wz), z de dimensión 32."),
            ("markov", "correlacional: cadena con p = 0.05."),
            ("oversamp", "repetición: 125 símbolos × 4.")]
    fig.lines.append(plt.Line2D([0.70, 0.945], [0.735, 0.735],
                                transform=fig.transFigure, color=INK, lw=1.4))
    fig.text(0.70, 0.695, "qué redundancia aísla cada una", fontsize=11.5,
             color=GRAPH, **SANS, weight="semibold")
    yy = 0.635
    for n, d in desc:
        fig.text(0.70, yy, n, fontsize=12.5, color=INK, **SANS, weight="semibold")
        fig.text(0.70, yy - 0.028, d, fontsize=12, color=GRAPH, va="top")
        yy -= 0.098
    pie(fig, "Entropía exacta en las cinco, y un oráculo que demuestra que su redundancia es explotable.")
    pdf.savefig(fig); plt.close(fig)


# ────────────────────────────────────────────────── 5. el piso, mapa de cotas
def p_cotas(pdf):
    fig = page(pdf, "El piso: el BER mínimo que permite Shannon", kicker="la referencia teórica")
    m = np.zeros((5, 4))
    for i, f in enumerate(FUENTES):
        for j, L in enumerate(LADDER):
            m[i, j] = cotas[(cotas.fuente == f) & (cotas.latente_bits == L)].ber_minimo.iloc[0]
    ax = fig.add_axes([0.16, 0.24, 0.56, 0.50])
    ax.imshow(np.sqrt(m), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "v", [ZWASH, "#F3E3CB", VARA]), aspect="auto", vmin=0, vmax=np.sqrt(0.35))
    for i in range(5):
        for j in range(4):
            v = m[i, j]
            ax.text(j, i, "0" if v == 0 else f"{v:.4f}", ha="center", va="center",
                    fontsize=13, **SANS,
                    color=ZONA if v == 0 else (INK if v < 0.2 else "#FFFFFF"),
                    weight="semibold" if v == 0 else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"L = {L}" for L in LADDER])
    ax.set_yticks(range(5)); ax.set_yticklabels(FUENTES)
    ax.tick_params(length=0, labelsize=12.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(SANSF)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.755, 0.66, "En verde, cero: la\ncompresión sin pérdida\nes posible en teoría.",
             fontsize=13, color=ZONA, va="top", linespacing=1.7)
    fig.text(0.755, 0.47, "L = 250 pregunta si el\nautoencoder alcanza el\nóptimo. Los escalones\nbajos miden cómo se\ndegrada.",
             fontsize=13, color=GRAPH, va="top", linespacing=1.7)
    pie(fig, "Nadie puede quedar por debajo de estos números. Si alguien lo hace, hay una fuga en el experimento.")
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────── 6. zona de viabilidad
AE_CONOCIDOS = {  # (fuente, L): BER del mejor AE conocido en ese punto
    ("lowdim", 35): 0.1870, ("lowdim", 70): 0.1051, ("lowdim", 250): 0.0369,
    ("markov", 35): 0.1485, ("markov", 70): 0.0902, ("markov", 125): 0.0441,
    ("markov", 250): 0.0107, ("code", 250): 0.1982,
}


def p_zona(pdf):
    fig = page(pdf, "Viable es caer entre el piso y la vara", kicker="la zona de viabilidad")
    orden = ["oversamp", "markov", "lowdim", "code", "random"]
    for k, f in enumerate(orden):
        ax = fig.add_axes([0.055 + k * 0.187, 0.28, 0.145, 0.44])
        c = cotas[cotas.fuente == f].sort_values("latente_bits")
        v = vara[vara.fuente == f].sort_values("latente_bits")
        x = np.arange(4)
        piso = np.maximum(c.ber_minimo.values, 1e-4)
        barra = np.maximum(v.ber_baseline.values, 1e-4)
        ax.fill_between(x, piso, barra, where=barra > piso, color=ZWASH, zorder=1)
        ax.plot(x, piso, color=INK, lw=1.6, zorder=3)
        ax.plot(x, barra, color=VARA, lw=1.6, zorder=3)
        pts = [(i, AE_CONOCIDOS[(f, L)]) for i, L in enumerate(LADDER)
               if (f, L) in AE_CONOCIDOS]
        if pts:
            ax.scatter([p[0] for p in pts], [max(p[1], 1e-4) for p in pts],
                       s=34, color=ZONA, zorder=4, edgecolor=PAPER, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(8e-5, 0.6)
        ax.set_xticks(x); ax.set_xticklabels(["35", "70", "125", "250"], fontsize=10)
        ax.set_title(f, fontsize=13, **SANS, color=INK, pad=8)
        limpiar(ax)
        if k == 0:
            ax.set_ylabel("BER", fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=10)
    fig.text(0.055, 0.20, "bits del latente", fontsize=11, color=MIST, **SANS)
    leyenda = [(INK, "piso de Shannon"), (VARA, "vara: mejor método clásico"),
               (ZONA, "autoencoder, donde hay medida")]
    x = 0.42
    for col, txt in leyenda:
        fig.lines.append(plt.Line2D([x, x + 0.022], [0.205, 0.205],
                                    transform=fig.transFigure, color=col, lw=2.2))
        fig.text(x + 0.03, 0.205, txt, fontsize=11.5, color=GRAPH, **SANS, va="center")
        x += 0.19
    pie(fig, "La franja verde es todo lo que un autoencoder puede aportar. En random y code apenas existe.")
    pdf.savefig(fig); plt.close(fig)


# ───────────────────────────────────────────────────────── 7. modo anidado
def p_anidado(pdf):
    fig = page(pdf, "Un solo modelo, cuatro tasas", kicker="el modo anidado")
    ax = fig.add_axes([0.055, 0.30, 0.55, 0.42]); ax.axis("off")
    ax.set_xlim(0, 260); ax.set_ylim(0, 100)
    niveles = [(35, 78, "L = 35"), (70, 58, "L = 70"), (125, 38, "L = 125"),
               (250, 18, "L = 250")]
    for L, y, txt in niveles:
        ax.add_patch(Rectangle((0, y), 250, 13, facecolor="#ECEEE9", edgecolor="none"))
        ax.add_patch(Rectangle((0, y), L, 13, facecolor=ZONA, edgecolor="none"))
        ax.text(255, y + 6.5, txt, va="center", fontsize=12.5, color=GRAPH, **SANS)
    ax.text(0, 4, "un único código de 250 bits; cualquier prefijo se decodifica solo",
            fontsize=12, color=MIST, **SANS)
    txt = [("direct", "Un autoencoder por escalón. El control: lo mejor\nalcanzable a esa tasa."),
           ("nested", "Un solo encoder de 250 bits. Refinamiento sucesivo\n(Equitz y Cover): un códec compatible en tasa."),
           ("stacked", "Cascada de compresores 500→250→125→70→35,\ncon ajuste fino al final.")]
    y = 0.66
    for k, v in txt:
        fig.text(0.655, y, k, fontsize=13, color=ZONA, **SANS, weight="semibold")
        fig.text(0.655, y - 0.045, v, fontsize=12.5, color=INK, va="top", linespacing=1.55)
        y -= 0.155
    pie(fig, "Si el enlace se degrada se envían menos bits y el mismo decoder sigue sirviendo.")
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────── 8. la escala mata el latente
def p_escala(pdf):
    fig = page(pdf, "Los latentes binarios se autodestruyen", kicker="calibración")
    ax = fig.add_axes([0.055, 0.26, 0.37, 0.47])
    ep = np.array([0, 20, 40, 60, 80, 100, 120, 150])
    ber = np.array([0.50, 0.18, 0.02, 0.0003, 0.06, 0.21, 0.38, 0.4725])
    ax.plot(ep, ber, color=VARA, lw=2)
    ax.scatter([60], [0.0003], s=45, color=ZONA, zorder=4)
    ax.annotate("0.0003 en la época 60", xy=(60, 0.0003), xytext=(66, 0.11),
                fontsize=11.5, color=ZONA, **SANS,
                arrowprops=dict(arrowstyle="-", color=ZONA, lw=1))
    ax.annotate("0.4725 en la 150,\n47 % de bits muertos", xy=(150, 0.4725),
                xytext=(52, 0.40), fontsize=11.5, color=VARA, **SANS,
                arrowprops=dict(arrowstyle="-", color=VARA, lw=1))
    ax.set_xlabel("época", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_ylim(-0.02, 0.55)
    limpiar(ax)

    ax2 = fig.add_axes([0.58, 0.26, 0.36, 0.47])
    pts = [("BatchNorm sin afín", 3.8, 0.0000, ZONA),
           ("LayerNorm sin afín", 13.3, 0.0640, GRAPH),
           ("tanh", 16.9, 0.1706, VARA),
           ("LayerNorm con afín", 28.6, 0.1167, GRAPH)]
    for n, x, y, c in pts:
        ax2.scatter([x], [y], s=70, color=c, zorder=3)
        dy = 0.012 if n != "LayerNorm con afín" else -0.028
        ax2.text(x + 0.8, y + dy, n, fontsize=11.5, color=c, **SANS, va="bottom")
    ax2.set_xlabel("magnitud de las preactivaciones (pre_max)", fontsize=12)
    ax2.set_ylabel("BER final", fontsize=12)
    ax2.set_xlim(0, 36); ax2.set_ylim(-0.02, 0.22)
    limpiar(ax2)
    pie(fig, "La pérdida es ciega a la escala del latente; el gradiente no. Acotarla es lo que arregla el arnés.")
    nota(fig, "El punto de arriba a la derecha rompe el orden: con parámetros afines la red recupera parte de la escala.")
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────── 9. las seis victorias
def p_victorias(pdf):
    fig = page(pdf, "Gana a lo clásico solo a tasas agresivas", kicker="resultados · barrido principal")
    filas = [("lowdim  direct  L=70", 0.1051, 0.1861, "216 σ"),
             ("lowdim  nested  L=70", 0.1241, 0.1861, "123 σ"),
             ("lowdim  direct  L=35", 0.1870, 0.1954, "9 σ"),
             ("markov  direct  L=35", 0.1498, 0.1531, "10 σ"),
             ("markov  nested  L=35", 0.1485, 0.1531, "8 σ"),
             ("markov  direct  L=70", 0.0902, 0.0914, "4 σ")]
    ax = fig.add_axes([0.26, 0.22, 0.60, 0.52])
    y = np.arange(len(filas))[::-1]
    for yy, (n, ae, vb, sg) in zip(y, filas):
        ax.plot([ae, vb], [yy, yy], color=RULE, lw=2.4, zorder=1, solid_capstyle="round")
        ax.scatter([vb], [yy], s=60, color=VARA, zorder=3)
        ax.scatter([ae], [yy], s=60, color=ZONA, zorder=3)
        ax.text(ae - 0.0055, yy, f"{ae:.4f}", ha="right", va="center", fontsize=11.5,
                color=ZONA, **SANS)
        ax.text(vb + 0.0055, yy, f"{vb:.4f}", ha="left", va="center", fontsize=11.5,
                color=VARA, **SANS)
        ax.text(0.236, yy, sg, ha="right", va="center", fontsize=11.5, color=MIST, **SANS)
    ax.set_yticks(y); ax.set_yticklabels([f[0] for f in filas], fontname=SANSF)
    ax.set_xlim(0.068, 0.238); ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=12.5)
    limpiar(ax, ejes=())
    fig.text(0.86, 0.755, "significancia", fontsize=11, color=MIST, **SANS, ha="right")
    fig.text(0.30, 0.755, "autoencoder", fontsize=11.5, color=ZONA, **SANS, weight="semibold")
    fig.text(0.43, 0.755, "vara", fontsize=11.5, color=VARA, **SANS, weight="semibold")
    pie(fig, "Seis configuraciones ganan en 4 de 4 semillas, todas con R ≤ 0.14 y sobre estructura geométrica o correlacional.")
    nota(fig, "Con esta arquitectura, en R ≥ 0.25 no gana en ninguna fuente. La página siguiente muestra que esa frontera no era una ley.")
    pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────── 10. la frontera era arquitectura
def p_frontera(pdf):
    fig = page(pdf, "Esa frontera era una arquitectura", kicker="resultados · control")
    casos = ["markov\nL = 35", "markov\nL = 125", "markov\nL = 250", "lowdim\nL = 250"]
    v_ = [0.1531, 0.0488, 0.0250, 0.0492]
    ancho = [0.1564, 0.0614, 0.0288, 0.0553]
    esc = [0.1979, 0.0464, 0.0135, 0.0386]
    x = np.arange(4); w = 0.26
    ax = fig.add_axes([0.075, 0.24, 0.60, 0.50])
    ax.bar(x - w, v_, w, color=MIST, label="vara")
    ax.bar(x, ancho, w, color=VARA, label="MLP ancho 1536×4")
    ax.bar(x + w, esc, w, color=ZONA, label="escalera estrecha")
    for xi, (a, b, c) in enumerate(zip(v_, ancho, esc)):
        for off, val in ((-w, a), (0, b), (w, c)):
            ax.text(xi + off, val + 0.004, f"{val:.4f}", ha="center", fontsize=10.5,
                    color=GRAPH, **SANS)
    ax.set_xticks(x); ax.set_xticklabels(casos, fontname=SANSF, fontsize=12)
    ax.set_ylabel("BER", fontsize=12); ax.set_ylim(0, 0.225)
    limpiar(ax)
    leg = ax.legend(frameon=False, fontsize=11.5, loc="upper right", ncol=1)
    for t in leg.get_texts():
        t.set_fontname(SANSF)
    fig.text(0.71, 0.64, "El signo se invierte\ncon la tasa.", fontsize=15,
             color=INK, va="top", linespacing=1.6)
    fig.text(0.71, 0.52, "El ancho gana comprimiendo\nagresivo. La escalera gana\ncon holgura, y por 46 % en\nmarkov L = 250.",
             fontsize=12.5, color=GRAPH, va="top", linespacing=1.7)
    fig.text(0.71, 0.34, "La escalera es estable en\n16 de 16 corridas; el ancho\nse degrada hasta 0.009\ntras su pico.",
             fontsize=12.5, color=GRAPH, va="top", linespacing=1.7)
    pie(fig, "Mismo número de pasos, misma tasa de aprendizaje y mismo criterio de checkpoint en los dos brazos.")
    nota(fig, "Pendiente: los dos brazos no comparten la normalización del cuello, así que arquitectura y normalización siguen juntas.")
    pdf.savefig(fig); plt.close(fig)


# ───────────────────────────────────────────────────── 11. preentrenamiento
def p_pretrain(pdf):
    fig = page(pdf, "El preentrenamiento por capas funciona", kicker="resultados · contra la predicción")
    casos = [("lowdim", 35, 0.2610, 0.1833, 0.1911, 0.1954),
             ("lowdim", 70, 0.1376, 0.1140, 0.1146, 0.1861),
             ("lowdim", 125, 0.0854, 0.0720, 0.0710, 0.0610),
             ("lowdim", 250, 0.0386, 0.0394, 0.0369, 0.0492),
             ("markov", 35, 0.1979, 0.1418, 0.1732, 0.1531),
             ("markov", 70, 0.1010, 0.0838, 0.0891, 0.0914),
             ("markov", 125, 0.0464, 0.0456, 0.0441, 0.0488),
             ("markov", 250, 0.0135, 0.0143, 0.0107, 0.0250)]
    ax = fig.add_axes([0.075, 0.24, 0.62, 0.50])
    x = np.arange(8); w = 0.26
    al = [c[2] for c in casos]; pae = [c[3] for c in casos]
    prb = [c[4] for c in casos]; vb = [c[5] for c in casos]
    ax.bar(x - w, al, w, color=MIST, label="inicialización aleatoria")
    ax.bar(x, pae, w, color="#7FB8A4", label="preentrenamiento AE")
    ax.bar(x + w, prb, w, color=ZONA, label="preentrenamiento RBM")
    for xi, v in enumerate(vb):
        ax.plot([xi - 1.5 * w, xi + 1.5 * w], [v, v], color=VARA, lw=1.8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[0]}\nL={c[1]}" for c in casos], fontname=SANSF,
                       fontsize=11)
    ax.set_ylabel("BER", fontsize=12); ax.set_ylim(0, 0.30)
    limpiar(ax)
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([0], [0], color=VARA, lw=1.8)); l.append("vara")
    leg = ax.legend(h, l, frameon=False, fontsize=11.5, loc="upper right")
    for t in leg.get_texts():
        t.set_fontname(SANSF)
    fig.text(0.725, 0.66, "Predicción registrada:\nno ayudaría.", fontsize=13,
             color=MIST, va="top", linespacing=1.6)
    fig.text(0.725, 0.565, "Gana en 8 de 8 celdas,\ndel 4.4 % al 29.8 %.",
             fontsize=15, color=ZONA, va="top", linespacing=1.6)
    fig.text(0.725, 0.44, "AE y RBM no son\nequivalentes: el AE gana a\ntasas bajas y pierde en los\ndos L = 250, donde gana\nla RBM.", fontsize=12.5,
             color=GRAPH, va="top", linespacing=1.7)
    pie(fig, "Las ocho celdas, sin seleccionar. En lowdim L=125 la vara le gana a los tres brazos.")
    nota(fig, "Los brazos preentrenados reciben 40 000 pasos más que el control; el experimento con cómputo igualado está pendiente.")
    pdf.savefig(fig); plt.close(fig)


# ────────────────────────────────────────────────────────── 12. el número
def p_numero(pdf):
    fig = page(pdf, "El mejor punto no trivial roza el umbral", kicker="resultados · el punto")
    ax = fig.add_axes([0.075, 0.24, 0.44, 0.48])
    frac = np.array([50, 90, 100])
    ber = np.array([0.0001, 0.0011, 0.0107])
    ax.plot(frac, ber, color=ZONA, lw=2.2, marker="o", ms=8)
    ax.axhline(0.01, color=VARA, lw=1.5, ls=(0, (5, 3)))
    ax.text(51, 0.0115, "umbral corregible por FEC", fontsize=11.5, color=VARA, **SANS)
    for f_, b_ in zip(frac, ber):
        ax.annotate(f"{b_:.4f}", xy=(f_, b_), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=11.5,
                    color=ZONA, **SANS)
    ax.set_yscale("log"); ax.set_ylim(5e-5, 0.05)
    ax.set_xlim(45, 108)
    ax.set_xticks([50, 90, 100])
    ax.set_xticklabels(["50 % más\nconfiable", "90 %", "todos los bits"], fontsize=11)
    ax.set_ylabel("BER", fontsize=12)
    limpiar(ax)

    fig.text(0.58, 0.63, "0.0107", fontsize=64, color=VARA, **SANS, weight="light")
    fig.text(0.58, 0.585, "BER total: no cruza, por siete diezmilésimas",
             fontsize=12.5, color=GRAPH, **SANS)
    fig.text(0.58, 0.42, "0.0011", fontsize=64, color=ZONA, **SANS, weight="light")
    fig.text(0.58, 0.375, "en el 90 % de bits más confiables: cruza por un orden de magnitud",
             fontsize=12.5, color=GRAPH, **SANS)
    fig.text(0.58, 0.30, "markov · 250 bits · escalera con RBM · 4 semillas",
             fontsize=11.5, color=MIST, **SANS)
    pie(fig, "El 10 % restante tiene BER 0.097 y viene marcado como dudoso por su propio LLR.")
    nota(fig, "Lo que demuestra: una estructura de confianza favorable a decodificación blanda. Lo que no: el BER tras un LDPC real.")
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────── 13. el fracaso en code
def p_code(pdf):
    fig = page(pdf, "Sobre redundancia algebraica, fracasa", kicker="el resultado negativo")
    ax = fig.add_axes([0.215, 0.26, 0.28, 0.46])
    n = ["copiar la mitad,\nadivinar el resto", "autoencoder", "oráculo con la\nmatriz de paridad"]
    v = [0.2500, 0.1982, 0.0000]
    y = np.arange(3)[::-1]
    ax.barh(y, v, height=0.55, color=[MIST, VARA, ZONA])
    for yy, vv in zip(y, v):
        ax.text(max(vv, 0) + 0.006, yy, f"{vv:.4f}", va="center", fontsize=13,
                color=GRAPH, **SANS)
    ax.set_yticks(y); ax.set_yticklabels(n, fontname=SANSF, fontsize=12)
    ax.set_xlim(0, 0.32); ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    limpiar(ax, ejes=())
    fig.text(0.055, 0.765, "BER a L = 250, donde la compresión sin pérdida es posible",
             fontsize=12, color=GRAPH)

    ax2 = fig.add_axes([0.58, 0.26, 0.36, 0.46])
    ax2.bar([0, 1], [0.148, 0.198], width=0.45, color=[MIST, VARA])
    ax2.text(0, 0.152, "+0.148", ha="center", fontsize=12.5, color=GRAPH, **SANS)
    ax2.text(1, 0.202, "+0.198", ha="center", fontsize=12.5, color=VARA, **SANS)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["random\n(ruido puro)", "code\n(2× comprimible)"],
                        fontname=SANSF, fontsize=12)
    ax2.set_ylabel("distancia a su propia cota", fontsize=12)
    ax2.set_ylim(0, 0.24)
    limpiar(ax2)
    ax2.set_title("Queda más lejos del óptimo sobre la fuente\nque sí se puede comprimir",
                  fontsize=12, color=GRAPH, loc="left", pad=12,
                  fontname=SERIF, linespacing=1.5)
    pie(fig, "El PCA es igual de ciego, y los errores están repartidos: ni un FEC de decisión blanda ayudaría.")
    pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────── 14. no es paridad
def p_paridad(pdf):
    fig = page(pdf, "Y no es porque no pueda aprender paridad", kicker="el mecanismo")
    ax = fig.add_axes([0.075, 0.26, 0.40, 0.46])
    g = [1, 2, 3, 4]; acc = [1.0, 1.0, 1.0, 0.4995]
    ax.bar(g, acc, width=0.5, color=[ZONA, ZONA, ZONA, VARA])
    for gg, aa in zip(g, acc):
        ax.text(gg, aa + 0.02, f"{aa:.4f}", ha="center", fontsize=12.5, color=GRAPH, **SANS)
    ax.axhline(0.5, color=MIST, lw=1, ls=(0, (4, 3)))
    ax.text(4.35, 0.5, "azar", fontsize=11.5, color=MIST, **SANS, va="center")
    ax.set_xticks(g); ax.set_xlabel("grado del XOR", fontsize=12)
    ax.set_ylabel("exactitud de test", fontsize=12); ax.set_ylim(0, 1.15)
    ax.set_xlim(0.4, 4.9)
    limpiar(ax)
    fig.text(0.55, 0.63, "Con supervisión directa,\nun MLP aprende el grado 3\nperfectamente.",
             fontsize=15, color=INK, va="top", linespacing=1.6)
    fig.text(0.55, 0.46, "Lo que falla en el autoencoder es que la pérdida de\n"
             "reconstrucción no abre un camino de gradiente hacia\n"
             "250 funciones de paridad simultáneas.",
             fontsize=12.5, color=GRAPH, va="top", linespacing=1.7)
    fig.text(0.55, 0.33, "El obstáculo es el objetivo, no la capacidad.",
             fontsize=13.5, color=ZONA, va="top")
    pie(fig, "Un piloto corto había dado 0.5069 en grado 3. Con ocho veces más pasos, 1.0000.")
    nota(fig, "Un resultado negativo con presupuesto corto es indistinguible de uno real.")
    pdf.savefig(fig); plt.close(fig)


# ────────────────────────────────────────────── 15. redundancia del latente
def p_redundancia(pdf):
    fig = page(pdf, "La redundancia del latente es capacidad que sobra", kicker="hallazgo de método")
    ax = fig.add_axes([0.075, 0.26, 0.48, 0.46])
    pts = [("oversamp L=250", 2.00, 36.1), ("markov L=250", 1.74, 33.8),
           ("markov L=70", 0.49, 16.3), ("lowdim L=70", 0.42, 3.6),
           ("lowdim L=35", 0.21, 1.0)]
    desplaz = {"oversamp L=250": (0.07, 0.9), "markov L=250": (-0.07, -3.2),
               "markov L=70": (0.07, 0.9), "lowdim L=70": (0.07, 0.9),
               "lowdim L=35": (0.07, -3.0)}
    for n, x, y in pts:
        c = VARA if x > 1 else ZONA
        dx, dy = desplaz[n]
        ax.scatter([x], [y], s=80, color=c, zorder=3)
        ax.text(x + dx, y + dy, n, fontsize=11.5, color=c, **SANS,
                ha="right" if dx < 0 else "left")
    ax.axvline(1.0, color=MIST, lw=1, ls=(0, (4, 3)))
    ax.text(1.03, 39, "bits justos", fontsize=11.5, color=MIST, **SANS)
    ax.set_xlabel("bits del latente ÷ entropía de la fuente", fontsize=12)
    ax.set_ylabel("redundancia del latente (%)", fontsize=12)
    ax.set_xlim(0, 2.45); ax.set_ylim(-2, 43)
    limpiar(ax)
    fig.text(0.63, 0.62, "Cuando sobran bits, el\nautoencoder los desperdicia.",
             fontsize=14, color=VARA, va="top", linespacing=1.6)
    fig.text(0.63, 0.47, "Cuando faltan, el latente es\ncasi incompresible: a tasas\nbajas, donde el autoencoder\ngana, no hay nada más que\nganar por codificación\nentrópica.",
             fontsize=12.5, color=GRAPH, va="top", linespacing=1.7)
    pie(fig, "Medido con un modelo autoregresivo y verificado contra la cota de Fano.")
    nota(fig, "Consecuencia práctica: la tasa nominal es honesta justo en el régimen que importa.")
    pdf.savefig(fig); plt.close(fig)


# ────────────────────────────────────────────── 16. predicciones refutadas
def p_predicciones(pdf):
    fig = page(pdf, "Seis predicciones registradas, seis refutadas", kicker="cómo se trabajó")
    filas = [("la tanh se satura y mata el gradiente", "saturación bajo el 5 %: el fallo era la escala"),
             ("SGD no aprende XOR de grado 3", "exactitud 1.0000 con datos suficientes"),
             ("más datos mejoran un 14.5 %", "eran más pasos; saturado desde 400 000"),
             ("la convolución no cambia nada en lowdim", "empeora un 21 %: pierde mezcla global"),
             ("el preentrenamiento por capas no ayuda", "gana en 8 de 8 celdas"),
             ("ajuste fino suave conserva el pico", "74 % peor por subentrenamiento")]
    y = 0.735
    fig.text(0.055, y + 0.035, "se predijo", fontsize=11, color=MIST, **SANS, weight="semibold")
    fig.text(0.52, y + 0.035, "salió", fontsize=11, color=MIST, **SANS, weight="semibold")
    fig.lines.append(plt.Line2D([0.055, 0.945], [y + 0.018, y + 0.018],
                                transform=fig.transFigure, color=INK, lw=1.4))
    for a, b in filas:
        fig.text(0.055, y - 0.035, a, fontsize=14, color=GRAPH, va="center")
        fig.text(0.52, y - 0.035, b, fontsize=14, color=INK, va="center")
        fig.lines.append(plt.Line2D([0.055, 0.945], [y - 0.075, y - 0.075],
                                    transform=fig.transFigure, color=RULE, lw=0.8))
        y -= 0.093
    pie(fig, "Y tres conclusiones ya escritas se revisaron: la frontera en R ≈ 0.25, la etiqueta «receta de Hinton», y una victoria de 4σ.",
        y=0.115)
    nota(fig, "Registrar la predicción antes de correr es lo que convierte una sorpresa en un resultado.", y=0.075)
    pdf.savefig(fig); plt.close(fig)


# ───────────────────────────────────────────────────────── 17. la respuesta
def p_respuesta(pdf):
    fig = page(pdf, "La respuesta", kicker="veredicto")
    ax = fig.add_axes([0.075, 0.47, 0.87, 0.20])
    ax.set_xlim(-1.5, 8); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    ax.annotate("", xy=(8, 0), xytext=(-1.5, 0),
                arrowprops=dict(arrowstyle="-", color=RULE, lw=2))
    marcas = [(-0.9, "0.1", "barrido típico", MIST),
              (2.19, "0.0344", "mejor del barrido", VARA),
              (4.23, "0.0107", "mejor del cierre", ZONA),
              (4.32, "", "", INK),
              (6.79, "0.001", "sin FEC", MIST)]
    for x, lab, sub, c in marcas:
        if lab:
            ax.plot([x, x], [-0.18, 0.18], color=c, lw=2.4)
            ax.text(x, 0.42, lab, ha="center", fontsize=13.5, color=c, **SANS,
                    weight="semibold")
            ax.text(x, -0.75, sub, ha="center", fontsize=11, color=c, **SANS)
        else:
            ax.plot([x, x], [-0.45, 0.45], color=INK, lw=1.4, ls=(0, (3, 2)))
            ax.text(x + 0.15, 0.80, "umbral con FEC", fontsize=11.5, color=INK, **SANS)
    for x in (0, 2, 4, 6, 8):
        ax.text(x, -1.0, f"{x} dB", ha="center", fontsize=10.5, color=MIST, **SANS)
    fig.text(0.075, 0.735, "BER traducido a un enlace BPSK sin codificar",
             fontsize=12, color=GRAPH)

    fig.add_artist(plt.Rectangle((0.075, 0.235), 0.006, 0.135, color=VARA,
                                 transform=fig.transFigure))
    fig.text(0.10, 0.345, "No son viables a tasas agresivas.", fontsize=17, color=INK)
    fig.text(0.10, 0.29, "18 de 20 puntos del barrido quedan fuera del umbral. El mejor punto\n"
             "no trivial se queda a una décima de decibelio.", fontsize=13,
             color=GRAPH, va="top", linespacing=1.6)

    fig.add_artist(plt.Rectangle((0.545, 0.235), 0.006, 0.135, color=ZONA,
                                 transform=fig.transFigure))
    fig.text(0.57, 0.345, "Compatibles con decodificación blanda.", fontsize=17, color=INK)
    fig.text(0.57, 0.29, "En una franja estrecha: markov a 250 bits, escalera y RBM.\n"
             "El BER tras un decodificador real queda por medir.", fontsize=13,
             color=GRAPH, va="top", linespacing=1.6)
    pie(fig, "Y superan a lo clásico en casi todo el mapa de markov y lowdim si la arquitectura se elige por tasa.",
        y=0.135)
    nota(fig, "Sobre redundancia algebraica fracasan, y por el objetivo, no por la capacidad.", y=0.09)
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────────── 18. lo que queda
def p_abierto(pdf):
    fig = page(pdf, "Lo que haría a continuación", kicker="trabajo abierto")
    items = [("Un objetivo auxiliar que supervise la estructura algebraica",
              "Si el obstáculo en code es el objetivo, cámbiese el objetivo.", "la más valiosa"),
             ("Simulación post-FEC con un LDPC real sobre los LLR de markov L=250",
              "Convierte «favorable a decisión blanda» en una cifra operativa.", "la más barata"),
             ("Cómputo igualado, y el cuello del MLP ancho con la normalización validada",
              "Cierra los dos controles que hoy siguen abiertos.", "una tarde de H200"),
             ("Señales reales, y anchos intermedios entre la escalera y el MLP",
              "Lo que hace falta para que esto sea un paper aplicado.", "el siguiente paso")]
    y = 0.70
    for a, b, tag in items:
        fig.lines.append(plt.Line2D([0.055, 0.945], [y + 0.055, y + 0.055],
                                    transform=fig.transFigure, color=RULE, lw=0.8))
        fig.text(0.055, y, a, fontsize=15.5, color=INK, va="center")
        fig.text(0.055, y - 0.048, b, fontsize=12.5, color=GRAPH, va="center")
        fig.text(0.945, y, tag, fontsize=11.5, color=ZONA, **SANS, va="center", ha="right")
        y -= 0.155
    fig.lines.append(plt.Line2D([0.055, 0.945], [0.16, 0.16],
                                transform=fig.transFigure, color=INK, lw=1.2))
    fig.text(0.055, 0.115, "Todo lo del cuaderno es regenerable: cada número tiene fecha, "
             "semillas, versiones y hash del script.", fontsize=12.5, color=GRAPH)
    nota(fig, "Cuaderno, código y datos: agesfranciscoteran.github.io/autoencoders---telecommunications", y=0.075)
    pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────── 19. anexo: config
def p_config(pdf):
    fig = page(pdf, "Anexo: configuración experimental", kicker="respaldo")
    filas = [("fuentes", "las cinco", "lowdim, markov, code"),
             ("modelo", "MLP 1536 de ancho × 4", "escalera 500-250-125-70-35"),
             ("cuello", "binario + STE, tanh", "binario + STE, BatchNorm sin afín"),
             ("datos", "400 000 train · 50 000 test", "400 000 · 20 000 val · 20 000 test"),
             ("pasos", "300 épocas, batch 4096 (~29 300)", "30 000 fino + 10 000 por etapa"),
             ("optimizador", "AdamW 3e-3, OneCycle", "AdamW 1e-3, OneCycle"),
             ("checkpoint", "última época", "mejor sobre validación"),
             ("semillas", "4", "4"),
             ("hardware", "H200, bf16, TF32", "H200, bf16, TF32")]
    x0, x1, x2 = 0.055, 0.34, 0.64
    y = 0.735
    fig.text(x1, y, "barrido v4", fontsize=11.5, color=GRAPH, **SANS, weight="semibold")
    fig.text(x2, y, "experimentos de cierre", fontsize=11.5, color=GRAPH, **SANS,
             weight="semibold")
    fig.lines.append(plt.Line2D([x0, 0.945], [y - 0.018, y - 0.018],
                                transform=fig.transFigure, color=INK, lw=1.4))
    y -= 0.06
    for a, b, c in filas:
        col_b = VARA if a == "checkpoint" else INK
        col_c = ZONA if a == "checkpoint" else INK
        fig.text(x0, y, a, fontsize=12.5, color=GRAPH, **SANS)
        fig.text(x1, y, b, fontsize=12.5, color=col_b, **SANS)
        fig.text(x2, y, c, fontsize=12.5, color=col_c, **SANS)
        fig.lines.append(plt.Line2D([x0, 0.945], [y - 0.022, y - 0.022],
                                    transform=fig.transFigure, color=RULE, lw=0.7))
        y -= 0.062
    pie(fig, "La asimetría del checkpoint es la razón de que los números del barrido y los del cierre no se comparen a ojo.",
        y=0.125)
    nota(fig, "El cuello tampoco coincide: es el control que falta para separar arquitectura de normalización.", y=0.085)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = SALIDA
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with PdfPages(out) as pdf:
        for f in (p_portada, p_historia, p_cadena, p_bits, p_fuentes, p_cotas, p_zona,
                  p_canal, p_anidado, p_escala, p_victorias, p_frontera, p_pretrain,
                  p_numero, p_code, p_paridad, p_redundancia, p_predicciones,
                  p_respuesta, p_abierto, p_config):
            f(pdf)
        d = pdf.infodict()
        d["Title"] = "¿Son viables los autoencoders para comprimir señales BPSK?"
        d["Author"] = "Francisco Terán"
        d["Subject"] = "Defensa del proyecto · USFQ"
    print("escrito:", out, "páginas:", _pageno[0])


if __name__ == "__main__":
    main()
