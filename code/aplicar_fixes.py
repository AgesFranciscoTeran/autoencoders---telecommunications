#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica las 3 correcciones a ae2.py y check_v4.py.  Hace backup .bak primero.
    python3 aplicar_fixes.py
"""
import os, shutil, sys

FIXES_AE2 = [
    # ---- FIX 2: solo el shard 0 escribe baselines (evita duplicados) ----
    ("FIX 2  baselines duplicados",
     "and args.shard == 0",
     '        bk = job_key({"bl": job["regime"]})\n'
     '        if bk not in bl_done:\n',
     '        bk = job_key({"bl": job["regime"]})\n'
     '        if bk not in bl_done and args.shard == 0:\n'),

    # ---- FIX 3a: NestedAE, LayerNorm en vez de tanh ----
    ("FIX 3a NestedAE.__init__",
     "self.enc_norm = nn.LayerNorm(self.Lmax)",
     '        self.enc = mlp(enc_sizes)\n'
     '        self.dec = mlp(dec_sizes)\n',
     '        self.enc = mlp(enc_sizes)\n'
     '        self.enc_norm = nn.LayerNorm(self.Lmax)\n'
     '        self.dec = mlp(dec_sizes)\n'),

    ("FIX 3b NestedAE.encode",
     "        return BinarizeSTE.apply(self.enc_norm(self.enc(x)))",
     '    def encode(self, x):\n'
     '        return BinarizeSTE.apply(torch.tanh(self.enc(x)))\n',
     '    def encode(self, x):\n'
     '        return BinarizeSTE.apply(self.enc_norm(self.enc(x)))\n'),

    # ---- FIX 3c: PlainAE, LayerNorm en vez de tanh ----
    ("FIX 3c PlainAE.__init__",
     "self.enc_norm = nn.LayerNorm(latent)",
     '        self.enc = mlp([dim_in] + [hidden] * depth + [latent])\n'
     '        self.dec = mlp([latent] + [hidden] * depth + [dim_out])\n',
     '        self.enc = mlp([dim_in] + [hidden] * depth + [latent])\n'
     '        self.enc_norm = nn.LayerNorm(latent)\n'
     '        self.dec = mlp([latent] + [hidden] * depth + [dim_out])\n'),

    ("FIX 3d PlainAE.encode",
     "def encode(self, x): return BinarizeSTE.apply(self.enc_norm",
     '    def encode(self, x): return BinarizeSTE.apply(torch.tanh(self.enc(x)))\n',
     '    def encode(self, x): return BinarizeSTE.apply(self.enc_norm(self.enc(x)))\n'),
]

FIXES_CHK = [
    # ---- FIX 1: deduplicar baselines antes del merge ----
    ("FIX 1  merge duplicado",
     "bl = bl.drop_duplicates",
     '    m = res.merge(bl[["regime", "latent_bits", "baseline_ber", "baseline_metodo"]],\n',
     '    bl = bl.drop_duplicates(["regime", "latent_bits"])\n'
     '    m = res.merge(bl[["regime", "latent_bits", "baseline_ber", "baseline_metodo"]],\n'),

    # ---- FIX 1b: mediana en vez de media (robusta a corridas colapsadas) ----
    ("FIX 1b costo por mediana",
     'piv["costo"].median()',
     '        cm = piv["costo"].mean()\n'
     '        print(f"\\n  Costo medio de anidar: {cm:+.4f} BER")\n',
     '        cm = piv["costo"].median()\n'
     '        print(f"\\n  Costo MEDIANO de anidar: {cm:+.4f} BER "\n'
     '              f"(media {piv[\'costo\'].mean():+.4f}, sensible a colapsos)")\n'),
]


def aplica(path, fixes):
    if not os.path.exists(path):
        print(f"  [!] no existe {path}, lo salto"); return 0
    src = open(path, encoding="utf-8").read()
    shutil.copy(path, path + ".bak")
    n = 0
    for nombre, marcador, viejo, nuevo in fixes:
        if marcador in src:
            print(f"  [ya] {nombre}"); continue
        c = src.count(viejo)
        if c == 0:
            print(f"  [!!] {nombre}: NO encontrado -- aplicalo a mano")
        elif c > 1:
            print(f"  [!!] {nombre}: aparece {c} veces -- aplicalo a mano")
        else:
            src = src.replace(viejo, nuevo); n += 1
            print(f"  [ok] {nombre}")
    open(path, "w", encoding="utf-8").write(src)
    return n


def main():
    print("Aplicando correcciones (backup en *.bak)\n")
    print("ae2.py:");      a = aplica("ae2.py", FIXES_AE2)
    print("\ncheck_v4.py:"); b = aplica("check_v4.py", FIXES_CHK)
    print(f"\n{a+b} cambios aplicados.")
    import ast
    for p in ("ae2.py", "check_v4.py"):
        if os.path.exists(p):
            try:
                ast.parse(open(p, encoding="utf-8").read())
                print(f"  {p}: sintaxis OK")
            except SyntaxError as e:
                print(f"  {p}: SINTAXIS ROTA linea {e.lineno} -- restaura con "
                      f"'mv {p}.bak {p}'")
                sys.exit(1)


if __name__ == "__main__":
    main()
