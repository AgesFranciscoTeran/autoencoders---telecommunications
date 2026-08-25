#!/usr/bin/env bash
# Lanzador para DGX-H200. Reanudable: si se cae, re-ejecuta y salta lo hecho.
set -euo pipefail
OUT="${1:-./ae_results_v4}"
GPUS="${2:-0,1}"                       # logicas dentro del contenedor
IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
mkdir -p "$OUT"

echo "[*] smoke test antes de gastar GPU..."
python3 ae_telecom_v4.py --mode smoke --device cpu --outdir "$OUT/_smoke"

echo "[*] lanzando $N shards en GPUs: $GPUS"
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES="${G[$i]}" nohup python3 -u ae_telecom_v4.py \
      --mode full --device cuda --outdir "$OUT" \
      --shard "$i" --num-shards "$N" > "$OUT/shard_$i.log" 2>&1 &
  echo "    shard $i -> GPU ${G[$i]} (PID $!)"
done
wait
echo "[*] consolidando..."
python3 ae_telecom_v4.py --report --outdir "$OUT"
