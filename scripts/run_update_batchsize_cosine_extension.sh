#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

BATCH_SIZES=(1 2 4 8)
STARTS=(0 8 16 24 32)

for bs in "${BATCH_SIZES[@]}"; do
  for start in "${STARTS[@]}"; do
    out="results/reconstructions/update_batchsize_cosine/bs${bs}_start${start}"

    if [ -f "$out/attack_metrics.json" ]; then
      echo "[SKIP] batch_size=$bs sample_start=$start already exists"
      continue
    fi

    echo "===== update batch-size experiment: bs=$bs start=$start ====="

    python -m attacks.local_update_inversion_attack \
      --sample-start "$start" \
      --batch-size "$bs" \
      --split train \
      --iterations 2000 \
      --grad-scope all \
      --matching-loss cosine \
      --local-lr 0.02 \
      --weight-decay 0.0005 \
      --attack-lr 0.03 \
      --tv-weight 0.000001 \
      --seed 42 \
      --save-every 1000 \
      --log-every 500 \
      --output-dir "$out"
  done
done
