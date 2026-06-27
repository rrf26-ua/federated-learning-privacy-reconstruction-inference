#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

LOCAL_STEPS=(1 2 5 10)
SAMPLES=(0 1 2 3 4)

for steps in "${LOCAL_STEPS[@]}"; do
  for sample in "${SAMPLES[@]}"; do
    out="results/reconstructions/local_steps_cosine/steps${steps}_sample${sample}"

    if [ -f "$out/attack_metrics.json" ]; then
      echo "[SKIP] local_steps=$steps sample=$sample"
      continue
    fi

    echo "===== local steps cosine: steps=$steps sample=$sample ====="

    python -m attacks.local_update_multistep_inversion_attack \
      --sample-start "$sample" \
      --batch-size 1 \
      --split train \
      --local-steps "$steps" \
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
