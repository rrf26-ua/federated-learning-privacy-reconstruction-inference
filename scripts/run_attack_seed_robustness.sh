#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

SAMPLES=(0 1 2 3 4)
SEEDS=(0 1 2 3 4)

run_config() {
  local config_name="$1"
  shift

  for sample in "${SAMPLES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="results/reconstructions/seed_robustness/${config_name}/sample${sample}_seed${seed}"

      if [ -f "$out/attack_metrics.json" ]; then
        echo "[SKIP] config=$config_name sample=$sample seed=$seed"
        continue
      fi

      echo "===== seed robustness: config=$config_name sample=$sample seed=$seed ====="

      python -m attacks.local_update_inversion_attack \
        --sample-start "$sample" \
        --batch-size 1 \
        --split train \
        --iterations 2000 \
        --grad-scope all \
        --matching-loss cosine \
        --local-lr 0.02 \
        --weight-decay 0.0005 \
        --attack-lr 0.03 \
        --tv-weight 0.000001 \
        --seed "$seed" \
        --save-every 1000 \
        --log-every 500 \
        --output-dir "$out" \
        "$@"
    done
  done
}

run_config none \
  --observed-defense none

run_config noise_s0002 \
  --observed-defense noise \
  --observed-noise-std 0.002 \
  --observed-noise-seed 42

run_config clip_c05_noise_s0002 \
  --observed-defense clipping_noise \
  --observed-clip-norm 0.5 \
  --observed-noise-std 0.002 \
  --observed-noise-seed 42
