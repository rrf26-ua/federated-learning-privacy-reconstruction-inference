#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

run_config() {
  local config_name="$1"
  shift

  for i in 0 1 2 3 4 5 6 7 8 9; do
    local out="results/reconstructions/defense_ablation_cosine_10samples/${config_name}/sample${i}"

    if [ -f "$out/attack_metrics.json" ]; then
      echo "[SKIP] config=$config_name sample=$i already exists"
      continue
    fi

    echo "===== defense ablation: config=$config_name sample=$i ====="

    python -m attacks.local_update_inversion_attack \
      --sample-start "$i" \
      --batch-size 1 \
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
      --output-dir "$out" \
      "$@"
  done
}

run_config none \
  --observed-defense none

run_config clip_c05 \
  --observed-defense clipping \
  --observed-clip-norm 0.5

run_config clip_c10 \
  --observed-defense clipping \
  --observed-clip-norm 1.0

run_config noise_s0001 \
  --observed-defense noise \
  --observed-noise-std 0.001 \
  --observed-noise-seed 42

run_config noise_s0002 \
  --observed-defense noise \
  --observed-noise-std 0.002 \
  --observed-noise-seed 42

run_config noise_s0005 \
  --observed-defense noise \
  --observed-noise-std 0.005 \
  --observed-noise-seed 42

run_config clip_c10_noise_s0002 \
  --observed-defense clipping_noise \
  --observed-clip-norm 1.0 \
  --observed-noise-std 0.002 \
  --observed-noise-seed 42

run_config clip_c05_noise_s0002 \
  --observed-defense clipping_noise \
  --observed-clip-norm 0.5 \
  --observed-noise-std 0.002 \
  --observed-noise-seed 42
