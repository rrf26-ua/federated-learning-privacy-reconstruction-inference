#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

run_none() {
  local i="$1"
  local out="results/reconstructions/fedavg_10samples_none/sample${i}"

  if [ -f "$out/attack_metrics.json" ]; then
    echo "[SKIP] no defense sample $i already exists"
    return
  fi

  echo "===== FedAvg no defense sample $i ====="

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
    --save-every 500 \
    --log-every 500 \
    --output-dir "$out"
}

run_clipnoise() {
  local i="$1"
  local out="results/reconstructions/fedavg_10samples_clipnoise/sample${i}"

  if [ -f "$out/attack_metrics.json" ]; then
    echo "[SKIP] clip+noise sample $i already exists"
    return
  fi

  echo "===== FedAvg clip+noise sample $i ====="

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
    --observed-defense clipping_noise \
    --observed-clip-norm 7.5 \
    --observed-noise-std 0.002 \
    --observed-noise-seed 42 \
    --save-every 500 \
    --log-every 500 \
    --output-dir "$out"
}

for i in 0 1 2 3 4 5 6 7 8 9; do
  run_none "$i"
done

for i in 0 1 2 3 4 5 6 7 8 9; do
  run_clipnoise "$i"
done
