#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
mkdir -p results/raw

ROUNDS=400
CLIENTS=6
BATCH_SIZE=64
LR=0.05
WEIGHT_DECAY=0.0005
SCHEDULER="cosine"
SCHEDULER_MIN_LR=0.005
SEED=42

run_config() {
  local config_id="$1"
  local defense_type="$2"
  local clip_norm="$3"
  local noise_std="$4"

  local log="results/raw/${config_id}.log"

  if [ -f "$log" ]; then
    echo "[SKIP] $config_id already exists: $log"
    return
  fi

  echo "===== FedSGD defense utility 400 rounds: $config_id ====="

  flwr run . --stream \
    --run-config "fl-algorithm=\"fedsgd\" fl-strategy=\"fedavg\" num-server-rounds=${ROUNDS} seed=${SEED} learning-rate=${LR} weight-decay=${WEIGHT_DECAY} scheduler=\"${SCHEDULER}\" scheduler-min-lr=${SCHEDULER_MIN_LR} seed-with-server-round=true batch-size=${BATCH_SIZE} fraction-train=1.0 fraction-evaluate=0.0 defense-type=\"${defense_type}\" defense-clip-norm=${clip_norm} defense-noise-std=${noise_std} save-final-model=false" \
    --federation-config "num-supernodes=${CLIENTS} client-resources-num-cpus=4 client-resources-num-gpus=0.5 init-args-num-cpus=8 init-args-num-gpus=1" \
    2>&1 | tee "$log"
}

run_config fedsgd_defutil_r400_none none 0.0 0.0
run_config fedsgd_defutil_r400_clip_c005_noise_s000005 clipping_noise 0.05 0.00005
run_config fedsgd_defutil_r400_noise_s0002_extreme noise 0.0 0.002
