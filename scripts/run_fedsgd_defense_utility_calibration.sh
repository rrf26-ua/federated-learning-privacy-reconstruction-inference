#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
mkdir -p results/raw

ROUNDS=50
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

  echo "===== FedSGD defense utility calibration: $config_id ====="

  flwr run . --stream \
    --run-config "fl-algorithm=\"fedsgd\" fl-strategy=\"fedavg\" num-server-rounds=${ROUNDS} seed=${SEED} learning-rate=${LR} weight-decay=${WEIGHT_DECAY} scheduler=\"${SCHEDULER}\" scheduler-min-lr=${SCHEDULER_MIN_LR} seed-with-server-round=true batch-size=${BATCH_SIZE} fraction-train=1.0 fraction-evaluate=0.0 defense-type=\"${defense_type}\" defense-clip-norm=${clip_norm} defense-noise-std=${noise_std} save-final-model=false" \
    --federation-config "num-supernodes=${CLIENTS} client-resources-num-cpus=4 client-resources-num-gpus=0.5 init-args-num-cpus=8 init-args-num-gpus=1" \
    2>&1 | tee "$log"
}

run_config fedsgd_defutil_r50_none none 0.0 0.0

run_config fedsgd_defutil_r50_clip_c005 clipping 0.05 0.0
run_config fedsgd_defutil_r50_clip_c001 clipping 0.01 0.0

run_config fedsgd_defutil_r50_noise_s000001 noise 0.0 0.00001
run_config fedsgd_defutil_r50_noise_s000005 noise 0.0 0.00005

run_config fedsgd_defutil_r50_clip_c005_noise_s000001 clipping_noise 0.05 0.00001
run_config fedsgd_defutil_r50_clip_c005_noise_s000005 clipping_noise 0.05 0.00005

# Referencia extrema: probablemente destruirá el entrenamiento, pero sirve para demostrar
# que el mismo ruido usado en FedAvg no escala directamente a FedSGD.
run_config fedsgd_defutil_r50_noise_s0002_extreme noise 0.0 0.002
