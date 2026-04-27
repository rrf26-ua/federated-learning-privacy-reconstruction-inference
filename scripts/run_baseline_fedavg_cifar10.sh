#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-baseline_r20_lr001_le1_bs32_seed42}"
NUM_ROUNDS="${NUM_ROUNDS:-20}"
LR="${LR:-0.01}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
FRACTION_EVAL="${FRACTION_EVAL:-1.0}"
FED_CFG="${FED_CFG:-num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2}"

mkdir -p results/raw

flwr run . --stream \
  --run-config "num-server-rounds=${NUM_ROUNDS} learning-rate=${LR} local-epochs=${LOCAL_EPOCHS} batch-size=${BATCH_SIZE} seed=${SEED} fraction-evaluate=${FRACTION_EVAL}" \
  --federation-config "${FED_CFG}" | tee "results/raw/${EXP_NAME}.log"
