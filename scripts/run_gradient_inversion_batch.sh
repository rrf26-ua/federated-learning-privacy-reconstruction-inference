#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-train}"
ITERATIONS="${ITERATIONS:-2000}"
GRAD_SCOPE="${GRAD_SCOPE:-all}"
ATTACK_LR="${ATTACK_LR:-0.03}"
TV_WEIGHT="${TV_WEIGHT:-0.001}"
SEED="${SEED:-42}"
SAMPLES="${SAMPLES:-0 1 2 3 4}"

BASE_DIR="results/reconstructions/batch_${GRAD_SCOPE}_iter${ITERATIONS}_lr${ATTACK_LR}_tv${TV_WEIGHT}_seed${SEED}"
mkdir -p "$BASE_DIR"

echo "[INFO] Output base directory: $BASE_DIR"
echo "[INFO] Samples: $SAMPLES"

for SAMPLE_INDEX in $SAMPLES; do
  OUT_DIR="${BASE_DIR}/sample_${SAMPLE_INDEX}"

  echo
  echo "============================================================"
  echo "[INFO] Running attack for sample ${SAMPLE_INDEX}"
  echo "============================================================"

  python -m attacks.gradient_inversion_attack \
    --sample-index "$SAMPLE_INDEX" \
    --split "$SPLIT" \
    --iterations "$ITERATIONS" \
    --grad-scope "$GRAD_SCOPE" \
    --attack-lr "$ATTACK_LR" \
    --tv-weight "$TV_WEIGHT" \
    --save-every 500 \
    --log-every 100 \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "${OUT_DIR}_run.log"
done

echo
echo "[INFO] Batch finished."
echo "[INFO] Results stored in: $BASE_DIR"
