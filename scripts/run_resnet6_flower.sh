#!/usr/bin/env bash
set -euo pipefail

flwr run . --stream \
  --run-config "num-server-rounds=20 seed=42 learning-rate=0.02 local-epochs=10 batch-size=32 fraction-train=1.0 fraction-evaluate=0.0" \
  --federation-config "num-supernodes=6 client-resources-num-cpus=1 client-resources-num-gpus=0.166 init-args-num-cpus=8 init-args-num-gpus=1"
