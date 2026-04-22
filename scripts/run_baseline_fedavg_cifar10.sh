#!/usr/bin/env bash
set -euo pipefail

flwr run . --stream \
  --run-config "num-server-rounds=3" \
  --federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2"
