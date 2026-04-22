# baseline_fedavg_cifar10_local

## Purpose
Stable local baseline for the thesis repository.

## Framework
- Flower
- PyTorch

## Dataset
- CIFAR-10

## Strategy
- FedAvg

## Stable simulation parameters
- num-server-rounds = 3
- num-supernodes = 4
- client-resources-num-cpus = 2
- init-args-num-cpus = 2

## Execution command
```bash
./scripts/run_baseline_fedavg_cifar10.sh
Notes

This configuration was selected because the default Flower quickstart settings caused out-of-memory failures on the available machine.
