# Baseline selection summary

## Objective

Select suitable federated learning baselines for the privacy and security evaluation.

## Main results

| Configuration | Rounds | Local epochs | Final / best accuracy | Decision |
|---|---:|---:|---:|---|
| Strict FedSGD | 50 | 1 global update | 30.88% / 31.28% | Contrast baseline; slow convergence |
| FedSGD + FedAvgM | smoke tests | - | poor / unstable | Discarded |
| FedSGD + FedAdam | smoke tests | - | near random | Discarded |
| FedAvg E=1, lr=0.05 | 20 | 1 | 76.04% | Intermediate baseline |
| FedAvg E=2, lr=0.03 | 20 | 2 | 83.96% | Main intermediate baseline |
| FedAvg E=10, lr=0.02 | 5 | 10 | 86.72% | High-performance baseline |

## Final decision

The main baseline for subsequent privacy and defense experiments will be FedAvg with 2 local epochs, 20 rounds, learning rate 0.03, batch size 32 and 6 clients.

Strict FedSGD is retained as a contrast scenario because it is closer to gradient-style updates and is useful for discussing privacy exposure, but it is not used as the main performance baseline.

FedAvg E=10 remains the high-performance baseline.
