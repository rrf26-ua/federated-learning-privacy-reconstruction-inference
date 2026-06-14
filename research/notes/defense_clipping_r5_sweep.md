# Client update clipping sweep — 5 rounds

## Context

A short 5-round sweep was executed to select meaningful clipping values before running full 20-round defense experiments.

The baseline configuration is FedAvg E=2, learning rate 0.03, batch size 32, 6 clients and centralized CIFAR-10 evaluation.

## Results

| Clip norm | Final accuracy at R5 | Interpretation |
|---:|---:|---|
| 15.0 | 0.6598 | Almost no clipping; too weak |
| 12.0 | 0.6640 | Soft clipping; only affects the first update clearly |
| 10.0 | 0.6543 | Moderate clipping; useful candidate |
| 7.5 | 0.6175 | Strong clipping; larger accuracy penalty |

## Decision

The following clipping values are selected for the full defense experiments:

- C=12.0 as soft clipping.
- C=10.0 as medium clipping.
- C=7.5 as strong clipping.

C=15.0 is discarded as a main defense setting because it barely clips the updates and is therefore unlikely to provide meaningful privacy protection.
