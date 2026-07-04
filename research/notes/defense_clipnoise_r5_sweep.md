# Clipping plus Gaussian noise sweep — 5 rounds

## Context

A short 5-round sweep was executed to evaluate Gaussian noise after strong client update clipping.

The base defense is clipping with C=7.5, using the main FedAvg E=2 baseline configuration.

## Results

| Clip norm | Noise std | Final accuracy R5 | Final loss R5 | Interpretation |
|---:|---:|---:|---:|---|
| 7.5 | 0.0005 | 0.6161 | 1.0733 | Soft noise |
| 7.5 | 0.001 | 0.6287 | 1.0407 | Moderate noise candidate |
| 7.5 | 0.002 | 0.6271 | 1.0309 | Stronger noise candidate |

## Interpretation

Adding small Gaussian noise after clipping does not collapse training in the short sweep. Noise values 0.001 and 0.002 preserve utility similarly to clipping-only C=7.5 and are selected for full 20-round experiments.

## Decision

Run full experiments with:

- C=7.5, noise_std=0.001
- C=7.5, noise_std=0.002
