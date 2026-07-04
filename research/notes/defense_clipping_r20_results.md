# Client update clipping — 20-round defense results

## Context

Full 20-round clipping experiments were executed using the main intermediate baseline:

- FedAvg
- 6 clients
- 20 rounds
- 2 local epochs
- learning rate 0.03
- batch size 32
- centralized CIFAR-10 evaluation

The non-defended baseline reached 83.96% test accuracy.

## Results

| Defense | Final accuracy | Final loss | Final clipping scale | Interpretation |
|---|---:|---:|---:|---|
| No defense | 0.8396 | 0.5095 | 1.0000 | Baseline |
| Clipping C=12.0 | 0.8374 | 0.5182 | 0.7900 | Soft clipping |
| Clipping C=10.0 | 0.8401 | 0.5162 | 0.6859 | Medium clipping |
| Clipping C=7.5 | 0.8359 | 0.5305 | 0.5705 | Strong clipping |

## Interpretation

All clipping configurations preserve model performance. Even the strongest clipping value, C=7.5, only reduces final accuracy from 83.96% to 83.59%.

This indicates that client update clipping can substantially reduce the norm of the transmitted updates while maintaining competitive model utility.

## Experimental decision

The selected clipping settings for privacy evaluation are:

- C=12.0 as soft clipping.
- C=10.0 as medium clipping.
- C=7.5 as strong clipping.

For privacy-focused experiments, C=7.5 is the most relevant configuration because it applies the strongest update reduction while preserving high accuracy.
