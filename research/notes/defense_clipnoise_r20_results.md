# Clipping plus Gaussian noise — 20-round defense results

## Context

Full 20-round experiments were executed using the main FedAvg E=2 baseline configuration:

- 6 clients
- 20 rounds
- 2 local epochs
- learning rate 0.03
- batch size 32
- centralized CIFAR-10 evaluation

The non-defended baseline reached 83.96% final accuracy. The clipping-only strong defense C=7.5 reached 83.59%.

## Results

| Defense | Final accuracy | Final loss | Final clipping scale | Interpretation |
|---|---:|---:|---:|---|
| C=7.5 + noise 0.001 | 0.8174 | 0.6189 | 0.5551 | Moderate noise, larger utility cost |
| C=7.5 + noise 0.002 | 0.8300 | 0.5823 | 0.5430 | Stronger noise selected |

## Interpretation

Adding Gaussian noise after clipping introduces a measurable utility cost compared with clipping-only. However, the configuration C=7.5 with noise_std=0.002 still preserves high utility, reaching 83.00% final accuracy.

The fact that noise_std=0.002 outperforms noise_std=0.001 should not be overinterpreted as a general improvement caused by stronger noise. It may be due to stochastic variability. The relevant conclusion is that clipping plus noise remains viable under this configuration.

## Experimental decision

For privacy-oriented reconstruction experiments, the selected defense configurations are:

- No defense.
- Clipping C=7.5.
- Clipping C=7.5 plus Gaussian noise with noise_std=0.002.

The noise_std=0.001 run is kept as a secondary comparison.
