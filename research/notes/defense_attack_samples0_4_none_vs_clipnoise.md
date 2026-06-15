# Reconstruction attack comparison — no defense vs clipping plus noise

## Context

A controlled local update inversion attack was evaluated on CIFAR-10 samples 0 to 4.

The attack used:

- ResNet-18 adapted to CIFAR-10
- batch size 1
- cosine matching loss
- 2000 optimization iterations
- local learning rate 0.03
- weight decay 0.0005
- all trainable parameter tensors

Two observed-update settings were compared:

1. No defense.
2. Clipping plus Gaussian noise, with C=7.5 and noise_std=0.002.

## Results

| Sample | Label | No defense PSNR | Clip+noise PSNR | Drop | Label inference |
|---:|---|---:|---:|---:|---|
| 0 | airplane | 35.7765 | 17.0550 | 18.7215 | correct in both |
| 1 | frog | 15.0569 | 10.4280 | 4.6289 | correct in both |
| 2 | airplane | 29.9152 | 14.1616 | 15.7536 | correct in both |
| 3 | bird | 29.5776 | 13.4829 | 16.0947 | correct in both |
| 4 | horse | 31.4294 | 12.4244 | 19.0050 | correct in both |

Mean best-oracle PSNR without defense: 28.3511 dB.

Mean best-oracle PSNR with clipping plus noise: 13.5104 dB.

Mean PSNR drop: 14.8408 dB.

## Interpretation

The clipping plus noise defense substantially reduces reconstruction quality. The average best-oracle PSNR decreases from 28.3511 dB to 13.5104 dB, a drop of 14.8408 dB.

This indicates that adding Gaussian noise after clipping strongly perturbs the directional information exploited by the cosine-matching reconstruction attack.

However, label inference remains successful in all tested samples. Therefore, this defense significantly reduces visual reconstruction leakage but does not eliminate label leakage in this controlled setup.

## Conclusion

The selected defense C=7.5 with noise_std=0.002 preserves high federated learning utility in the training experiments, reaching 83.00% final accuracy, while strongly degrading image reconstruction quality in the controlled attack scenario.

This provides a clear privacy-utility result:

- Utility remains high.
- Visual reconstruction leakage is substantially reduced.
- Label inference leakage persists.
