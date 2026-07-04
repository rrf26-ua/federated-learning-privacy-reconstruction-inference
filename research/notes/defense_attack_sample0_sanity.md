# Reconstruction attack under defenses — sample 0 sanity check

## Context

A controlled local update inversion attack was executed on CIFAR-10 sample 0 using cosine matching, batch size 1, 2000 iterations, ResNet-18 and local learning rate 0.03.

The compared observed-update defenses were:

- No defense.
- Clipping with C=7.5.
- Clipping with C=7.5 plus Gaussian noise with noise_std=0.002.

## Results

| Defense | Raw norm | Defended norm | Clip scale | Best oracle PSNR | Label inference |
|---|---:|---:|---:|---:|---|
| None | 1.218655 | 1.218655 | 1.0000 | 35.7765 | airplane |
| Clipping C=7.5 | 1.218655 | 1.218655 | 1.0000 | 36.1270 | airplane |
| C=7.5 + noise 0.002 | 1.218655 | 6.794307 | 1.0000 | 16.7802 | airplane |

## Interpretation

The clipping threshold C=7.5 does not activate in this controlled single-sample update attack because the raw update norm is only 1.218655. Therefore, clipping C=7.5 leaves the observed update unchanged and reconstruction quality remains essentially identical to the no-defense case.

Gaussian noise has a strong effect on reconstruction quality, reducing the best oracle PSNR from approximately 35.78 dB to 16.78 dB. However, label inference still succeeds in this sample.

## Consequence

For this controlled single-sample attack, C=7.5 is not an effective clipping threshold because it was selected from full federated training update norms, not from single-sample update norms.

The next step is to run an additional controlled clipping experiment with thresholds relative to the observed single-sample update norm, or to move to a more FedAvg-like multi-step update attack where update norms are larger.
