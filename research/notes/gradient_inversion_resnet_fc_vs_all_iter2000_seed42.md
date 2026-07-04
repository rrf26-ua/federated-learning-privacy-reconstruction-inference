# gradient_inversion_resnet_fc_vs_all_iter2000_seed42

## Purpose

Compare two gradient scopes in a controlled gradient inversion attack:

- `fc`: only the final fully connected layer gradients.
- `all`: gradients from all trainable model parameters.

The goal is to distinguish semantic leakage, especially label inference, from visual reconstruction leakage.

## Configuration

Common configuration:

- Dataset: CIFAR-10
- Split: train
- Model: ResNet-18 adapted for CIFAR-10
- Attack: gradient inversion
- Number of samples: 5
- Sample indices: 0, 1, 2, 3, 4
- Iterations per sample: 2000
- Attack learning rate: 0.03
- Total variation weight: 0.001
- Seed: 42
- Device: cuda:0

## Results: all gradients

| Sample | True label | Inferred label | MSE | PSNR |
|---:|---|---|---:|---:|
| 0 | airplane | airplane | 0.0196 | 17.08 |
| 1 | frog | frog | 0.0922 | 10.35 |
| 2 | airplane | airplane | 0.0454 | 13.43 |
| 3 | bird | bird | 0.0768 | 11.15 |
| 4 | horse | horse | 0.0789 | 11.03 |

Summary:

- Label inference accuracy: 5/5 = 100%
- Average MSE: approximately 0.0626
- Average PSNR: approximately 12.61 dB
- Average runtime: approximately 63 seconds per sample

Visual observation: reconstructions are noisy but preserve some dominant colors and approximate object structures or silhouettes.

## Results: final fully connected layer only

| Sample | True label | Inferred label | MSE | PSNR |
|---:|---|---|---:|---:|
| 0 | airplane | airplane | 0.0284 | 15.47 |
| 1 | frog | frog | 0.1386 | 8.58 |
| 2 | airplane | airplane | 0.0545 | 12.63 |
| 3 | bird | bird | 0.1330 | 8.76 |
| 4 | horse | horse | 0.0797 | 10.99 |

Summary:

- Label inference accuracy: 5/5 = 100%
- Average MSE: approximately 0.0868
- Average PSNR: approximately 11.29 dB
- Average runtime: approximately 18 seconds per sample

Visual observation: reconstructions do not preserve clear dominant colors, silhouettes or object shapes. The final layer gradients are sufficient for label inference, but insufficient for meaningful visual reconstruction.

## Interpretation

The experiment shows two levels of privacy leakage.

First, semantic leakage is strong: the true label is inferred correctly in all evaluated samples both when using only the final layer gradients and when using all gradients.

Second, visual leakage depends on the available gradient information. Using only the final fully connected layer does not produce meaningful image reconstructions. Using all model gradients produces noisy but partially informative reconstructions, with some dominant colors and approximate structures.

This supports the idea that gradient information can leak private information even before considering full federated updates. It also shows that the amount and type of leaked information depends strongly on which gradients are exposed to the attacker.
