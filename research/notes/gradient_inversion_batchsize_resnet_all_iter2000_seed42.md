# gradient_inversion_batchsize_resnet_all_iter2000_seed42

## Purpose

Evaluate how batch size affects controlled gradient inversion attacks on CIFAR-10 using the ResNet-18 model adapted for CIFAR-10.

This experiment studies visual reconstruction leakage when the observed gradient is computed from batches of different sizes.

## Configuration

- Dataset: CIFAR-10
- Split: train
- Model: ResNet-18 adapted for CIFAR-10
- Attack: batch gradient inversion
- Gradient scope: all trainable model parameters
- Iterations: 2000
- Attack learning rate: 0.03
- Total variation weight: 0.001
- Seed: 42
- Device: cuda:0
- Batch sizes evaluated: 1, 2, 4, 8
- Labels: true labels are provided to the attacker in this controlled experiment

## Results

| Batch size | Sample indices | Labels | MSE | PSNR |
|---:|---|---|---:|---:|
| 1 | 0 | airplane | 0.0188 | 17.25 |
| 2 | 0, 1 | airplane, frog | 0.0624 | 12.05 |
| 4 | 0, 1, 2, 3 | airplane, frog, airplane, bird | 0.0966 | 10.15 |
| 8 | 0, 1, 2, 3, 4, 5, 6, 7 | airplane, frog, airplane, bird, horse, bird, automobile, bird | 0.0831 | 10.81 |

## Interpretation

The results show a clear degradation in reconstruction quality when moving from batch size 1 to larger batches. With batch size 1, the attack obtains the best individual reconstruction quality. With batch sizes 2, 4 and 8, the reconstruction becomes substantially noisier and less useful for identifying individual samples.

The degradation is not perfectly monotonic: batch size 8 obtains a slightly better global PSNR than batch size 4. This does not invalidate the main conclusion, because both batch sizes produce much poorer reconstructions than batch size 1. The difference can be explained by the non-convex nature of the optimization problem, the specific samples included in each batch, and the fact that global metrics can hide per-sample variation.

This experiment uses true labels during reconstruction. Therefore, it isolates the effect of batch size on visual reconstruction and does not yet evaluate the harder problem of multi-label inference from batch gradients.

## Conclusion

Increasing the batch size acts as a partial privacy mitigation against individual image reconstruction, but it is not a complete defense. Larger batches reduce the clarity of individual reconstructions, while still allowing noisy aggregate visual information to leak.
