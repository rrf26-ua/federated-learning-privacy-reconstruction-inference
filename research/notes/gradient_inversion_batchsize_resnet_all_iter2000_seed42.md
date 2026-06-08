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
- Batch sizes evaluated: 1, 2, 4
- Labels: true labels are provided to the attacker in this controlled experiment

## Results

| Batch size | Sample indices | Labels | MSE | PSNR |
|---:|---|---|---:|---:|
| 1 | 0 | airplane | 0.0188 | 17.25 |
| 2 | 0, 1 | airplane, frog | 0.0624 | 12.05 |
| 4 | 0, 1, 2, 3 | airplane, frog, airplane, bird | 0.0966 | 10.15 |

## Per-sample results

### Batch size 1

| Sample | Label | MSE | PSNR |
|---:|---|---:|---:|
| 0 | airplane | 0.0188 | 17.25 |

### Batch size 2

| Sample | Label | MSE | PSNR |
|---:|---|---:|---:|
| 0 | airplane | 0.0221 | 16.55 |
| 1 | frog | 0.1027 | 9.89 |

### Batch size 4

| Sample | Label | MSE | PSNR |
|---:|---|---:|---:|
| 0 | airplane | 0.0589 | 12.30 |
| 1 | frog | 0.1156 | 9.37 |
| 2 | airplane | 0.0777 | 11.10 |
| 3 | bird | 0.1342 | 8.72 |

## Interpretation

The results show a clear degradation in reconstruction quality as batch size increases. With batch size 1, the attack obtains the best reconstruction quality. With batch size 2, the global MSE increases and PSNR drops substantially. With batch size 4, the reconstruction becomes significantly more degraded and mixed.

This supports the hypothesis that gradients computed from larger batches leak less precise information about individual samples because the signal from several images is mixed. However, this does not eliminate privacy leakage completely: even in larger batches, some dominant colors and approximate visual structures may still appear.

This experiment uses true labels during reconstruction. Therefore, it isolates the effect of batch size on visual reconstruction and does not yet evaluate the harder problem of multi-label inference from batch gradients.

## Conclusion

Increasing the batch size acts as a partial privacy mitigation against individual image reconstruction, but it is not a complete defense. The attacker still receives enough information to produce noisy aggregate reconstructions and, in some cases, partial visual cues.
