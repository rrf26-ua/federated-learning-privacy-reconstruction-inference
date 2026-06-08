# local_update_inversion_batchsize_resnet_all_iter2000_tv1e-6_seed42

## Purpose

Evaluate how batch size affects a controlled local update inversion attack on CIFAR-10 using a ResNet-18 model adapted for CIFAR-10.

This experiment extends the previous direct gradient inversion analysis to a more federated-learning-like setting, where the attacker observes a one-step local SGD weight delta instead of a raw gradient.

## Configuration

- Dataset: CIFAR-10
- Split: train
- Model: ResNet-18 adapted for CIFAR-10
- Attack: local update inversion
- Update type: one-step local SGD weight delta
- Gradient/update scope: all trainable parameters
- Iterations: 2000
- Local learning rate: 0.02
- Weight decay: 0.0005
- Attack learning rate: 0.03
- Total variation weight: 0.000001
- Seed: 42
- Device: cuda:0
- Batch sizes evaluated: 1, 2, 4
- Labels: true labels are provided to the attacker in this controlled experiment

## Results

| Batch size | Sample indices | Labels | Best-update MSE | Best-update PSNR | Final PSNR |
|---:|---|---|---:|---:|---:|
| 1 | 0 | airplane | 0.01825 | 17.39 | 17.40 |
| 2 | 0, 1 | airplane, frog | 0.07693 | 11.14 | 11.15 |
| 4 | 0, 1, 2, 3 | airplane, frog, airplane, bird | 0.09634 | 10.16 | 10.17 |

## Interpretation

The results show a clear degradation in reconstruction quality as the batch size increases.

With batch size 1, the local update inversion attack produces the strongest reconstruction obtained so far in the project. The label is inferred correctly from the local update, and the reconstructed image preserves relevant visual structure.

With batch sizes 2 and 4, the reconstruction quality drops substantially. This indicates that when the local update is computed from several samples, the individual visual signal is mixed and becomes harder to recover.

The best-update and final metrics are very close in all cases. This is important because best-update is the realistic attacker criterion: the attacker can minimize the difference with the observed update without knowing the original images.

## Comparison with direct gradient inversion

The local update inversion results are comparable to the previous direct gradient inversion results:

| Batch size | Direct gradient PSNR | Local update PSNR |
|---:|---:|---:|
| 1 | approximately 17.25 | approximately 17.40 |
| 2 | approximately 12.05 | approximately 11.15 |
| 4 | approximately 10.15 | approximately 10.17 |

This suggests that, in a controlled one-step SGD setting, attacking a local weight delta can leak information comparable to attacking a direct gradient.

## Limitations

This experiment uses true labels for reconstruction. Therefore, it isolates the effect of batch size on visual reconstruction and does not evaluate multi-label inference from updates.

The update is generated from a single local SGD step. More realistic federated learning settings may involve several local steps or full local epochs, which can further transform the relationship between private data and the observed update.

## Conclusion

A one-step local client update can leak both semantic and visual information. Batch size acts as a partial mitigation because larger batches produce more mixed and less individually identifiable reconstructions, but it does not remove the privacy leakage entirely.
