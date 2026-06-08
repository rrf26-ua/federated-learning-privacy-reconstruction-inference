# local_update_inversion_bs1_resnet_all_iter2000_tv1e-6_seed42

## Purpose

Evaluate whether a local client update, represented as a one-step SGD weight delta, leaks private information about the training sample.

This experiment bridges the previous direct gradient inversion experiments and a more realistic federated learning scenario, where the server observes model updates rather than raw gradients.

## Configuration

- Dataset: CIFAR-10
- Split: train
- Model: ResNet-18 adapted for CIFAR-10
- Attack: local update inversion
- Update type: one-step local SGD update
- Batch size: 1
- Sample index: 0
- True label: airplane
- Gradient/update scope: all trainable parameters
- Local learning rate: 0.02
- Weight decay: 0.0005
- Attack learning rate: 0.03
- Total variation weight: 0.000001
- Iterations: 2000
- Seed: 42
- Device: cuda:0
- Labels used for reconstruction: true labels, in this controlled experiment

## Results

| Metric | Value |
|---|---:|
| Inferred label from update | airplane |
| Best update loss | 3.5114e-06 |
| Best update iteration | 1988 |
| Best-update MSE | 0.01825 |
| Best-update PSNR | 17.39 dB |
| Best-oracle MSE | 0.01820 |
| Best-oracle PSNR | 17.40 dB |
| Final MSE | 0.01820 |
| Final PSNR | 17.40 dB |
| Runtime | 70.76 s |

## Interpretation

The attack successfully infers the correct label from the local update. This shows that semantic information can leak from a one-step SGD weight delta.

The visual reconstruction is also the strongest obtained so far in these experiments. The best-update reconstruction, the best-oracle reconstruction and the final reconstruction have very similar MSE and PSNR values. This is important because the best-update criterion is the realistic attacker criterion: the attacker can minimize the mismatch with the observed update without access to the original image.

The result shows that, at least in a controlled one-sample and one-step SGD setting, a local model update can leak information comparable to direct gradients.

## Conclusion

A local update can leak both semantic information and partial visual information. This supports the relevance of studying privacy attacks not only on raw gradients, but also on the actual updates exchanged in federated learning protocols.
