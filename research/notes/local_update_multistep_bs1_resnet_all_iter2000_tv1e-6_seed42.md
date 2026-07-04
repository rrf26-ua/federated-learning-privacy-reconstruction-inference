# local_update_multistep_bs1_resnet_all_iter2000_tv1e-6_seed42

## Purpose

Evaluate what happens to reconstruction quality when the local client update accumulates several SGD steps over the same private batch.

This experiment studies whether multi-step local training reduces the privacy leakage observed in one-step local update inversion.

## Configuration

- Dataset: CIFAR-10
- Split: train
- Model: ResNet-18 adapted for CIFAR-10
- Attack: multi-step local update inversion
- Batch size: 1
- Sample index: 0
- Label: airplane
- Gradient/update scope: all trainable parameters
- Iterations: 2000
- Local learning rate: 0.02
- Weight decay: 0.0005
- Attack learning rate: 0.03
- Total variation weight: 0.000001
- Seed: 42
- Device: cuda:0
- Labels: true labels are used in this controlled reconstruction experiment

## Results

| Local steps | Best-update MSE | Best-update PSNR | Best-oracle PSNR | Final PSNR | Runtime |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01825 | 17.39 | 17.40 | 17.40 | 70.76 s |
| 2 | 0.01909 | 17.19 | 17.23 | 17.16 | 144.13 s |
| 5 | 0.02658 | 15.76 | 16.60 | 16.58 | 351.41 s |
| 10 | 0.01939 | 17.12 | 17.23 | 17.23 | 645.82 s |

## Interpretation

The results show that privacy leakage persists when the local update accumulates several local SGD steps over the same private sample.

The reconstruction quality does not degrade monotonically with the number of local steps. The 5-step update produces a weaker best-update reconstruction than 1 and 2 steps, but the 10-step update again reaches a final PSNR close to the 1-step and 2-step cases.

This suggests that increasing the number of local steps is not, by itself, a reliable privacy defense in this controlled setting. Although the update becomes more expensive to attack and the relationship between update loss and visual similarity can become less direct, the accumulated update can still preserve enough information to reconstruct the original sample partially.

## Key observation

The runtime increases strongly with the number of local steps:

- 1 step: approximately 71 seconds
- 2 steps: approximately 144 seconds
- 5 steps: approximately 351 seconds
- 10 steps: approximately 646 seconds

Therefore, local steps increase the computational cost of the attack, but they do not eliminate visual leakage in this experiment.

## Limitations

This experiment uses batch size 1 and repeats local steps over the same sample. This is not yet equivalent to a realistic local training epoch over multiple different mini-batches.

True labels are used for reconstruction, so this experiment isolates visual reconstruction leakage and does not evaluate multi-label inference.

The result should not be generalized directly to all FedAvg settings. It shows that repeated local steps over the same private sample can still leak information, but the next experiment should study local updates produced by several batches or full local epochs.

## Conclusion

Multi-step local updates remain vulnerable to reconstruction attacks in this controlled setup. More local steps increase attack cost, but they are not a sufficient privacy mitigation when the update is still strongly determined by a small private batch.
