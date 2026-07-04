# gradient_inversion_resnet_all_iter2000_seed42

## Purpose

Evaluate a controlled gradient inversion attack on individual CIFAR-10 samples using the ResNet-18 model adapted for CIFAR-10.

This experiment is performed outside Flower in order to validate the reconstruction attack pipeline before applying it to federated client updates.

## Configuration

- Dataset: CIFAR-10
- Split: train
- Model: ResNet-18 adapted for CIFAR-10
- Attack: gradient inversion
- Gradient scope: all trainable parameter tensors
- Number of samples: 5
- Sample indices: 0, 1, 2, 3, 4
- Iterations per sample: 2000
- Attack learning rate: 0.03
- Total variation weight: 0.001
- Seed: 42
- Device: cuda:0

## Command

```bash
./scripts/run_gradient_inversion_batch.sh
with default values:

SAMPLES="0 1 2 3 4"
ITERATIONS=2000
GRAD_SCOPE=all
ATTACK_LR=0.03
TV_WEIGHT=0.001
SEED=42
Results
Sample	True label	Inferred label	MSE	PSNR
0	airplane	airplane	0.0196	17.08
1	frog	frog	0.0922	10.35
2	airplane	airplane	0.0454	13.43
3	bird	bird	0.0768	11.15
4	horse	horse	0.0789	11.03
Summary
Label inference accuracy: 5/5 = 100%
Average MSE: approximately 0.0626
Average PSNR: approximately 12.61 dB
Average runtime: approximately 63 seconds per sample
Interpretation

The attack successfully infers the correct label for all evaluated samples using gradient information. Visual reconstruction quality is partial: the reconstructed images preserve some dominant colors and approximate structures or silhouettes, but they remain noisy and deformed.

This result demonstrates a relevant privacy leakage even before attacking full federated updates. The strongest leakage observed in this experiment is semantic leakage through label inference, while visual reconstruction is present but limited.

Notes

The experiment should be extended by comparing different gradient scopes, batch sizes, number of iterations, and possibly alternative gradient matching losses.
