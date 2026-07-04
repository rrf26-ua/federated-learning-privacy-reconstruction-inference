# BN-aware inversion on a real Flower update

This experiment analyzes the inversion of a real client update captured from a Flower FedAvg execution. The captured update corresponds to round 1, client 0, batch size 1, using CIFAR-10 and a ResNet-18 model adapted to CIFAR-10.

## Motivation

The initial parameter-only inversion attack produced poor visual reconstructions from the real captured update. A diagnostic decomposition showed that the captured update contains both trainable parameter deltas and BatchNorm buffer deltas.

The norm decomposition was:

- parameter update norm: approximately 0.55
- BatchNorm buffer update norm: approximately 5.47
- total floating-point update norm: approximately 5.50

Therefore, the real Flower update is dominated by BatchNorm buffer changes (`running_mean` and `running_var`). A parameter-only attack ignores most of the observable signal.

## BN-aware attack

A BN-aware inversion script was implemented to include three terms:

- parameter update matching;
- BatchNorm `running_mean` delta matching;
- BatchNorm `running_var` delta matching.

The BatchNorm buffer deltas were computed differentiably from the dummy input by capturing the activations entering each BatchNorm layer and manually reconstructing the expected change in the running statistics.

## Operator validation

Before optimizing from random initialization, the BN-aware operator was evaluated on the true private image stored in the capture. The true image produced near-zero BatchNorm losses and a very low parameter cosine loss:

- true parameter cosine loss: approximately 0.0038
- true BN mean cosine loss: approximately 2.4e-7
- true BN var cosine loss: approximately 2.5e-6

This validates that the BN-aware simulator is aligned with the captured Flower update.

## Optimization from random initialization

Despite the correct operator alignment, optimization from random initialization did not produce high-quality visual reconstructions.

The following variants were evaluated:

- Adam, 200 iterations;
- Adam with increased parameter weight;
- Adam multi-start with seeds 1, 7, 42, 123 and 999;
- LBFGS from random initialization.

All random-initialized runs converged to low-quality reconstructions around PSNR ≈ 10.

## Warm-start diagnostic

A diagnostic warm-start was performed using the high-quality controlled reconstruction as initialization. This is not considered a fair standalone attack, because it uses information from a previous controlled experiment. However, it is useful to evaluate the objective landscape.

The warm-start began at PSNR ≈ 26 and remained around PSNR ≈ 25-26 after BN-aware optimization. This shows that high-quality images are compatible with the BN-aware objective, but the optimizer does not robustly reach them from random initialization.

## Interpretation

The real captured update contains enough signal to validate the private image and correctly infer the label. However, direct pixel-level reconstruction from random initialization is unstable under the current BN-aware objective and optimizer settings.

The limitation is not capture correctness. The main limitation is the non-convex optimization landscape induced by training-mode updates and BatchNorm statistics.

## Conclusion

The real-update experiment supports three conclusions:

1. Real Flower updates contain significant non-parametric leakage through BatchNorm buffers.
2. The BN-aware objective is correctly aligned with the captured update.
3. High-fidelity visual reconstruction from random initialization remains difficult, even when BatchNorm buffers are explicitly exploited.
