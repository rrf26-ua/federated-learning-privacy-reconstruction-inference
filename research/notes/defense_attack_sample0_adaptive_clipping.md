# Adaptive clipping analysis — sample 0 reconstruction attack

## Context

A controlled local update inversion attack was executed on CIFAR-10 sample 0 using cosine matching. After observing that clipping C=7.5 did not activate for a single-sample update, additional clipping thresholds were tested below the observed raw update norm.

The raw single-sample update norm was approximately 1.218655.

## Results

| Defense | Raw norm | Defended norm | Clip scale | Best oracle PSNR | Label inference |
|---|---:|---:|---:|---:|---|
| No defense | 1.218655 | 1.218655 | 1.0000 | 35.7765 | airplane |
| Clipping C=1.0 | 1.218655 | 1.000000 | 0.8206 | 36.3997 | airplane |
| Clipping C=0.6 | 1.218655 | 0.600000 | 0.4923 | 37.2908 | airplane |
| Clipping C=0.3 | 1.218655 | 0.300000 | 0.2462 | 36.1842 | airplane |
| Clipping C=7.5 + noise 0.002 | 1.218655 | 6.794307 | 1.0000 | 16.7802 | airplane |

## Interpretation

Pure clipping strongly reduces the norm of the observed update when the threshold is below the raw update norm. However, reconstruction quality remains high when the attacker uses cosine matching.

This is expected because clipping scales the update magnitude but preserves its direction. Since cosine matching is largely invariant to global scaling, the attacker can still exploit the directional information contained in the update.

In contrast, Gaussian noise substantially degrades reconstruction quality because it changes the direction of the observed update. The best oracle PSNR drops from approximately 35.78 dB without defense to approximately 16.78 dB with clipping plus noise.

Label inference still succeeds for this sample in all tested configurations.

## Conclusion

For this controlled single-sample local update attack, pure norm clipping is not sufficient against a cosine-matching reconstruction attacker. The most effective tested defense is the addition of Gaussian noise after clipping.
