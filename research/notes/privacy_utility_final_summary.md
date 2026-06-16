# Final privacy-utility summary

## Main result

The final comparison combines model utility and reconstruction privacy.

| Configuration | Final accuracy | Final loss | Mean reconstruction PSNR | Mean PSNR drop | Label inference | Interpretation |
|---|---:|---:|---:|---:|---|---|
| No defense | 0.8396 | 0.5095 | 28.3511 dB | 0.0000 dB | 5/5 | High utility, high visual leakage |
| Clipping C=7.5 | 0.8359 | 0.5305 | NA | NA | Not evaluated on samples 0-4 | Preserves utility; pure clipping is weak against cosine matching |
| Clipping C=7.5 + noise 0.002 | 0.8300 | 0.5823 | 13.5104 dB | 14.8408 dB | 5/5 | High utility and strongly reduced visual leakage |

## Interpretation

The selected clipping plus Gaussian noise defense preserves high federated learning utility, reaching 83.00% final accuracy compared with 83.96% without defense.

At the same time, it substantially reduces reconstruction quality. The mean best-oracle PSNR decreases from 28.3511 dB without defense to 13.5104 dB with clipping plus noise, corresponding to a mean drop of 14.8408 dB.

This indicates a clear privacy-utility trade-off: visual reconstruction leakage is strongly reduced while model accuracy remains high.

However, label inference remains successful in all five evaluated samples. Therefore, the defense mitigates visual reconstruction leakage but does not eliminate label leakage in this controlled setup.

## Final conclusion

The strongest defensible result of this experimental block is:

Clipping plus Gaussian noise with C=7.5 and noise_std=0.002 maintains high model utility, with 83.00% final accuracy, while reducing the average reconstruction PSNR from 28.35 dB to 13.51 dB. Nevertheless, label inference remains correct in all five evaluated samples.
