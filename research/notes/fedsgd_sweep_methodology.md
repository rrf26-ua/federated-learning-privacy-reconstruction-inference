# FedSGD sweep methodology

## Invariants

- Each client computes the accumulated gradient over its complete local training
  split and applies exactly one parameter update per server round.
- The server uses example-weighted FedAvg aggregation of those single-step client
  parameters. Local epochs are not used by the FedSGD path.
- Dataset, model, normalization, IID partitioner, defenses and centralized
  evaluation are unchanged.

## Configurable parameters

- `weight-decay`: L2 term applied during the single FedSGD update.
- `scheduler`: `none` for a constant learning rate or `cosine` for cosine decay.
- `scheduler-min-lr`: final learning rate for cosine decay. Round 1 uses the base
  `learning-rate`; the final round uses `scheduler-min-lr`.
- `seed-with-server-round`: disabled by default. When enabled, the process-wide
  training seed incorporates the server round, varying stochastic data
  augmentation between rounds. The client partition and its 80/20 train split
  remain tied to the fixed client seed and therefore do not change by round.

## BatchNorm caveat

BatchNorm behavior is intentionally unchanged in this phase. Its running buffers
are updated while each client traverses local minibatches and are subsequently
aggregated with the client state. Consequently, the trainable-parameter update is
FedSGD-style, but the complete state transition is not exactly identical to one
centralized full-batch SGD step. Batch size can affect these BatchNorm buffers even
though client gradients are accumulated over the complete local training split.

## Result preservation

The sweep utility creates every raw log with an exclusive, timestamped filename,
appends one row per attempted run to `fedsgd_sweep_summary.csv`, and disables final
model saving so an existing checkpoint cannot be overwritten.
