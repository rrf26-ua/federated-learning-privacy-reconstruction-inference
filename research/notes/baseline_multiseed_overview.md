# baseline_multiseed_overview

## Objective
Assess the stability of the seeded FedAvg CIFAR-10 baseline under multiple seeds.

## Configuration
- Framework: Flower + PyTorch
- Dataset: CIFAR-10
- Strategy: FedAvg
- Simulated clients: 4
- num-server-rounds: 3
- client-resources-num-cpus: 2
- init-args-num-cpus: 2

## Seeds
- 42
- 43
- 44

## What to compare
- Initial and final global accuracy
- Initial and final global loss
- Train loss by round
- Eval accuracy by round
- Eval loss by round
- Execution time

## Interpretation goal
Determine whether the baseline is stable enough to serve as the reference point for later attack and defense experiments.

## Next step
Summarize the multiseed results and then begin building the attack experiment scaffolding.
