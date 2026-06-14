# Strict FedSGD baseline — 50 rounds

## Configuration

- Dataset: CIFAR-10
- Model: ResNet-18 adapted to CIFAR-10
- Client algorithm: FedSGD-style update
- Server strategy: FedAvg parameter aggregation
- Clients: 6
- Rounds: 50
- Learning rate: 0.05
- Batch size: 64
- Seed: 42
- Participation: 100%

## Result

- Final accuracy: 0.3088
- Best accuracy: 0.3128 at round 49
- Final loss: 1.8200
- Elapsed time: 896.51 s

## Interpretation

The strict FedSGD configuration learns steadily but converges slowly. This is expected because each communication round corresponds to only one effective global update. In contrast, FedAvg performs multiple local optimization steps before communicating.

This result should be treated as a strict FedSGD baseline, not as a competitive final model.
