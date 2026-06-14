# FedAvg E=2 baseline with lr=0.02

## Configuration

- Dataset: CIFAR-10
- Model: ResNet-18 adapted to CIFAR-10
- Federated algorithm: FedAvg
- Server strategy: FedAvg
- Clients: 6
- Rounds: 20
- Local epochs: 2
- Learning rate: 0.02
- Batch size: 32
- Seed: 42
- Participation: 100%
- Evaluation: centralized server-side CIFAR-10 test set

## Main results

- Final accuracy: 0.8330
- Best accuracy: 0.8382 at round 19
- Final loss: 0.5395
- Best loss: 0.5137 at round 19
- Elapsed time: 762.64 s

## Interpretation

FedAvg with two local epochs substantially improves over FedAvg E=1 and strict FedSGD. This confirms that the low performance of strict FedSGD was mainly caused by the very small number of effective optimization steps.

Compared with FedAvg E=2 using lr=0.03, this configuration is slightly worse, but still provides a strong baseline and is useful as part of the learning-rate comparison.

## Experimental decision

FedAvg E=2 with lr=0.02 is kept as a comparison point. FedAvg E=2 with lr=0.03 is selected as the main intermediate baseline.
