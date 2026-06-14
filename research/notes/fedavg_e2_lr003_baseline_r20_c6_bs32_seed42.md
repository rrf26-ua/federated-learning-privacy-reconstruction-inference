# FedAvg E=2 baseline with lr=0.03

## Configuration

- Dataset: CIFAR-10
- Model: ResNet-18 adapted to CIFAR-10
- Federated algorithm: FedAvg
- Server strategy: FedAvg
- Clients: 6
- Rounds: 20
- Local epochs: 2
- Learning rate: 0.03
- Batch size: 32
- Seed: 42
- Participation: 100%
- Evaluation: centralized server-side CIFAR-10 test set

## Main results

- Final accuracy: 0.8396
- Best accuracy: 0.8396 at round 20
- Final loss: 0.5095
- Elapsed time: 759.48 s

## Interpretation

FedAvg with two local epochs and learning rate 0.03 achieved the best result among the E=2 configurations tested. It slightly improves over lr=0.02, which reached a best accuracy of 0.8382 and final accuracy of 0.8330.

This configuration is selected as the main intermediate FedAvg baseline. It offers a strong trade-off between performance and local computation, reaching 83.96% centralized test accuracy while remaining close to the stronger FedAvg E=10 baseline.

## Experimental decision

Use FedAvg E=2, lr=0.03, 20 rounds, 6 clients, batch size 32 as the main intermediate baseline for subsequent privacy and defense experiments.
