# baseline_security_fedavg_cifar10_cnn_seed42

## Purpose

Stable security baseline for the federated learning privacy/security experiments.

## Configuration

- Framework: Flower + PyTorch
- Dataset: CIFAR-10
- Strategy: FedAvg
- Model: lightweight CNN
- Simulated clients: 4
- Rounds: 20
- Local epochs: 1
- Learning rate: 0.01
- Batch size: 32
- Seed: 42
- Fraction evaluate: 0.5

## Federation configuration

```bash
--federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2"

Execution command
flwr run . --stream \
  --run-config "num-server-rounds=20 seed=42 learning-rate=0.01 local-epochs=1 batch-size=32 fraction-evaluate=0.5" \
  --federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2"

Results
Server-side evaluation metrics
Round	Accuracy	Loss
0	0.1015	2.3059
1	0.3564	1.7758
2	0.4426	1.5396
3	0.4741	1.4454
4	0.4962	1.3764
5	0.5214	1.3225
6	0.5443	1.2654
7	0.5587	1.2315
8	0.5741	1.1970
9	0.5832	1.1745
10	0.5870	1.1557
11	0.5899	1.1481
12	0.5951	1.1386
13	0.5966	1.1295
14	0.5999	1.1221
15	0.5991	1.1238
16	0.6041	1.1228
17	0.6028	1.1228
18	0.6091	1.1284
19	0.6013	1.1254
20	0.6019	1.1423

Summary
Best global accuracy: 0.6091
Best round: 18
Final global accuracy: 0.6019
Final global loss: 1.1423
Interpretation

This baseline is suitable as the security baseline for the privacy and reconstruction attack experiments. It is lightweight, reproducible, stable on the available hardware, and avoids the heavy computational cost observed with ResNet-18.

For the joint utility-oriented part of the thesis, a stronger baseline should be explored separately in another branch.
