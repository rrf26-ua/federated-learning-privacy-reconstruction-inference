# resnet6_fedavg_r5_e10_lr002_seed42

## Purpose

Strong federated learning utility baseline using the ResNet-18 configuration adapted from the companion Flower implementation.

## Configuration

- Framework: Flower + PyTorch
- Dataset: CIFAR-10
- Strategy: FedAvg
- Model: ResNet-18 adapted to CIFAR-10
- Clients: 6
- Rounds: 5
- Local epochs: 10
- Learning rate: 0.02
- Batch size: 32
- Seed: 42
- Federated client-side evaluation: disabled
- Server-side centralized evaluation: enabled

## Federation configuration

```bash
--federation-config "num-supernodes=6 client-resources-num-cpus=1 client-resources-num-gpus=0.5 init-args-num-cpus=8 init-args-num-gpus=1"

Execution command
flwr run . --stream \
  --run-config "num-server-rounds=5 seed=42 learning-rate=0.02 local-epochs=10 batch-size=32 fraction-train=1.0 fraction-evaluate=0.0" \
  --federation-config "num-supernodes=6 client-resources-num-cpus=1 client-resources-num-gpus=0.5 init-args-num-cpus=8 init-args-num-gpus=1" \
  | tee results/raw/resnet6_fedavg_r5_e10_lr002_seed42.log

Results
Round	Accuracy	Loss
0	0.1000	2.9329
1	0.1033	2.6165
2	0.5315	1.3397
3	0.8212	0.5367
4	0.8447	0.4725
5	0.8672	0.4151

Summary
Best accuracy: 0.8672
Best round: 5
Final accuracy: 0.8672
Final loss: 0.4151
Runtime: 949.72 seconds
Client failures: 0
Interpretation

This configuration provides a strong utility baseline for the joint part of the TFG. It substantially outperforms the lightweight CNN baseline, but it is also much more computationally expensive and should be considered separately from the lightweight security baseline.
