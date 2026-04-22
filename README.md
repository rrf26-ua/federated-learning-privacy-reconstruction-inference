# Privacy and Security Evaluation of Federated Learning

This repository contains the code, experiments, and research notes for a Bachelor's Thesis focused on privacy and security in federated learning using Flower and PyTorch.

## Research goal

The main objective is to study how much private information can leak from federated learning updates and how different defenses affect both privacy and utility.

The project starts from a stable Flower baseline and will progressively incorporate:
- reconstruction and inference attacks,
- alternative federated optimization strategies,
- privacy and security defenses,
- evaluation of privacy-utility trade-offs.

## Current status

The repository currently includes a stable baseline based on:
- Flower
- PyTorch
- CIFAR-10
- FedAvg

A stable local simulation configuration was obtained after reducing the number of simulated clients and constraining the available resources of the simulation backend.

## Stable baseline configuration

Baseline execution used the following simulation parameters:

- `num-supernodes=4`
- `client-resources-num-cpus=2`
- `init-args-num-cpus=2`

Example command:

```bash
flwr run . --stream \
  --run-config "num-server-rounds=3" \
  --federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2"
First baseline result

Experiment ID: baseline_flower_pytorch_cifar10_v1

Main observations:

the default quickstart configuration caused out-of-memory failures on the available machine,
the constrained configuration completed successfully,
the pipeline now runs end-to-end without client failures,
results are preliminary and correspond to a single run.
Repository structure

This repository is being transformed from the official Flower quickstart into a research-oriented benchmark.

Current research-oriented directories include:

research/notes/ for experiment tracking,
attacks/ for reconstruction and inference attacks,
defenses/ for privacy and security mechanisms,
experiments/ for experiment orchestration,
configs/ for reproducible settings,
results/ for generated outputs and processed results.
Planned work
Keep a stable FedAvg baseline.
Add experiment tracking and reproducibility support.
Integrate privacy attacks such as gradient-based reconstruction.
Compare multiple federated learning strategies.
Add defenses such as clipping, differential privacy, and secure aggregation.
Measure the privacy-utility trade-off.
Reproducibility note

Baseline behavior should remain reproducible while the repository evolves.

License

This repository currently includes the license distributed with the initial Flower app template. It may be revised later depending on thesis and supervisor requirements.