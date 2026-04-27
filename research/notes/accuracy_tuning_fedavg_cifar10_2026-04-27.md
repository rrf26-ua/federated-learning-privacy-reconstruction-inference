# accuracy_tuning_fedavg_cifar10_2026-04-27

## Scope
This note documents the baseline verification and controlled tuning attempts for Flower + PyTorch + CIFAR-10 under the stable simulation setting:

- `num-supernodes=4`
- `client-resources-num-cpus=2`
- `init-args-num-cpus=2`

## Baseline target (to verify)
Expected known-good baseline to verify:

- Strategy: FedAvg
- Clients: 4
- Rounds: 20
- Learning rate: 0.01
- Local epochs: 1
- Batch size: 32
- Seed: 42
- No augmentation

## Commands executed

### 1) Baseline verification
```bash
flwr run . --stream \
  --run-config "num-server-rounds=20 learning-rate=0.01 local-epochs=1 batch-size=32 seed=42 fraction-evaluate=1.0" \
  --federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2" \
  | tee results/raw/exp_baseline_r20_lr001_le1_bs32_seed42.log
```

### 2) Controlled change: learning-rate only
```bash
flwr run . --stream \
  --run-config "num-server-rounds=20 learning-rate=0.008 local-epochs=1 batch-size=32 seed=42 fraction-evaluate=1.0" \
  --federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2" \
  | tee results/raw/exp_lr0008_r20_le1_bs32_seed42.log
```

### 3) Controlled change: local-epochs only
```bash
flwr run . --stream \
  --run-config "num-server-rounds=20 learning-rate=0.01 local-epochs=2 batch-size=32 seed=42 fraction-evaluate=1.0" \
  --federation-config "num-supernodes=4 client-resources-num-cpus=2 init-args-num-cpus=2" \
  | tee results/raw/exp_le2_r20_lr001_bs32_seed42.log
```

## Outcome
All runs failed before round 0 evaluation with the same error chain:

- `httpx.ProxyError: 403 Forbidden`
- Triggered while calling `load_dataset("uoft-cs/cifar10", split="test")`

Therefore:

- Baseline accuracy could not be re-verified in this environment.
- No best checkpoint or best round can be computed from this session.
- No fair hyperparameter comparison can be made until dataset access is restored.

## Traceability
- Structured experiment table: `results/processed/accuracy_tuning_experiments.csv`
- Raw logs:
  - `results/raw/exp_baseline_r20_lr001_le1_bs32_seed42.log`
  - `results/raw/exp_lr0008_r20_le1_bs32_seed42.log`
  - `results/raw/exp_le2_r20_lr001_bs32_seed42.log`

## Reproduction helpers
- Baseline script: `scripts/run_baseline_fedavg_cifar10.sh`
- Parameterized experiment script: `scripts/run_experiment_fedavg_cifar10.sh`

## Next action once dataset access is available
1. Re-run baseline command above and record best round/accuracy.
2. Continue controlled grid in this order:
   - local-epochs (`1 -> 2`)
   - learning-rate near baseline (`0.008`, `0.01`, `0.012`)
   - rounds (`20 -> 30`) once early behavior is stable
   - batch-size (`32 -> 64`) only after confirming no regression
3. Keep one-variable-at-a-time changes and append to CSV.
