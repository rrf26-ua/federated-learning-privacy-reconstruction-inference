# baseline_flower_pytorch_cifar10_v1

## Metadata
- Date: 2026-04-22
- Framework: Flower + PyTorch
- Dataset: CIFAR-10
- Strategy: FedAvg
- Simulated clients: 4
- Concurrent execution: constrained through federation config
- Rounds: 3

## Stable execution config
- num-supernodes = 4
- client-resources-num-cpus = 2
- init-args-num-cpus = 2

## Results
- Initial global accuracy: 0.1000
- Round 1 global accuracy: 0.0999
- Round 2 global accuracy: 0.1720
- Round 3 global accuracy: 0.1638

- Initial global loss: 2.3044
- Round 1 global loss: 2.3083
- Round 2 global loss: 2.2549
- Round 3 global loss: 2.3216

- Train loss round 1: 2.1838
- Train loss round 2: 2.1079
- Train loss round 3: 2.0443

- Eval acc round 1: 0.0994
- Eval acc round 2: 0.1636
- Eval acc round 3: 0.1654

- Eval loss round 1: 2.3064
- Eval loss round 2: 2.2578
- Eval loss round 3: 2.3224

- Execution time: 49.29 s

## Observations
1. The default Flower quickstart configuration caused out-of-memory failures on the available machine.
2. A stable baseline was obtained by reducing the number of simulated clients and constraining simulation resources.
3. The pipeline now runs end-to-end without client failures.
4. Results are still preliminary because only one run/seed has been executed.
5. The baseline is sufficient as the starting point for repository refactoring and later attack integration.

## Next step
Refactor the quickstart project into a research-oriented repository structure without changing the baseline behavior.
