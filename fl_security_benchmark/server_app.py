"""fl_security_benchmark: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdam, FedAvg, FedAvgM

from fl_security_benchmark.task import Net, load_centralized_dataset, test
from fl_security_benchmark.utils.reproducibility import set_seed

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    seed = int(context.run_config["seed"])
    fraction_train = float(context.run_config.get("fraction-train", 1.0))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 0.0))
    algorithm = str(context.run_config.get("fl-algorithm", "fedavg")).lower()
    strategy_name = str(context.run_config.get("fl-strategy", "fedavg")).lower()

    server_learning_rate = float(context.run_config.get("server-learning-rate", 1.0))
    server_momentum = float(context.run_config.get("server-momentum", 0.0))

    fedopt_eta = float(context.run_config.get("fedopt-eta", 0.1))
    fedopt_eta_l = float(context.run_config.get("fedopt-eta-l", lr))
    fedopt_beta1 = float(context.run_config.get("fedopt-beta1", 0.9))
    fedopt_beta2 = float(context.run_config.get("fedopt-beta2", 0.99))
    fedopt_tau = float(context.run_config.get("fedopt-tau", 0.001))

    set_seed(seed)

    print(f"[INFO] Federated algorithm mode: {algorithm}")
    print(f"[INFO] Server strategy mode: {strategy_name}")

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    common_strategy_kwargs = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": 1,
        "min_evaluate_nodes": 0,
        "min_available_nodes": 1,
    }

    if strategy_name == "fedavg":
        strategy = FedAvg(**common_strategy_kwargs)
    elif strategy_name == "fedavgm":
        strategy = FedAvgM(
            **common_strategy_kwargs,
            server_learning_rate=server_learning_rate,
            server_momentum=server_momentum,
        )
    elif strategy_name == "fedadam":
        strategy = FedAdam(
            **common_strategy_kwargs,
            eta=fedopt_eta,
            eta_l=fedopt_eta_l,
            beta_1=fedopt_beta1,
            beta_2=fedopt_beta2,
            tau=fedopt_tau,
        )
    else:
        raise ValueError(f"Unknown fl-strategy: {strategy_name}")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if bool(context.run_config.get("save-final-model", True)):
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"final_model_{algorithm}_{strategy_name}.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global model on the centralized CIFAR-10 test set."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    print(
        f"Server-side evaluation round {server_round}: "
        f"loss={test_loss:.4f}, accuracy={test_acc:.4f}"
    )

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
