"""fl_security_benchmark: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

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

    set_seed(seed)

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=1,
        min_evaluate_nodes=0,
        min_available_nodes=1,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


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
