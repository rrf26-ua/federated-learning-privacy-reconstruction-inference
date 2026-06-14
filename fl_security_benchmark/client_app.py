"""fl_security_benchmark: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_security_benchmark.task import Net, load_data
from fl_security_benchmark.task import test as test_fn
from fl_security_benchmark.task import train as train_fn
from fl_security_benchmark.task import train_fedsgd as train_fedsgd_fn
from fl_security_benchmark.utils.reproducibility import set_seed

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    base_seed = int(context.run_config["seed"])
    client_seed = base_seed + partition_id

    set_seed(client_seed)

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        seed=client_seed,
    )

    algorithm = str(context.run_config.get("fl-algorithm", "fedavg")).lower()

    if algorithm == "fedavg":
        train_loss = train_fn(
            model,
            trainloader,
            int(context.run_config["local-epochs"]),
            float(msg.content["config"]["lr"]),
            device,
        )
    elif algorithm == "fedsgd":
        train_loss = train_fedsgd_fn(
            model,
            trainloader,
            float(msg.content["config"]["lr"]),
            device,
        )
    else:
        raise ValueError(f"Unknown fl-algorithm: {algorithm}")

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "algorithm": algorithm,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    base_seed = int(context.run_config["seed"])
    client_seed = base_seed + partition_id

    set_seed(client_seed)

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        seed=client_seed,
    )

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)