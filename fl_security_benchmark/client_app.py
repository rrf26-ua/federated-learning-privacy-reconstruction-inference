"""fl_security_benchmark: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_security_benchmark.task import (
    Net,
    apply_update_defense,
    fedsgd_learning_rate,
    load_data,
)
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

    server_round = int(msg.content["config"].get("server-round", 0))
    seed_with_server_round = bool(
        context.run_config.get("seed-with-server-round", False)
    )
    training_seed = client_seed
    if seed_with_server_round:
        training_seed += server_round * num_partitions

    # The partition split remains tied to client_seed below. Optionally changing
    # this process-wide seed only varies stochastic training operations by round.
    set_seed(training_seed)

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    initial_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    trainable_parameter_names = {
        name for name, _ in model.named_parameters()
    }

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
        learning_rate = fedsgd_learning_rate(
            base_lr=float(msg.content["config"]["lr"]),
            server_round=server_round,
            num_rounds=int(context.run_config["num-server-rounds"]),
            scheduler=str(context.run_config.get("scheduler", "none")),
            min_lr=float(context.run_config.get("scheduler-min-lr", 0.0)),
        )
        train_loss = train_fedsgd_fn(
            model,
            trainloader,
            learning_rate,
            device,
            weight_decay=float(context.run_config.get("weight-decay", 5e-4)),
        )
    else:
        raise ValueError(f"Unknown fl-algorithm: {algorithm}")

    defense_type = str(context.run_config.get("defense-type", "none")).lower()
    defense_clip_norm = float(context.run_config.get("defense-clip-norm", 0.0))
    defense_noise_std = float(context.run_config.get("defense-noise-std", 0.0))
    defense_seed = client_seed + 10_000

    defended_state, defense_metrics = apply_update_defense(
        initial_state=initial_state,
        updated_state=model.state_dict(),
        trainable_parameter_names=trainable_parameter_names,
        defense_type=defense_type,
        clip_norm=defense_clip_norm,
        noise_std=defense_noise_std,
        seed=defense_seed,
    )

    model_record = ArrayRecord(defended_state)
    metrics = {
        "train_loss": train_loss,
        "learning_rate": (
            learning_rate
            if algorithm == "fedsgd"
            else float(msg.content["config"]["lr"])
        ),
        "num-examples": len(trainloader.dataset),
        "update_norm": defense_metrics["update_norm"],
        "clipping_scale": defense_metrics["clipping_scale"],
        "noise_std": defense_metrics["noise_std"],
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
