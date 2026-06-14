"""Federated CIFAR-10 task using a ResNet-18 model adapted for Flower."""

import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


DATALOADER_NUM_WORKERS = 0
PIN_MEMORY = False


class Net(nn.Module):
    """ResNet-18 adapted for CIFAR-10 32x32 RGB images."""

    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None, num_classes=10)

        # Original ResNet-18 is designed for ImageNet 224x224 images.
        # CIFAR-10 images are 32x32, so use a 3x3 stride-1 conv and remove maxpool.
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


fds = None  # Cache FederatedDataset


train_transforms = Compose(
    [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


test_transforms = Compose(
    [
        ToTensor(),
        Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def apply_train_transforms(batch):
    """Apply training transforms to a batch."""
    batch["img"] = [train_transforms(img) for img in batch["img"]]
    return batch


def apply_test_transforms(batch):
    """Apply validation/test transforms to a batch."""
    batch["img"] = [test_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, seed: int):
    """Load one CIFAR-10 client partition and return train/validation loaders."""
    global fds

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    train_dataset = partition_train_test["train"].with_transform(
        apply_train_transforms
    )
    test_dataset = partition_train_test["test"].with_transform(
        apply_test_transforms
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return trainloader, testloader


def load_centralized_dataset():
    """Load the full CIFAR-10 test split and return a dataloader."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    test_dataset = test_dataset.with_transform(apply_test_transforms)

    return DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def train(net, trainloader, epochs, lr, device):
    """Train the model on the local client training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    net.train()
    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

    return running_loss / max(num_batches, 1)


def train_fedsgd(net, trainloader, lr, device, weight_decay=5e-4):
    """Apply one FedSGD-style update using the average local gradient.

    This computes the gradient over the complete local client partition and
    applies a single SGD step. When the server aggregates the resulting
    parameters with FedAvg weighting, this is equivalent to aggregating client
    gradients and applying one global FedSGD update.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

    net.train()
    net.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_examples = 0

    for batch in trainloader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        total_loss += loss.item()
        total_examples += labels.size(0)

    if total_examples == 0:
        return 0.0

    with torch.no_grad():
        for param in net.parameters():
            if param.grad is None:
                continue

            avg_grad = param.grad / total_examples

            if weight_decay > 0:
                avg_grad = avg_grad + weight_decay * param

            param -= lr * avg_grad

    net.zero_grad(set_to_none=True)

    return total_loss / total_examples


def test(net, testloader, device):
    """Evaluate the model on a validation/test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()

    correct = 0
    total = 0
    loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            batch_loss = criterion(outputs, labels)

            loss += batch_loss.item()
            num_batches += 1
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = loss / max(num_batches, 1)
    return avg_loss, accuracy

def apply_update_defense(
    initial_state,
    updated_state,
    trainable_parameter_names,
    defense_type="none",
    clip_norm=0.0,
    noise_std=0.0,
    seed=None,
):
    """Apply clipping/noise to a client model update.

    The defense is applied only to trainable parameters, not to BatchNorm buffers.
    This keeps running statistics untouched while perturbing the actual model update.
    """
    defense_type = str(defense_type).lower()

    if defense_type not in {"none", "clipping", "noise", "clipping_noise"}:
        raise ValueError(f"Unknown defense_type: {defense_type}")

    defended_state = {
        key: value.detach().clone()
        for key, value in updated_state.items()
    }

    deltas = []
    keys = []

    for key in trainable_parameter_names:
        if key not in initial_state or key not in updated_state:
            continue
        if not torch.is_floating_point(updated_state[key]):
            continue

        updated_tensor = updated_state[key].detach()
        initial_tensor = initial_state[key].detach().to(
            device=updated_tensor.device,
            dtype=updated_tensor.dtype,
        )

        delta = updated_tensor - initial_tensor
        deltas.append(delta.reshape(-1))
        keys.append(key)

    if not deltas:
        return defended_state, {
            "update_norm": 0.0,
            "clipping_scale": 1.0,
            "noise_std": float(noise_std if defense_type in {"noise", "clipping_noise"} else 0.0),
        }

    flat_delta = torch.cat(deltas)
    update_norm = torch.norm(flat_delta, p=2).item()

    clipping_scale = 1.0
    if defense_type in {"clipping", "clipping_noise"} and clip_norm > 0:
        clipping_scale = min(1.0, float(clip_norm) / (update_norm + 1e-12))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=flat_delta.device)
        generator.manual_seed(int(seed))

    for key in keys:
        updated_tensor = updated_state[key].detach()
        initial_tensor = initial_state[key].detach().to(
            device=updated_tensor.device,
            dtype=updated_tensor.dtype,
        )

        delta = updated_tensor - initial_tensor
        defended_delta = delta * clipping_scale

        if defense_type in {"noise", "clipping_noise"} and noise_std > 0:
            noise = torch.randn(
                defended_delta.shape,
                generator=generator,
                device=defended_delta.device,
                dtype=defended_delta.dtype,
            ) * float(noise_std)
            defended_delta = defended_delta + noise

        defended_state[key] = initial_tensor + defended_delta

    return defended_state, {
        "update_norm": float(update_norm),
        "clipping_scale": float(clipping_scale),
        "noise_std": float(noise_std if defense_type in {"noise", "clipping_noise"} else 0.0),
    }

