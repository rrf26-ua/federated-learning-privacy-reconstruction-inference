"""Local update inversion attack prototype for CIFAR-10.

This script attacks a simulated one-step local SGD update.

Instead of matching raw gradients directly, it builds an observed local update:

    delta = theta_after - theta_before

for one local SGD step on a private batch. Then it optimizes dummy images so
that their simulated one-step update matches the observed update.

This is a controlled bridge between direct gradient inversion and attacks on
federated client updates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from fl_security_benchmark.task import Net


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def set_seed(seed: int) -> None:
    """Set reproducibility seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Normalize CIFAR-10 image tensor in [0, 1]."""
    mean = CIFAR10_MEAN.to(x.device)
    std = CIFAR10_STD.to(x.device)
    return (x - mean) / std


def mse_value(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute MSE between two tensors."""
    return torch.mean(
        (original.detach().cpu() - reconstructed.detach().cpu()) ** 2
    ).item()


def psnr_value(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute PSNR between two image tensors in [0, 1]."""
    value = mse_value(original, reconstructed)
    if value == 0:
        return float("inf")
    return 20 * math.log10(1.0) - 10 * math.log10(value)


def save_json(path: Path, data: dict) -> None:
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_cifar10_batch(
    sample_start: int,
    batch_size: int,
    split: str,
    device: torch.device,
):
    """Load a consecutive CIFAR-10 batch."""
    dataset = load_dataset("uoft-cs/cifar10", split=split)
    to_tensor = ToTensor()

    images = []
    labels = []
    sample_indices = []

    for idx in range(sample_start, sample_start + batch_size):
        sample = dataset[idx]
        images.append(to_tensor(sample["img"]))
        labels.append(int(sample["label"]))
        sample_indices.append(idx)

    pixels = torch.stack(images, dim=0).to(device)
    normalized = normalize_cifar10(pixels)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    return pixels, normalized, labels_tensor, labels, sample_indices


def select_parameters(model: torch.nn.Module, grad_scope: str):
    """Select parameters used to compute/match the update."""
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    if grad_scope == "all":
        return named_params

    if grad_scope == "fc":
        selected = [
            (name, param)
            for name, param in named_params
            if "fc" in name
        ]
        if not selected:
            raise RuntimeError("No fully-connected layer parameters found.")
        return selected

    raise ValueError(f"Unknown grad_scope: {grad_scope}")


def compute_gradients(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    named_params,
    create_graph: bool,
):
    """Compute gradients of the loss with respect to selected parameters."""
    params = [param for _, param in named_params]
    model.zero_grad(set_to_none=True)

    outputs = model(x)
    loss = criterion(outputs, y)

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph,
    )

    return loss, grads


def gradients_to_sgd_deltas(
    named_params,
    grads,
    local_lr: float,
    weight_decay: float,
):
    """Convert gradients into one-step SGD deltas.

    For one SGD step without previous momentum:

        theta_after = theta_before - lr * (grad + weight_decay * theta_before)
        delta = theta_after - theta_before

    Therefore:

        delta = -lr * (grad + weight_decay * theta_before)

    The parameter term is detached because the attack optimizes the dummy input,
    not the model parameters.
    """
    deltas = []

    for (_, param), grad in zip(named_params, grads):
        if weight_decay > 0:
            delta = -local_lr * (grad + weight_decay * param.detach())
        else:
            delta = -local_lr * grad
        deltas.append(delta)

    return deltas


def infer_label_from_fc_bias_delta(
    named_params,
    observed_deltas,
    local_lr: float,
    weight_decay: float,
):
    """Infer label for batch size 1 from the final layer bias update.

    This recovers the bias gradient from the observed delta and applies the
    iDLG argmin rule. It is meaningful mainly for single-sample gradients.
    """
    for (name, param), delta in zip(named_params, observed_deltas):
        if name.endswith("fc.bias"):
            recovered_grad = -(delta / local_lr)
            if weight_decay > 0:
                recovered_grad = recovered_grad - weight_decay * param.detach()
            return int(torch.argmin(recovered_grad).item())

    return None


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Compute simple total variation regularization."""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def build_per_sample_metrics(
    original_pixels: torch.Tensor,
    reconstructed_pixels: torch.Tensor,
    labels: list[int],
    label_names: list[str],
    sample_indices: list[int],
):
    """Build per-sample MSE/PSNR metrics."""
    rows = []

    for local_idx, sample_idx in enumerate(sample_indices):
        original_one = original_pixels[local_idx : local_idx + 1]
        reconstructed_one = reconstructed_pixels[local_idx : local_idx + 1]

        rows.append(
            {
                "sample_index": sample_idx,
                "true_label": labels[local_idx],
                "true_label_name": label_names[local_idx],
                "mse": mse_value(original_one, reconstructed_one),
                "psnr": psnr_value(original_one, reconstructed_one),
            }
        )

    return rows


def run_attack(args) -> None:
    """Run the controlled local update inversion attack."""
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Sample start: {args.sample_start}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Local LR used to generate update: {args.local_lr}")
    print(f"[INFO] Weight decay used to generate update: {args.weight_decay}")

    (
        original_pixels,
        original_input,
        labels_tensor,
        labels,
        sample_indices,
    ) = load_cifar10_batch(
        sample_start=args.sample_start,
        batch_size=args.batch_size,
        split=args.split,
        device=device,
    )

    label_names = [CIFAR10_CLASSES[label] for label in labels]

    print(f"[INFO] Sample indices: {sample_indices}")
    print(f"[INFO] True labels: {labels}")
    print(f"[INFO] True label names: {label_names}")

    model = Net().to(device)
    model.eval()

    if args.model_path:
        print(f"[INFO] Loading model weights from: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    named_params = select_parameters(model, args.grad_scope)

    print(f"[INFO] Gradient/update scope: {args.grad_scope}")
    print(f"[INFO] Number of parameter tensors used: {len(named_params)}")

    true_loss, true_grads = compute_gradients(
        model=model,
        criterion=criterion,
        x=original_input,
        y=labels_tensor,
        named_params=named_params,
        create_graph=False,
    )

    observed_deltas = gradients_to_sgd_deltas(
        named_params=named_params,
        grads=[grad.detach() for grad in true_grads],
        local_lr=args.local_lr,
        weight_decay=args.weight_decay,
    )
    observed_deltas = [delta.detach() for delta in observed_deltas]

    inferred_label = None
    if args.batch_size == 1:
        inferred_label = infer_label_from_fc_bias_delta(
            named_params=named_params,
            observed_deltas=observed_deltas,
            local_lr=args.local_lr,
            weight_decay=args.weight_decay,
        )
        if inferred_label is not None:
            print(
                f"[INFO] Label inferred from update: "
                f"{inferred_label} ({CIFAR10_CLASSES[inferred_label]})"
            )

    print("[INFO] Using true labels for reconstruction in this controlled update attack.")

    dummy_logits = torch.randn_like(
        original_pixels,
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([dummy_logits], lr=args.attack_lr)

    nrow = args.nrow if args.nrow > 0 else args.batch_size
    save_image(
        original_pixels.detach().cpu(),
        output_dir / "original_grid.png",
        nrow=nrow,
    )

    start_time = time.time()

    best_update_loss = float("inf")
    best_update_reconstructed = None
    best_update_iteration = None

    best_oracle_mse = float("inf")
    best_oracle_reconstructed = None
    best_oracle_iteration = None

    for iteration in range(1, args.iterations + 1):
        reconstructed_pixels = torch.sigmoid(dummy_logits)
        reconstructed_input = normalize_cifar10(reconstructed_pixels)

        _, dummy_grads = compute_gradients(
            model=model,
            criterion=criterion,
            x=reconstructed_input,
            y=labels_tensor,
            named_params=named_params,
            create_graph=True,
        )

        predicted_deltas = gradients_to_sgd_deltas(
            named_params=named_params,
            grads=dummy_grads,
            local_lr=args.local_lr,
            weight_decay=args.weight_decay,
        )

        update_loss = torch.zeros((), device=device)
        for predicted_delta, observed_delta in zip(predicted_deltas, observed_deltas):
            update_loss = update_loss + F.mse_loss(predicted_delta, observed_delta)

        tv_loss = total_variation(reconstructed_pixels)
        total_loss = update_loss + args.tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        current_update_loss = float(update_loss.detach().cpu().item())
        current_mse = mse_value(original_pixels, reconstructed_pixels)

        if current_update_loss < best_update_loss:
            best_update_loss = current_update_loss
            best_update_reconstructed = reconstructed_pixels.detach().clone()
            best_update_iteration = iteration

        if current_mse < best_oracle_mse:
            best_oracle_mse = current_mse
            best_oracle_reconstructed = reconstructed_pixels.detach().clone()
            best_oracle_iteration = iteration

        if iteration % args.log_every == 0 or iteration == 1:
            current_psnr = psnr_value(original_pixels, reconstructed_pixels)
            print(
                f"[ITER {iteration:04d}] "
                f"update_loss={current_update_loss:.6e} "
                f"tv={float(tv_loss.detach().cpu().item()):.6e} "
                f"mse={current_mse:.6e} "
                f"psnr={current_psnr:.2f}"
            )

        if iteration % args.save_every == 0:
            save_image(
                reconstructed_pixels.detach().cpu(),
                output_dir / f"reconstructed_iter_{iteration:04d}.png",
                nrow=nrow,
            )

    elapsed = time.time() - start_time

    final_reconstructed = torch.sigmoid(dummy_logits).detach()

    if best_update_reconstructed is None:
        best_update_reconstructed = final_reconstructed
        best_update_iteration = args.iterations

    if best_oracle_reconstructed is None:
        best_oracle_reconstructed = final_reconstructed
        best_oracle_mse = mse_value(original_pixels, final_reconstructed)
        best_oracle_iteration = args.iterations

    save_image(
        best_update_reconstructed.detach().cpu(),
        output_dir / "reconstructed_best_update.png",
        nrow=nrow,
    )

    save_image(
        best_oracle_reconstructed.detach().cpu(),
        output_dir / "reconstructed_best_oracle_mse.png",
        nrow=nrow,
    )

    save_image(
        final_reconstructed.detach().cpu(),
        output_dir / "reconstructed_final.png",
        nrow=nrow,
    )

    # Backward-compatible name used by previous experiments.
    save_image(
        best_update_reconstructed.detach().cpu(),
        output_dir / "reconstructed_grid.png",
        nrow=nrow,
    )

    per_sample_best_update = build_per_sample_metrics(
        original_pixels=original_pixels,
        reconstructed_pixels=best_update_reconstructed,
        labels=labels,
        label_names=label_names,
        sample_indices=sample_indices,
    )

    per_sample_best_oracle = build_per_sample_metrics(
        original_pixels=original_pixels,
        reconstructed_pixels=best_oracle_reconstructed,
        labels=labels,
        label_names=label_names,
        sample_indices=sample_indices,
    )

    per_sample_final = build_per_sample_metrics(
        original_pixels=original_pixels,
        reconstructed_pixels=final_reconstructed,
        labels=labels,
        label_names=label_names,
        sample_indices=sample_indices,
    )

    metrics = {
        "attack": "local_update_inversion",
        "dataset": "CIFAR-10",
        "split": args.split,
        "sample_start": args.sample_start,
        "sample_indices": sample_indices,
        "batch_size": args.batch_size,
        "model": "ResNet-18 adapted for CIFAR-10",
        "model_path": args.model_path,
        "grad_scope": args.grad_scope,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "tv_weight": args.tv_weight,
        "local_lr": args.local_lr,
        "weight_decay": args.weight_decay,
        "labels": labels,
        "label_names": label_names,
        "uses_true_labels": True,
        "inferred_label_from_update": inferred_label,
        "inferred_label_from_update_name": (
            CIFAR10_CLASSES[inferred_label] if inferred_label is not None else None
        ),
        "true_loss": float(true_loss.detach().cpu().item()),
        "best_update_loss": best_update_loss,
        "best_update_iteration": best_update_iteration,
        "best_update_mse": mse_value(original_pixels, best_update_reconstructed),
        "best_update_psnr": psnr_value(original_pixels, best_update_reconstructed),
        "best_oracle_iteration": best_oracle_iteration,
        "best_oracle_mse": best_oracle_mse,
        "best_oracle_psnr": psnr_value(original_pixels, best_oracle_reconstructed),
        "final_mse": mse_value(original_pixels, final_reconstructed),
        "final_psnr": psnr_value(original_pixels, final_reconstructed),
        "per_sample": per_sample_best_update,
        "per_sample_best_update": per_sample_best_update,
        "per_sample_best_oracle": per_sample_best_oracle,
        "per_sample_final": per_sample_final,
        "elapsed_seconds": elapsed,
        "device": str(device),
    }

    save_json(output_dir / "attack_metrics.json", metrics)

    print("[INFO] Local update inversion attack finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Best-update MSE: {metrics['best_update_mse']:.6e}")
    print(f"[INFO] Best-update PSNR: {metrics['best_update_psnr']:.2f}")
    print(f"[INFO] Best-oracle MSE: {metrics['best_oracle_mse']:.6e}")
    print(f"[INFO] Best-oracle PSNR: {metrics['best_oracle_psnr']:.2f}")
    print(f"[INFO] Final MSE: {metrics['final_mse']:.6e}")
    print(f"[INFO] Final PSNR: {metrics['final_psnr']:.2f}")
    print(f"[INFO] Elapsed seconds: {elapsed:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled local update inversion attack on CIFAR-10."
    )

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--attack-lr", type=float, default=0.03)
    parser.add_argument("--tv-weight", type=float, default=0.001)
    parser.add_argument("--local-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--grad-scope", type=str, default="all", choices=["fc", "all"])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--nrow", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reconstructions/local_update_bs1",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
