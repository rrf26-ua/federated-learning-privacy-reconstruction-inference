"""Multi-step local update inversion attack prototype for CIFAR-10.

This script attacks a simulated multi-step local SGD update.

Instead of matching raw gradients directly, it builds an observed local update:

    delta = theta_after_k_steps - theta_before

where theta_after_k_steps is obtained after several local SGD steps on the
same private batch. Then it optimizes dummy images so that their simulated
multi-step update matches the observed update.

This is a controlled experiment between one-step local update inversion and
more realistic FedAvg local training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.func import functional_call
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
    """Compute MSE between two image tensors."""
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


def get_base_params(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Clone model parameters as differentiable base parameters."""
    return OrderedDict(
        (name, param.detach().clone().requires_grad_(True))
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def get_buffers(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Get model buffers for functional_call."""
    return OrderedDict(
        (name, buffer.detach().clone())
        for name, buffer in model.named_buffers()
    )


def select_parameter_names(base_params: OrderedDict[str, torch.Tensor], grad_scope: str):
    """Select parameter names used for matching the update."""
    if grad_scope == "all":
        return list(base_params.keys())

    if grad_scope == "fc":
        selected = [name for name in base_params.keys() if "fc" in name]
        if not selected:
            raise RuntimeError("No fully-connected layer parameters found.")
        return selected

    raise ValueError(f"Unknown grad_scope: {grad_scope}")


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Compute simple total variation regularization."""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def simulate_multistep_sgd_update(
    model: torch.nn.Module,
    base_params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    local_steps: int,
    local_lr: float,
    weight_decay: float,
    create_graph: bool,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    """Simulate several local SGD steps functionally.

    The returned delta is:

        theta_after_local_steps - theta_before

    This function is differentiable with respect to x when create_graph=True.
    """
    params = OrderedDict((name, value) for name, value in base_params.items())
    last_loss = None

    for _ in range(local_steps):
        state = OrderedDict()
        state.update(params)
        state.update(buffers)

        outputs = functional_call(model, state, (x,))
        loss = criterion(outputs, y)
        last_loss = loss

        grads = torch.autograd.grad(
            loss,
            tuple(params.values()),
            create_graph=create_graph,
            retain_graph=create_graph,
        )

        next_params = OrderedDict()

        for (name, param), grad in zip(params.items(), grads):
            if weight_decay > 0:
                update_direction = grad + weight_decay * param
            else:
                update_direction = grad

            next_params[name] = param - local_lr * update_direction

        params = next_params

    deltas = OrderedDict(
        (name, params[name] - base_params[name])
        for name in base_params.keys()
    )

    if last_loss is None:
        raise RuntimeError("local_steps must be >= 1")

    return last_loss, deltas


def infer_label_from_fc_bias_delta(
    base_params: OrderedDict[str, torch.Tensor],
    observed_deltas: OrderedDict[str, torch.Tensor],
    local_lr: float,
    weight_decay: float,
):
    """Infer label from final layer bias update for the one-step bs=1 case.

    This is exact mainly for local_steps=1 and batch_size=1. For multi-step
    updates, the relationship between update and raw gradient is no longer
    equivalent, so this function should not be used as a general multi-step
    label inference method.
    """
    for name, delta in observed_deltas.items():
        if name.endswith("fc.bias"):
            recovered_grad = -(delta / local_lr)
            if weight_decay > 0:
                recovered_grad = recovered_grad - weight_decay * base_params[name].detach()
            return int(torch.argmin(recovered_grad).item())

    return None


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


def clear_base_param_grads(base_params: OrderedDict[str, torch.Tensor]) -> None:
    """Avoid accumulating unused gradients on base parameter leaves."""
    for param in base_params.values():
        param.grad = None


def run_attack(args) -> None:
    """Run the controlled multi-step local update inversion attack."""
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
    print(f"[INFO] Local steps: {args.local_steps}")
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

    base_params = get_base_params(model)
    buffers = get_buffers(model)
    selected_names = select_parameter_names(base_params, args.grad_scope)

    print(f"[INFO] Gradient/update scope: {args.grad_scope}")
    print(f"[INFO] Number of parameter tensors matched: {len(selected_names)}")

    true_loss, observed_deltas = simulate_multistep_sgd_update(
        model=model,
        base_params=base_params,
        buffers=buffers,
        x=original_input,
        y=labels_tensor,
        criterion=criterion,
        local_steps=args.local_steps,
        local_lr=args.local_lr,
        weight_decay=args.weight_decay,
        create_graph=False,
    )

    observed_deltas = OrderedDict(
        (name, delta.detach())
        for name, delta in observed_deltas.items()
    )

    inferred_label = None
    if args.batch_size == 1 and args.local_steps == 1:
        inferred_label = infer_label_from_fc_bias_delta(
            base_params=base_params,
            observed_deltas=observed_deltas,
            local_lr=args.local_lr,
            weight_decay=args.weight_decay,
        )
        if inferred_label is not None:
            print(
                f"[INFO] Label inferred from one-step update: "
                f"{inferred_label} ({CIFAR10_CLASSES[inferred_label]})"
            )
    else:
        print("[INFO] Label inference from update is skipped for multi-step/batch updates.")

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

        _, predicted_deltas = simulate_multistep_sgd_update(
            model=model,
            base_params=base_params,
            buffers=buffers,
            x=reconstructed_input,
            y=labels_tensor,
            criterion=criterion,
            local_steps=args.local_steps,
            local_lr=args.local_lr,
            weight_decay=args.weight_decay,
            create_graph=True,
        )

        update_loss = torch.zeros((), device=device)

        for name in selected_names:
            update_loss = update_loss + F.mse_loss(
                predicted_deltas[name],
                observed_deltas[name],
            )

        tv_loss = total_variation(reconstructed_pixels)
        total_loss = update_loss + args.tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        clear_base_param_grads(base_params)

        total_loss.backward()
        optimizer.step()

        clear_base_param_grads(base_params)

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
        "attack": "local_update_multistep_inversion",
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
        "local_steps": args.local_steps,
        "local_lr": args.local_lr,
        "weight_decay": args.weight_decay,
        "labels": labels,
        "label_names": label_names,
        "uses_true_labels": True,
        "inferred_label_from_update": inferred_label,
        "inferred_label_from_update_name": (
            CIFAR10_CLASSES[inferred_label] if inferred_label is not None else None
        ),
        "true_loss_after_last_local_step": float(true_loss.detach().cpu().item()),
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

    print("[INFO] Multi-step local update inversion attack finished.")
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
        description="Controlled multi-step local update inversion attack on CIFAR-10."
    )

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--attack-lr", type=float, default=0.03)
    parser.add_argument("--tv-weight", type=float, default=0.000001)
    parser.add_argument("--local-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--local-steps", type=int, default=1)
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
        default="results/reconstructions/local_update_multistep_bs1",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
