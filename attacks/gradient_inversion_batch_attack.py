"""Batch gradient inversion attack prototype for CIFAR-10.

This script extends the controlled gradient inversion attack to batches with
more than one image. It is used to study how reconstruction quality changes
when the observed gradient comes from several samples instead of a single one.

For this first controlled experiment, true labels are used during the attack.
This isolates visual reconstruction leakage from the harder problem of label
inference in multi-sample batches.
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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    mean = CIFAR10_MEAN.to(x.device)
    std = CIFAR10_STD.to(x.device)
    return (x - mean) / std


def mse_value(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return torch.mean((original.detach().cpu() - reconstructed.detach().cpu()) ** 2).item()


def psnr_value(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    value = mse_value(original, reconstructed)
    if value == 0:
        return float("inf")
    return 20 * math.log10(1.0) - 10 * math.log10(value)


def load_cifar10_batch(sample_start: int, batch_size: int, split: str, device: torch.device):
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
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    return pixels, normalized, label_tensor, labels, sample_indices


def select_parameters(model: torch.nn.Module, grad_scope: str):
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    if grad_scope == "all":
        return named_params

    if grad_scope == "fc":
        selected = [(name, param) for name, param in named_params if "fc" in name]
        if not selected:
            raise RuntimeError("No fully-connected layer parameters found.")
        return selected

    raise ValueError(f"Unknown grad_scope: {grad_scope}")


def compute_gradients(model, criterion, x, y, named_params, create_graph: bool):
    params = [param for _, param in named_params]
    model.zero_grad(set_to_none=True)
    outputs = model(x)
    loss = criterion(outputs, y)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return loss, grads


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_attack(args):
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Sample start: {args.sample_start}")
    print(f"[INFO] Batch size: {args.batch_size}")

    original_pixels, original_input, labels_tensor, labels, sample_indices = load_cifar10_batch(
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

    print(f"[INFO] Gradient scope: {args.grad_scope}")
    print(f"[INFO] Number of parameter tensors used: {len(named_params)}")
    print("[INFO] Using true labels for this controlled batch reconstruction attack.")

    true_loss, true_grads = compute_gradients(
        model=model,
        criterion=criterion,
        x=original_input,
        y=labels_tensor,
        named_params=named_params,
        create_graph=False,
    )
    true_grads = [grad.detach() for grad in true_grads]

    dummy_logits = torch.randn_like(original_pixels, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_logits], lr=args.attack_lr)

    nrow = args.nrow if args.nrow > 0 else args.batch_size
    save_image(original_pixels.detach().cpu(), output_dir / "original_grid.png", nrow=nrow)

    start_time = time.time()
    best_grad_loss = float("inf")
    best_reconstructed = None

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

        grad_loss = torch.zeros((), device=device)
        for dummy_grad, true_grad in zip(dummy_grads, true_grads):
            grad_loss = grad_loss + F.mse_loss(dummy_grad, true_grad)

        tv_loss = total_variation(reconstructed_pixels)
        total_loss = grad_loss + args.tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        current_grad_loss = float(grad_loss.detach().cpu().item())

        if current_grad_loss < best_grad_loss:
            best_grad_loss = current_grad_loss
            best_reconstructed = reconstructed_pixels.detach().clone()

        if iteration % args.log_every == 0 or iteration == 1:
            current_mse = mse_value(original_pixels, reconstructed_pixels)
            current_psnr = psnr_value(original_pixels, reconstructed_pixels)
            print(
                f"[ITER {iteration:04d}] "
                f"grad_loss={current_grad_loss:.6e} "
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

    if best_reconstructed is None:
        best_reconstructed = torch.sigmoid(dummy_logits).detach()

    save_image(best_reconstructed.detach().cpu(), output_dir / "reconstructed_grid.png", nrow=nrow)

    per_sample = []
    for local_idx, sample_idx in enumerate(sample_indices):
        original_one = original_pixels[local_idx : local_idx + 1]
        reconstructed_one = best_reconstructed[local_idx : local_idx + 1]
        per_sample.append(
            {
                "sample_index": sample_idx,
                "true_label": labels[local_idx],
                "true_label_name": label_names[local_idx],
                "mse": mse_value(original_one, reconstructed_one),
                "psnr": psnr_value(original_one, reconstructed_one),
            }
        )

    metrics = {
        "attack": "batch_gradient_inversion",
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
        "labels": labels,
        "label_names": label_names,
        "uses_true_labels": True,
        "true_loss": float(true_loss.detach().cpu().item()),
        "best_grad_loss": best_grad_loss,
        "final_mse": mse_value(original_pixels, best_reconstructed),
        "final_psnr": psnr_value(original_pixels, best_reconstructed),
        "per_sample": per_sample,
        "elapsed_seconds": elapsed,
        "device": str(device),
    }

    save_json(output_dir / "attack_metrics.json", metrics)

    print("[INFO] Batch attack finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Final MSE: {metrics['final_mse']:.6e}")
    print(f"[INFO] Final PSNR: {metrics['final_psnr']:.2f}")
    print(f"[INFO] Elapsed seconds: {elapsed:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled batch gradient inversion attack on CIFAR-10.")

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--attack-lr", type=float, default=0.03)
    parser.add_argument("--tv-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--grad-scope", type=str, default="all", choices=["fc", "all"])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--nrow", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/reconstructions/batch_attack_bs2")

    return parser.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
