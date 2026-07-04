"""Gradient inversion attack prototype for CIFAR-10.

This script performs a controlled reconstruction attack outside Flower.

It takes one CIFAR-10 image, computes the gradients produced by that image and
label on the current model, and then optimizes a dummy image so that its
gradients match the observed gradients.

This is the first controlled step before attacking real federated updates.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from attacks.metrics import mse, psnr, save_json
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
    """Set deterministic seeds as far as possible."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor in [0, 1] using CIFAR-10 statistics."""
    mean = CIFAR10_MEAN.to(x.device)
    std = CIFAR10_STD.to(x.device)
    return (x - mean) / std


def load_cifar10_sample(sample_index: int, split: str, device: torch.device):
    """Load one CIFAR-10 sample as both raw pixels and normalized input."""
    dataset = load_dataset("uoft-cs/cifar10", split=split)
    sample = dataset[sample_index]

    image = sample["img"]
    label = int(sample["label"])

    pixels = ToTensor()(image).unsqueeze(0).to(device)  # [1, 3, 32, 32], in [0, 1]
    normalized = normalize_cifar10(pixels)

    return pixels, normalized, torch.tensor([label], device=device), label


def select_parameters(model: torch.nn.Module, grad_scope: str):
    """Select which model parameters are used for gradient matching."""
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    if grad_scope == "all":
        return named_params

    if grad_scope == "fc":
        selected = [
            (name, param)
            for name, param in named_params
            if "fc" in name
        ]
        if not selected:
            raise RuntimeError("No fully-connected layer parameters found for grad_scope='fc'.")
        return selected

    raise ValueError(f"Unknown grad_scope: {grad_scope}")


def infer_label_from_last_bias(named_grads):
    """Infer label using the iDLG observation from the final layer bias gradient."""
    for name, grad in named_grads:
        if name.endswith("fc.bias"):
            return int(torch.argmin(grad).item())
    return None


def compute_gradients(model, criterion, x, y, named_params, create_graph: bool):
    """Compute gradients of loss with respect to selected model parameters."""
    params = [param for _, param in named_params]
    model.zero_grad(set_to_none=True)
    outputs = model(x)
    loss = criterion(outputs, y)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return loss, grads


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Simple total variation regularization on image pixels."""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def run_attack(args):
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading CIFAR-10 sample index={args.sample_index}, split={args.split}")

    original_pixels, original_input, true_label_tensor, true_label = load_cifar10_sample(
        sample_index=args.sample_index,
        split=args.split,
        device=device,
    )

    print(f"[INFO] True label: {true_label} ({CIFAR10_CLASSES[true_label]})")

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

    true_loss, true_grads = compute_gradients(
        model=model,
        criterion=criterion,
        x=original_input,
        y=true_label_tensor,
        named_params=named_params,
        create_graph=False,
    )

    true_grads = [grad.detach() for grad in true_grads]
    named_true_grads = [(name, grad) for (name, _), grad in zip(named_params, true_grads)]

    inferred_label = infer_label_from_last_bias(named_true_grads)
    if inferred_label is not None:
        print(f"[INFO] iDLG inferred label: {inferred_label} ({CIFAR10_CLASSES[inferred_label]})")
    else:
        print("[WARN] Could not infer label from final bias gradient. Using true label.")

    attack_label = true_label if args.use_true_label else inferred_label
    if attack_label is None:
        attack_label = true_label

    attack_label_tensor = torch.tensor([attack_label], device=device)

    # Optimize unconstrained logits. sigmoid(logits) gives reconstructed pixels in [0, 1].
    dummy_logits = torch.randn_like(original_pixels, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([dummy_logits], lr=args.attack_lr)

    save_image(original_pixels.detach().cpu(), output_dir / "original.png")

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
            y=attack_label_tensor,
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
            current_mse = mse(original_pixels, reconstructed_pixels)
            current_psnr = psnr(original_pixels, reconstructed_pixels)
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
            )

    elapsed = time.time() - start_time

    if best_reconstructed is None:
        best_reconstructed = torch.sigmoid(dummy_logits).detach()

    save_image(best_reconstructed.detach().cpu(), output_dir / "reconstructed.png")

    metrics = {
        "attack": "gradient_inversion",
        "dataset": "CIFAR-10",
        "split": args.split,
        "sample_index": args.sample_index,
        "model": "ResNet-18 adapted for CIFAR-10",
        "model_path": args.model_path,
        "grad_scope": args.grad_scope,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "tv_weight": args.tv_weight,
        "true_label": true_label,
        "true_label_name": CIFAR10_CLASSES[true_label],
        "inferred_label": inferred_label,
        "inferred_label_name": CIFAR10_CLASSES[inferred_label] if inferred_label is not None else None,
        "attack_label": attack_label,
        "attack_label_name": CIFAR10_CLASSES[attack_label],
        "true_loss": float(true_loss.detach().cpu().item()),
        "best_grad_loss": best_grad_loss,
        "final_mse": mse(original_pixels, best_reconstructed),
        "final_psnr": psnr(original_pixels, best_reconstructed),
        "elapsed_seconds": elapsed,
        "device": str(device),
    }

    save_json(output_dir / "attack_metrics.json", metrics)

    print("[INFO] Attack finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Final MSE: {metrics['final_mse']:.6e}")
    print(f"[INFO] Final PSNR: {metrics['final_psnr']:.2f}")
    print(f"[INFO] Elapsed seconds: {elapsed:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled gradient inversion attack on CIFAR-10.")

    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--grad-scope", type=str, default="fc", choices=["fc", "all"])
    parser.add_argument("--use-true-label", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/reconstructions/gradient_inversion_sample0")

    return parser.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
