from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from fl_security_benchmark.task import Net


CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BN-aware inversion attack on a real Flower client update."
    )

    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--attack-lr", type=float, default=0.03)
    parser.add_argument("--tv-weight", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--init-image",
        type=Path,
        default=None,
        help="Optional PNG/JPG image used to initialize the dummy pixels.",
    )
    parser.add_argument(
        "--init-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise added to init image pixels before inversion.",
    )

    parser.add_argument("--lambda-param", type=float, default=1.0)
    parser.add_argument("--lambda-bn-mean", type=float, default=1.0)
    parser.add_argument("--lambda-bn-var", type=float, default=1.0)

    parser.add_argument(
        "--bn-loss",
        choices=["cosine", "mse_normalized"],
        default="cosine",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs"],
        default="adam",
    )
    parser.add_argument("--lbfgs-history-size", type=int, default=100)
    parser.add_argument("--lbfgs-max-iter-per-step", type=int, default=1)
    parser.add_argument(
        "--lbfgs-line-search",
        choices=["none", "strong_wolfe"],
        default="strong_wolfe",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_cifar10(x_pixels: torch.Tensor) -> torch.Tensor:
    mean = CIFAR10_MEAN.to(x_pixels.device)
    std = CIFAR10_STD.to(x_pixels.device)
    return (x_pixels - mean) / std


def denormalize_cifar10(x_normalized: torch.Tensor) -> torch.Tensor:
    mean = CIFAR10_MEAN.to(x_normalized.device)
    std = CIFAR10_STD.to(x_normalized.device)
    return (x_normalized * std + mean).clamp(0.0, 1.0)


def image_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x.detach().cpu(), y.detach().cpu()).item())


def image_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = image_mse(x, y)
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def cosine_loss_flat(predicted: list[torch.Tensor], observed: list[torch.Tensor]) -> torch.Tensor:
    p = flatten_tensors(predicted)
    o = flatten_tensors(observed)
    return 1.0 - F.cosine_similarity(p, o, dim=0, eps=1e-12)


def mse_normalized_loss(predicted: list[torch.Tensor], observed: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    for p, o in zip(predicted, observed):
        p_norm = p / (p.norm() + 1e-12)
        o_norm = o / (o.norm() + 1e-12)
        losses.append(F.mse_loss(p_norm, o_norm))
    return torch.stack(losses).mean()


def grouped_loss(
    predicted: list[torch.Tensor],
    observed: list[torch.Tensor],
    kind: str,
) -> torch.Tensor:
    if not predicted:
        return torch.zeros((), device=observed[0].device if observed else "cpu")

    if kind == "cosine":
        return cosine_loss_flat(predicted, observed)

    if kind == "mse_normalized":
        return mse_normalized_loss(predicted, observed)

    raise ValueError(f"Unknown loss kind: {kind}")


def sgd_param_deltas(
    named_params,
    grads,
    lr: float,
    weight_decay: float,
) -> list[torch.Tensor]:
    deltas = []
    for (_, param), grad in zip(named_params, grads):
        if weight_decay > 0:
            deltas.append(-lr * (grad + weight_decay * param.detach()))
        else:
            deltas.append(-lr * grad)
    return deltas


def load_private_batch(capture_dir: Path, device: torch.device):
    batch = torch_load(capture_dir / "private_batch.pt", map_location=device)

    img = batch["img"].float().to(device)
    label = batch["label"].long().view(-1).to(device)

    # Current capture stores normalized CIFAR-10 tensors.
    if float(img.min().detach().cpu()) >= 0.0 and float(img.max().detach().cpu()) <= 1.0:
        pixels = img.detach()
        normalized = normalize_cifar10(pixels)
        storage_format = "pixels"
    else:
        normalized = img.detach()
        pixels = denormalize_cifar10(normalized)
        storage_format = "normalized"

    return pixels, normalized, label, storage_format




def load_init_image(path: Path, device: torch.device, target_shape: torch.Size) -> torch.Tensor:
    """Load an initialization image as pixels in [0, 1]."""
    image = Image.open(path).convert("RGB")
    pixels = ToTensor()(image).unsqueeze(0).to(device)

    if pixels.shape != target_shape:
        raise ValueError(
            f"Init image has shape {tuple(pixels.shape)}, "
            f"but target shape is {tuple(target_shape)}. "
            "Use a single reconstructed 32x32 image, not a comparison grid."
        )

    return pixels.clamp(0.0, 1.0)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map pixels in [0,1] to unconstrained logits."""
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def get_bn_modules(model: torch.nn.Module):
    modules = {}
    for name, module in model.named_modules():
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ),
        ):
            modules[name] = module
    return modules


def make_bn_input_hooks(model: torch.nn.Module):
    bn_inputs = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            bn_inputs[name] = inputs[0]
        return hook

    for name, module in get_bn_modules(model).items():
        handles.append(module.register_forward_hook(make_hook(name)))

    return bn_inputs, handles


def bn_factor(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], name: str) -> float:
    if module.momentum is not None:
        return float(module.momentum)

    key = f"{name}.num_batches_tracked"
    if key in state_dict:
        old_batches = int(state_dict[key].detach().cpu().item())
        return 1.0 / float(old_batches + 1)

    return 0.1


def batch_channel_stats(x: torch.Tensor):
    if x.dim() == 2:
        dims = [0]
    elif x.dim() == 3:
        dims = [0, 2]
    elif x.dim() == 4:
        dims = [0, 2, 3]
    elif x.dim() == 5:
        dims = [0, 2, 3, 4]
    else:
        raise ValueError(f"Unsupported BN input dimension: {x.shape}")

    mean = x.mean(dim=dims)

    count = 1
    for dim in dims:
        count *= x.shape[dim]

    if count > 1:
        var = x.var(dim=dims, unbiased=True)
    else:
        var = x.var(dim=dims, unbiased=False)

    return mean, var


def simulate_signals(
    *,
    initial_state: dict[str, torch.Tensor],
    x_pixels: torch.Tensor,
    label: torch.Tensor,
    lr: float,
    weight_decay: float,
    device: torch.device,
):
    model = Net().to(device)
    model.load_state_dict(initial_state, strict=True)
    model.train()

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    bn_inputs, handles = make_bn_input_hooks(model)

    x_normalized = normalize_cifar10(x_pixels)
    output = model(x_normalized)
    loss = F.cross_entropy(output, label)

    grads = torch.autograd.grad(
        loss,
        [param for _, param in named_params],
        create_graph=True,
        retain_graph=True,
    )

    param_deltas = sgd_param_deltas(
        named_params=named_params,
        grads=grads,
        lr=lr,
        weight_decay=weight_decay,
    )

    bn_mean_deltas = {}
    bn_var_deltas = {}

    for name, module in get_bn_modules(model).items():
        if name not in bn_inputs:
            continue

        running_mean_key = f"{name}.running_mean"
        running_var_key = f"{name}.running_var"

        if running_mean_key not in initial_state or running_var_key not in initial_state:
            continue

        activation = bn_inputs[name]
        batch_mean, batch_var = batch_channel_stats(activation)

        old_mean = initial_state[running_mean_key].to(device)
        old_var = initial_state[running_var_key].to(device)

        factor = bn_factor(module, initial_state, name)

        bn_mean_deltas[running_mean_key] = factor * (batch_mean - old_mean)
        bn_var_deltas[running_var_key] = factor * (batch_var - old_var)

    for handle in handles:
        handle.remove()

    return {
        "loss": loss,
        "named_params": named_params,
        "param_deltas": param_deltas,
        "bn_mean_deltas": bn_mean_deltas,
        "bn_var_deltas": bn_var_deltas,
    }


def load_observed_groups(
    observed_delta: dict[str, torch.Tensor],
    named_params,
    device: torch.device,
):
    observed_param = []
    missing_param = []

    for name, _ in named_params:
        if name not in observed_delta:
            missing_param.append(name)
        else:
            observed_param.append(observed_delta[name].detach().to(device))

    if missing_param:
        raise KeyError(
            "Missing parameter deltas: "
            + ", ".join(missing_param[:10])
            + (" ..." if len(missing_param) > 10 else "")
        )

    observed_bn_mean = {}
    observed_bn_var = {}

    for key, value in observed_delta.items():
        if not torch.is_floating_point(value):
            continue

        if key.endswith("running_mean"):
            observed_bn_mean[key] = value.detach().to(device)

        if key.endswith("running_var"):
            observed_bn_var[key] = value.detach().to(device)

    return observed_param, observed_bn_mean, observed_bn_var


def align_dict_values(predicted: dict[str, torch.Tensor], observed: dict[str, torch.Tensor]):
    pred_values = []
    obs_values = []
    missing = []

    for key in sorted(observed.keys()):
        if key not in predicted:
            missing.append(key)
            continue
        pred_values.append(predicted[key])
        obs_values.append(observed[key])

    if missing:
        raise KeyError(
            "Missing predicted BN deltas: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    return pred_values, obs_values


def norm_of_list(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    return float(flatten_tensors([v.detach().cpu() for v in values]).norm().item())


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    capture_dir = args.capture_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Capture dir: {capture_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))

    lr = float(metadata.get("learning_rate", 0.02))
    weight_decay = float(metadata.get("weight_decay", 0.0005))

    initial_state_cpu = torch_load(capture_dir / "global_before.pt", map_location="cpu")
    observed_delta_cpu = torch_load(capture_dir / "observed_delta.pt", map_location="cpu")

    initial_state = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in initial_state_cpu.items()
    }

    observed_delta = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in observed_delta_cpu.items()
    }

    target_pixels, target_normalized, target_label, storage_format = load_private_batch(
        capture_dir=capture_dir,
        device=device,
    )

    print(f"[INFO] Private batch storage format: {storage_format}")
    print(f"[INFO] Target label: {target_label.detach().cpu().tolist()}")
    print(f"[INFO] LR: {lr}")
    print(f"[INFO] Weight decay: {weight_decay}")

    # One dry run with the true image. This validates whether the simulated
    # BN-aware operator is aligned with the captured real update.
    true_signals = simulate_signals(
        initial_state=initial_state,
        x_pixels=target_pixels,
        label=target_label,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    observed_param, observed_bn_mean, observed_bn_var = load_observed_groups(
        observed_delta=observed_delta,
        named_params=true_signals["named_params"],
        device=device,
    )

    pred_bn_mean_true, obs_bn_mean = align_dict_values(
        true_signals["bn_mean_deltas"],
        observed_bn_mean,
    )
    pred_bn_var_true, obs_bn_var = align_dict_values(
        true_signals["bn_var_deltas"],
        observed_bn_var,
    )

    true_param_loss = cosine_loss_flat(
        true_signals["param_deltas"],
        observed_param,
    )
    true_bn_mean_loss = grouped_loss(
        pred_bn_mean_true,
        obs_bn_mean,
        args.bn_loss,
    )
    true_bn_var_loss = grouped_loss(
        pred_bn_var_true,
        obs_bn_var,
        args.bn_loss,
    )

    print("[SANITY TRUE IMAGE]")
    print(f"  true_param_cosine_loss: {float(true_param_loss.detach().cpu().item()):.10f}")
    print(f"  true_bn_mean_{args.bn_loss}_loss: {float(true_bn_mean_loss.detach().cpu().item()):.10f}")
    print(f"  true_bn_var_{args.bn_loss}_loss: {float(true_bn_var_loss.detach().cpu().item()):.10f}")
    print(f"  observed_param_norm: {norm_of_list(observed_param):.6f}")
    print(f"  observed_bn_mean_norm: {norm_of_list(obs_bn_mean):.6f}")
    print(f"  observed_bn_var_norm: {norm_of_list(obs_bn_var):.6f}")

    save_image(target_pixels.detach().cpu(), output_dir / "original_grid.png")

    if args.init_image is not None:
        init_pixels = load_init_image(
            path=args.init_image,
            device=device,
            target_shape=target_pixels.shape,
        )

        if args.init_noise_std > 0:
            init_pixels = (
                init_pixels
                + torch.randn_like(init_pixels) * float(args.init_noise_std)
            ).clamp(0.0, 1.0)

        init_mse = image_mse(init_pixels, target_pixels)
        init_psnr = image_psnr(init_pixels, target_pixels)

        print(f"[INFO] Init image: {args.init_image}")
        print(f"[INFO] Init image MSE: {init_mse:.8f}")
        print(f"[INFO] Init image PSNR: {init_psnr:.2f}")

        raw = inverse_sigmoid(init_pixels).detach().clone().requires_grad_(True)
    else:
        raw = torch.randn_like(target_pixels, device=device, requires_grad=True)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([raw], lr=args.attack_lr)
    elif args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [raw],
            lr=args.attack_lr,
            max_iter=args.lbfgs_max_iter_per_step,
            history_size=args.lbfgs_history_size,
            line_search_fn=(
                None if args.lbfgs_line_search == "none" else args.lbfgs_line_search
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    best_loss = float("inf")
    best_loss_pixels = None
    best_loss_iteration = None

    best_oracle_mse = float("inf")
    best_oracle_pixels = None
    best_oracle_iteration = None

    history = []
    start_time = time.time()

    def compute_loss_bundle(current_raw: torch.Tensor):
        dummy_pixels = torch.sigmoid(current_raw)

        signals = simulate_signals(
            initial_state=initial_state,
            x_pixels=dummy_pixels,
            label=target_label,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        pred_bn_mean, local_obs_bn_mean = align_dict_values(
            signals["bn_mean_deltas"],
            observed_bn_mean,
        )
        pred_bn_var, local_obs_bn_var = align_dict_values(
            signals["bn_var_deltas"],
            observed_bn_var,
        )

        param_loss = cosine_loss_flat(
            signals["param_deltas"],
            observed_param,
        )

        bn_mean_loss = grouped_loss(
            pred_bn_mean,
            local_obs_bn_mean,
            args.bn_loss,
        )

        bn_var_loss = grouped_loss(
            pred_bn_var,
            local_obs_bn_var,
            args.bn_loss,
        )

        tv_loss = total_variation(dummy_pixels)

        total_loss = (
            args.lambda_param * param_loss
            + args.lambda_bn_mean * bn_mean_loss
            + args.lambda_bn_var * bn_var_loss
            + args.tv_weight * tv_loss
        )

        return {
            "dummy_pixels": dummy_pixels,
            "total_loss": total_loss,
            "param_loss": param_loss,
            "bn_mean_loss": bn_mean_loss,
            "bn_var_loss": bn_var_loss,
            "tv_loss": tv_loss,
        }

    for iteration in range(1, args.iterations + 1):
        if args.optimizer == "adam":
            optimizer.zero_grad(set_to_none=True)
            bundle = compute_loss_bundle(raw)
            bundle["total_loss"].backward()
            optimizer.step()

        else:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                closure_bundle = compute_loss_bundle(raw)
                closure_bundle["total_loss"].backward()
                return closure_bundle["total_loss"]

            optimizer.step(closure)

            # Recompute after the LBFGS step for logging/saving.
            bundle = compute_loss_bundle(raw)

        with torch.no_grad():
            current_pixels = torch.sigmoid(raw).detach()
            current_mse = image_mse(current_pixels, target_pixels)
            current_psnr = image_psnr(current_pixels, target_pixels)
            current_total = float(bundle["total_loss"].detach().cpu().item())

            if current_total < best_loss:
                best_loss = current_total
                best_loss_pixels = current_pixels.clone()
                best_loss_iteration = iteration

            if current_mse < best_oracle_mse:
                best_oracle_mse = current_mse
                best_oracle_pixels = current_pixels.clone()
                best_oracle_iteration = iteration

            row = {
                "iteration": iteration,
                "total_loss": current_total,
                "param_loss": float(bundle["param_loss"].detach().cpu().item()),
                "bn_mean_loss": float(bundle["bn_mean_loss"].detach().cpu().item()),
                "bn_var_loss": float(bundle["bn_var_loss"].detach().cpu().item()),
                "tv_loss": float(bundle["tv_loss"].detach().cpu().item()),
                "mse": current_mse,
                "psnr": current_psnr,
            }
            history.append(row)

            if iteration % args.log_every == 0 or iteration == 1:
                print(
                    f"[ITER {iteration:04d}/{args.iterations}] "
                    f"total={row['total_loss']:.6e} "
                    f"param={row['param_loss']:.6e} "
                    f"bn_mean={row['bn_mean_loss']:.6e} "
                    f"bn_var={row['bn_var_loss']:.6e} "
                    f"tv={row['tv_loss']:.6e} "
                    f"mse={row['mse']:.6e} "
                    f"psnr={row['psnr']:.2f}"
                )

            if iteration % args.save_every == 0:
                save_image(
                    current_pixels.detach().cpu(),
                    output_dir / f"reconstructed_iter_{iteration:04d}.png",
                )

    final_pixels = torch.sigmoid(raw).detach()

    if best_loss_pixels is None:
        best_loss_pixels = final_pixels.clone()
        best_loss_iteration = args.iterations

    if best_oracle_pixels is None:
        best_oracle_pixels = final_pixels.clone()
        best_oracle_iteration = args.iterations

    save_image(final_pixels.detach().cpu(), output_dir / "reconstructed_final.png")
    save_image(best_loss_pixels.detach().cpu(), output_dir / "reconstructed_best_loss.png")
    save_image(best_oracle_pixels.detach().cpu(), output_dir / "reconstructed_best_oracle.png")

    comparison = torch.cat(
        [
            target_pixels.detach().cpu(),
            best_loss_pixels.detach().cpu(),
            best_oracle_pixels.detach().cpu(),
            final_pixels.detach().cpu(),
        ],
        dim=0,
    )
    save_image(comparison, output_dir / "comparison_original_bestloss_bestoracle_final.png", nrow=4)

    final_mse = image_mse(final_pixels, target_pixels)
    final_psnr = image_psnr(final_pixels, target_pixels)
    best_loss_psnr = image_psnr(best_loss_pixels, target_pixels)
    best_oracle_psnr = image_psnr(best_oracle_pixels, target_pixels)

    summary = {
        "attack": "real_update_inversion_bn_aware",
        "capture_dir": str(capture_dir),
        "output_dir": str(output_dir),
        "metadata": metadata,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "tv_weight": args.tv_weight,
        "optimizer": args.optimizer,
        "lbfgs_history_size": args.lbfgs_history_size,
        "lbfgs_max_iter_per_step": args.lbfgs_max_iter_per_step,
        "lbfgs_line_search": args.lbfgs_line_search,
        "init_image": str(args.init_image) if args.init_image is not None else None,
        "init_noise_std": args.init_noise_std,
        "lambda_param": args.lambda_param,
        "lambda_bn_mean": args.lambda_bn_mean,
        "lambda_bn_var": args.lambda_bn_var,
        "bn_loss": args.bn_loss,
        "seed": args.seed,
        "target_label": target_label.detach().cpu().tolist(),
        "private_batch_storage_format": storage_format,
        "true_param_cosine_loss": float(true_param_loss.detach().cpu().item()),
        f"true_bn_mean_{args.bn_loss}_loss": float(true_bn_mean_loss.detach().cpu().item()),
        f"true_bn_var_{args.bn_loss}_loss": float(true_bn_var_loss.detach().cpu().item()),
        "observed_param_norm": norm_of_list(observed_param),
        "observed_bn_mean_norm": norm_of_list(obs_bn_mean),
        "observed_bn_var_norm": norm_of_list(obs_bn_var),
        "best_loss": best_loss,
        "best_loss_iteration": best_loss_iteration,
        "best_loss_mse": image_mse(best_loss_pixels, target_pixels),
        "best_loss_psnr": best_loss_psnr,
        "best_oracle_iteration": best_oracle_iteration,
        "best_oracle_mse": best_oracle_mse,
        "best_oracle_psnr": best_oracle_psnr,
        "final_mse": final_mse,
        "final_psnr": final_psnr,
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }

    save_json(output_dir / "attack_metrics.json", summary)

    print("[RESULT]")
    for key in [
        "true_param_cosine_loss",
        f"true_bn_mean_{args.bn_loss}_loss",
        f"true_bn_var_{args.bn_loss}_loss",
        "observed_param_norm",
        "observed_bn_mean_norm",
        "observed_bn_var_norm",
        "best_loss",
        "best_loss_psnr",
        "best_oracle_psnr",
        "final_psnr",
        "elapsed_seconds",
    ]:
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
