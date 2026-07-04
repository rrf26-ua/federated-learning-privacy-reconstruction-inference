from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SAMPLES = [0, 1, 2, 3, 4]

# Usa "final" para la figura principal de memoria.
# También puedes cambiarlo a "best_update" para una versión de apoyo.
RECON_KIND = "final"

BASE_NONE = "results/reconstructions/defense_attack_sample{sample}_none_cosine_iter2000_lr003_seed42"
BASE_DEF = "results/reconstructions/defense_attack_sample{sample}_clipnoise_c7p5_n0p002_cosine_iter2000_lr003_seed42"

OUTPUT_PATH = f"results/figures/reconstruction_comparison_samples0_4_{RECON_KIND}.png"


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe la imagen: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img) / 255.0


def load_metrics(folder: Path):
    path = folder / "attack_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe attack_metrics.json en: {folder}")
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruction_path(folder: Path, recon_kind: str):
    if recon_kind == "final":
        return folder / "reconstructed_final.png"
    if recon_kind == "best_update":
        return folder / "reconstructed_best_update.png"
    if recon_kind == "best_oracle":
        return folder / "reconstructed_best_oracle_mse.png"
    raise ValueError(f"RECON_KIND no soportado: {recon_kind}")


def get_psnr(metrics: dict, recon_kind: str):
    if recon_kind == "final":
        return metrics.get("final_psnr")
    if recon_kind == "best_update":
        return metrics.get("best_update_psnr")
    if recon_kind == "best_oracle":
        return metrics.get("best_oracle_psnr")
    return None


def main():
    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

    fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(10, 3 * len(SAMPLES)))

    col_titles = [
        "Original",
        "Sin defensa",
        "Clipping + ruido\n(C=7.5, σ=0.002)",
    ]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)

    for row, sample in enumerate(SAMPLES):
        none_dir = Path(BASE_NONE.format(sample=sample))
        def_dir = Path(BASE_DEF.format(sample=sample))

        none_metrics = load_metrics(none_dir)
        def_metrics = load_metrics(def_dir)

        label_names = none_metrics.get("label_names") or def_metrics.get("label_names") or ["unknown"]
        label = label_names[0]

        # CLAVE: original desde la carpeta del ataque, no desde torchvision.
        original_img = load_image(none_dir / "original_grid.png")
        none_img = load_image(reconstruction_path(none_dir, RECON_KIND))
        def_img = load_image(reconstruction_path(def_dir, RECON_KIND))

        none_psnr = get_psnr(none_metrics, RECON_KIND)
        def_psnr = get_psnr(def_metrics, RECON_KIND)

        axes[row, 0].imshow(original_img)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(
            f"Sample {sample}\n{label}",
            rotation=0,
            labelpad=42,
            va="center",
            fontsize=10,
        )

        axes[row, 1].imshow(none_img)
        axes[row, 1].axis("off")
        if none_psnr is not None:
            axes[row, 1].text(
                0.5,
                -0.08,
                f"PSNR: {none_psnr:.2f} dB",
                transform=axes[row, 1].transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

        axes[row, 2].imshow(def_img)
        axes[row, 2].axis("off")
        if def_psnr is not None:
            axes[row, 2].text(
                0.5,
                -0.08,
                f"PSNR: {def_psnr:.2f} dB",
                transform=axes[row, 2].transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

    fig.suptitle(
        f"Comparación visual de reconstrucciones ({RECON_KIND})",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
