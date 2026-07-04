"""Summarize gradient inversion attack metrics into a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "sample_index",
    "split",
    "model",
    "grad_scope",
    "iterations",
    "attack_lr",
    "tv_weight",
    "true_label",
    "true_label_name",
    "inferred_label",
    "inferred_label_name",
    "attack_label",
    "attack_label_name",
    "final_mse",
    "final_psnr",
    "best_grad_loss",
    "elapsed_seconds",
    "device",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_files = sorted(results_dir.glob("sample_*/attack_metrics.json"))

    if not metrics_files:
        raise SystemExit(f"No attack_metrics.json files found under {results_dir}")

    output_path = Path(args.output) if args.output else results_dir / "summary.csv"

    rows = []
    for metrics_file in metrics_files:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append({field: data.get(field) for field in FIELDS})

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
