#!/usr/bin/env python3
"""Run one reproducible FedSGD configuration and append its summary CSV row."""

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "experiment_summaries" / "fedsgd_sweep_summary.csv"
SUMMARY_COLUMNS = [
    "config_id",
    "rounds",
    "clients",
    "batch_size",
    "lr",
    "weight_decay",
    "scheduler",
    "final_loss",
    "final_accuracy",
    "best_accuracy",
    "elapsed_time",
    "command",
]
EVALUATION_RE = re.compile(
    r"Server-side evaluation round (\d+): loss=([^,]+), accuracy=([^\s]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--scheduler-min-lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-with-server-round", action="store_true")
    parser.add_argument("--client-cpus", type=int, default=1)
    parser.add_argument("--client-gpus", type=float, default=0.166)
    parser.add_argument("--init-cpus", type=int, default=8)
    parser.add_argument("--init-gpus", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds < 1 or args.clients < 1 or args.batch_size < 1:
        raise ValueError("rounds, clients and batch-size must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay non-negative")
    if not 0 <= args.scheduler_min_lr <= args.lr:
        raise ValueError("scheduler-min-lr must satisfy 0 <= value <= lr")


def build_command(args: argparse.Namespace) -> list[str]:
    run_config = " ".join(
        [
            'fl-algorithm="fedsgd"',
            'fl-strategy="fedavg"',
            f"num-server-rounds={args.rounds}",
            f"seed={args.seed}",
            f"learning-rate={args.lr}",
            f"weight-decay={args.weight_decay}",
            f'scheduler="{args.scheduler}"',
            f"scheduler-min-lr={args.scheduler_min_lr}",
            f"seed-with-server-round={str(args.seed_with_server_round).lower()}",
            f"batch-size={args.batch_size}",
            "fraction-train=1.0",
            "fraction-evaluate=0.0",
            'defense-type="none"',
            "save-final-model=false",
        ]
    )
    federation_config = " ".join(
        [
            f"num-supernodes={args.clients}",
            f"client-resources-num-cpus={args.client_cpus}",
            f"client-resources-num-gpus={args.client_gpus}",
            f"init-args-num-cpus={args.init_cpus}",
            f"init-args-num-gpus={args.init_gpus}",
        ]
    )
    return [
        "flwr",
        "run",
        ".",
        "--stream",
        "--run-config",
        run_config,
        "--federation-config",
        federation_config,
    ]


def unique_log_path(config_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_id).strip("._")
    if not safe_id:
        safe_id = "fedsgd"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RAW_DIR / f"fedsgd_sweep_{safe_id}_{timestamp}_{uuid4().hex[:8]}.log"


def run_and_log(command: list[str], log_path: Path) -> tuple[int, float]:
    start = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log_file:
        log_file.write(f"[SWEEP] Command: {shlex.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return_code = process.wait()
    return return_code, time.monotonic() - start


def extract_metrics(log_path: Path) -> tuple[str, str, str]:
    evaluations = []
    for match in EVALUATION_RE.finditer(log_path.read_text(encoding="utf-8")):
        evaluations.append(
            (int(match.group(1)), float(match.group(2)), float(match.group(3)))
        )
    completed_rounds = [entry for entry in evaluations if entry[0] > 0]
    if not completed_rounds:
        return "", "", ""
    _, final_loss, final_accuracy = completed_rounds[-1]
    best_accuracy = max(entry[2] for entry in completed_rounds)
    return str(final_loss), str(final_accuracy), str(best_accuracy)


def append_summary(
    args: argparse.Namespace,
    command: list[str],
    log_path: Path,
    elapsed: float,
) -> None:
    final_loss, final_accuracy, best_accuracy = extract_metrics(log_path)
    write_header = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SUMMARY_COLUMNS,
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "config_id": args.config_id,
                "rounds": args.rounds,
                "clients": args.clients,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
                "best_accuracy": best_accuracy,
                "elapsed_time": f"{elapsed:.2f}",
                "command": shlex.join(command),
            }
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(args)
    log_path = unique_log_path(args.config_id)

    print(f"[INFO] Log: {log_path.relative_to(ROOT)}")
    print(f"[INFO] Command: {shlex.join(command)}")
    return_code, elapsed = run_and_log(command, log_path)
    append_summary(args, command, log_path, elapsed)
    print(f"[INFO] Summary: {SUMMARY_PATH.relative_to(ROOT)}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
