#!/usr/bin/env python3
"""Summarize validation metrics stored in named training checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_CHECKPOINTS = (
    "best.pt",
    "best_f1_best_threshold.pt",
    "best_pr_auc.pt",
    "best_chain_ap_macro.pt",
)


def load_row(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("val"), dict):
        raise ValueError(f"Checkpoint lacks validation metrics: {path}")
    validation = checkpoint["val"]
    return {
        "run": run_dir.name,
        "checkpoint": filename,
        "epoch": checkpoint.get("epoch"),
        "f1_best_threshold": validation.get("f1_best_threshold"),
        "best_threshold": validation.get("best_threshold"),
        "pr_auc": validation.get("pr_auc"),
        "chain_ap_macro": validation.get("chain_ap_macro"),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--checkpoints", default=",".join(DEFAULT_CHECKPOINTS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    filenames = [value.strip() for value in args.checkpoints.split(",") if value.strip()]
    if not filenames:
        raise ValueError("At least one checkpoint filename is required")
    rows = [load_row(run_dir, filename) for run_dir in args.runs for filename in filenames]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    columns = (
        "run",
        "checkpoint",
        "epoch",
        "f1_best_threshold",
        "best_threshold",
        "pr_auc",
        "chain_ap_macro",
    )
    widths = {
        column: max(len(column), *(len(format_value(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_value(row[column]).ljust(widths[column]) for column in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
