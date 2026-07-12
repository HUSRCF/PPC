#!/usr/bin/env python3
"""Summarize M8 validation-only checkpoints against the locked M7c control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


METRICS = (
    "chain_ap_macro",
    "pr_auc",
    "auroc",
    "f1_best_threshold",
    "mcc_at_best_f1_threshold",
    "chain_ratio_mae_raw",
    "chain_ratio_rmse_raw",
    "chain_ratio_bias_raw",
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/run_dir")
    return name, Path(path_text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name: str, run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = checkpoint.get("val") if isinstance(checkpoint, dict) else None
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(validation, dict) or not isinstance(config, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint lacks config/validation metrics")
    if config.get("selection_metric") != "chain_ap_macro":
        raise ValueError(f"{checkpoint_path}: checkpoint was not selected by chain_ap_macro")
    if bool(config.get("eval_test_each_epoch", True)):
        raise ValueError(f"{checkpoint_path}: test was visible during training")
    row: dict[str, Any] = {
        "name": name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_threshold": float(validation["best_threshold"]),
        "eval_test_each_epoch": False,
    }
    for metric in METRICS:
        value = validation.get(metric)
        row[metric] = None if value is None else float(value)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=_named_path)
    parser.add_argument("--candidate", action="append", required=True, type=_named_path)
    parser.add_argument("--macro-gate", type=float, default=0.005)
    parser.add_argument("--pooled-regression-tolerance", type=float, default=0.003)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load(*args.baseline)
    candidates = [_load(name, path) for name, path in args.candidate]
    for row in candidates:
        row["delta_chain_ap_macro"] = row["chain_ap_macro"] - baseline["chain_ap_macro"]
        row["delta_pr_auc"] = row["pr_auc"] - baseline["pr_auc"]
        row["macro_gate_pass"] = row["delta_chain_ap_macro"] >= args.macro_gate
        row["pooled_guardrail_pass"] = row["delta_pr_auc"] >= -args.pooled_regression_tolerance
        row["advance"] = row["macro_gate_pass"] and row["pooled_guardrail_pass"]
    candidates.sort(key=lambda row: (row["chain_ap_macro"], row["pr_auc"]), reverse=True)

    summary = {
        "protocol": "validation-only screen; chain-macro AP selection; test IDs and loader hidden",
        "macro_gate": args.macro_gate,
        "pooled_regression_tolerance": args.pooled_regression_tolerance,
        "baseline": baseline,
        "candidates": candidates,
        "advancing_candidates": [row["name"] for row in candidates if row["advance"]],
        "best_validation_candidate": candidates[0]["name"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m8_screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    headers = (
        "name",
        "epoch",
        "chain_ap_macro",
        "delta_chain_ap_macro",
        "pr_auc",
        "delta_pr_auc",
        "auroc",
        "f1_best_threshold",
        "mcc_at_best_f1_threshold",
        "chain_ratio_mae_raw",
        "best_threshold",
        "advance",
    )
    with (args.output_dir / "m8_screen_summary.tsv").open("w") as handle:
        handle.write("\t".join(headers) + "\n")
        for row in [baseline, *candidates]:
            handle.write("\t".join(str(row.get(header, "")) for header in headers) + "\n")

    lines = [
        "# M8 Validation-Only Screen",
        "",
        f"- Baseline: `{baseline['name']}`",
        "- Test visibility: hidden; checkpoints are selected only by validation chain-macro AP.",
        f"- Advance gate: macro AP delta >= `{args.macro_gate}` and pooled AP delta >= `-{args.pooled_regression_tolerance}`.",
        "",
        "| Variant | Epoch | Macro AP | Delta | Pooled AP | Delta | Raw ratio MAE | Advance |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
        f"| {baseline['name']} | {baseline['epoch']} | {baseline['chain_ap_macro']:.6f} | - | "
        f"{baseline['pr_auc']:.6f} | - | {baseline.get('chain_ratio_mae_raw')} | control |",
    ]
    for row in candidates:
        ratio = "n/a" if row["chain_ratio_mae_raw"] is None else f"{row['chain_ratio_mae_raw']:.6f}"
        lines.append(
            f"| {row['name']} | {row['epoch']} | {row['chain_ap_macro']:.6f} | "
            f"{row['delta_chain_ap_macro']:+.6f} | {row['pr_auc']:.6f} | {row['delta_pr_auc']:+.6f} | "
            f"{ratio} | {row['advance']} |"
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "m8_screen_summary.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
