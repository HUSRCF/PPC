#!/usr/bin/env python3
"""Summarize paired-seed M7 control/candidate validation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


METRICS = ("chain_ap_macro", "pr_auc", "f1_best_threshold", "mcc_at_best_f1_threshold")


def parse_seed_path(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Use SEED=/path/to/run_dir")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed {seed_text!r}") from error
    return seed, Path(path_text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("val"), dict):
        raise ValueError(f"Checkpoint has no validation metrics: {path}")
    config = checkpoint.get("config") or {}
    if config.get("selection_metric") != "chain_ap_macro":
        raise ValueError(f"Checkpoint was not selected by chain_ap_macro: {path}")
    validation = checkpoint["val"]
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "best_threshold": float(validation["best_threshold"]),
    }
    for metric in METRICS:
        row[metric] = float(validation[metric])
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="append", required=True, type=parse_seed_path)
    parser.add_argument("--candidate", action="append", required=True, type=parse_seed_path)
    parser.add_argument("--control-name", default="M7a")
    parser.add_argument("--candidate-name", default="M7c")
    parser.add_argument("--macro-gate", type=float, default=0.005)
    parser.add_argument("--pooled-regression-tolerance", type=float, default=0.002)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    control_paths = dict(args.control)
    candidate_paths = dict(args.candidate)
    if len(control_paths) != len(args.control) or len(candidate_paths) != len(args.candidate):
        raise ValueError("Duplicate seed in control or candidate arguments")
    if set(control_paths) != set(candidate_paths):
        raise ValueError("Control and candidate seed sets differ")
    seeds = sorted(control_paths)

    paired_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for seed in seeds:
        control = load_checkpoint(control_paths[seed])
        candidate = load_checkpoint(candidate_paths[seed])
        control_rows.append(control)
        candidate_rows.append(candidate)
        paired_rows.append(
            {
                "seed": seed,
                "control": control,
                "candidate": candidate,
                "delta_chain_ap_macro": candidate["chain_ap_macro"] - control["chain_ap_macro"],
                "delta_pr_auc": candidate["pr_auc"] - control["pr_auc"],
                "delta_f1_best_threshold": candidate["f1_best_threshold"] - control["f1_best_threshold"],
            }
        )

    control_aggregate = aggregate(control_rows)
    candidate_aggregate = aggregate(candidate_rows)
    macro_deltas = np.asarray([row["delta_chain_ap_macro"] for row in paired_rows], dtype=np.float64)
    pooled_deltas = np.asarray([row["delta_pr_auc"] for row in paired_rows], dtype=np.float64)
    mean_macro_delta = float(macro_deltas.mean())
    mean_pooled_delta = float(pooled_deltas.mean())
    gate_checks = {
        "mean_macro_delta_at_least_gate": mean_macro_delta >= args.macro_gate,
        "mean_pooled_delta_within_tolerance": mean_pooled_delta >= -args.pooled_regression_tolerance,
    }
    promote = all(gate_checks.values())
    summary = {
        "protocol": "paired seeds; validation chain-macro-AP checkpoints; test hidden",
        "control_name": args.control_name,
        "candidate_name": args.candidate_name,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "macro_gate": args.macro_gate,
        "pooled_regression_tolerance": args.pooled_regression_tolerance,
        "control": control_aggregate,
        "candidate": candidate_aggregate,
        "paired": paired_rows,
        "paired_delta": {
            "chain_ap_macro_mean": mean_macro_delta,
            "chain_ap_macro_std": float(macro_deltas.std(ddof=1)) if macro_deltas.size > 1 else 0.0,
            "chain_ap_macro_positive_seeds": int((macro_deltas > 0).sum()),
            "pr_auc_mean": mean_pooled_delta,
            "pr_auc_std": float(pooled_deltas.std(ddof=1)) if pooled_deltas.size > 1 else 0.0,
            "pr_auc_positive_seeds": int((pooled_deltas > 0).sum()),
        },
        "gate_checks": gate_checks,
        "promote_candidate": promote,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m7_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "m7_confirmation_per_seed.tsv").open("w") as handle:
        handle.write(
            "seed\tcontrol_macro_ap\tcandidate_macro_ap\tdelta_macro_ap\t"
            "control_pr_auc\tcandidate_pr_auc\tdelta_pr_auc\n"
        )
        for row in paired_rows:
            handle.write(
                f"{row['seed']}\t{row['control']['chain_ap_macro']}\t"
                f"{row['candidate']['chain_ap_macro']}\t{row['delta_chain_ap_macro']}\t"
                f"{row['control']['pr_auc']}\t{row['candidate']['pr_auc']}\t{row['delta_pr_auc']}\n"
            )

    markdown = f"""# M7 Paired-Seed Confirmation

- Control / candidate: `{args.control_name}` / `{args.candidate_name}`
- Seeds: `{', '.join(str(seed) for seed in seeds)}`
- Test visibility: hidden

| Metric | Control mean +/- SD | Candidate mean +/- SD | Paired mean delta |
|---|---:|---:|---:|
| Chain-macro AP | {control_aggregate['chain_ap_macro']['mean']:.6f} +/- {control_aggregate['chain_ap_macro']['std']:.6f} | {candidate_aggregate['chain_ap_macro']['mean']:.6f} +/- {candidate_aggregate['chain_ap_macro']['std']:.6f} | {mean_macro_delta:+.6f} |
| Pooled AP | {control_aggregate['pr_auc']['mean']:.6f} +/- {control_aggregate['pr_auc']['std']:.6f} | {candidate_aggregate['pr_auc']['mean']:.6f} +/- {candidate_aggregate['pr_auc']['std']:.6f} | {mean_pooled_delta:+.6f} |

- Macro-AP-positive seeds: `{int((macro_deltas > 0).sum())}/{len(seeds)}`
- Pooled-AP-positive seeds: `{int((pooled_deltas > 0).sum())}/{len(seeds)}`
- Promote candidate: `{promote}`
"""
    (args.output_dir / "m7_confirmation_summary.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
