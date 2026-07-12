#!/usr/bin/env python3
"""Aggregate completed M6 seed summaries into mean and standard deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "test_f1_frozen": ("test", "selected_threshold", "f1"),
    "test_mcc_frozen": ("test", "selected_threshold", "mcc"),
    "test_average_precision": ("test", "average_precision"),
    "test_auprc": ("test", "auprc"),
    "test_auroc": ("test", "auroc"),
    "test_chain_ap_macro": ("test", "chain_ap_macro"),
    "test_ratio_mae_raw": ("test", "chain_effect_site_ratio_mae_raw"),
    "test_ratio_mae_thresholded": ("test", "chain_effect_site_ratio_mae_thresholded"),
    "test_L20_ACC_macro": ("test", "L/20_ACC_macro"),
    "test_L10_ACC_macro": ("test", "L/10_ACC_macro"),
    "test_L5_ACC_macro": ("test", "L/5_ACC_macro"),
}


def nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing summary field: {'.'.join(keys)}")
        value = value[key]
    return value


def parse_seed_summary(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Use SEED=/path/to/m6_summary.json")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed {seed_text!r}") from error
    return seed, Path(path_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-summary", action="append", required=True, type=parse_seed_summary)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    entries = sorted(args.seed_summary)
    seeds = [seed for seed, _ in entries]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds: {seeds}")
    if len(entries) < 2:
        raise ValueError("At least two seed summaries are required")

    rows: list[dict[str, Any]] = []
    for seed, path in entries:
        payload = json.loads(path.read_text())
        row: dict[str, Any] = {
            "seed": seed,
            "summary": str(path),
            "alpha_m0": float(payload["selected_alpha_m0"]),
            "alpha_m2": float(payload["selected_alpha_m2"]),
            "validation_threshold": float(payload["selected_validation_threshold"]),
        }
        for metric, keys in METRICS.items():
            row[metric] = float(nested_get(payload, keys))
        rows.append(row)

    aggregate: dict[str, dict[str, float]] = {}
    for metric in ("alpha_m0", "alpha_m2", "validation_threshold", *METRICS):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "method": "m6_validation_f1_selected_multiseed",
        "selection_policy": (
            "Each seed independently uses the historical M6 validation-F1 checkpoint, "
            "validation-only logit alpha, and validation-only threshold"
        ),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "per_seed": rows,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m6_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    columns = ("seed", "alpha_m2", "validation_threshold", *METRICS)
    with (args.output_dir / "m6_multiseed_summary.tsv").open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")

    metric_lines = []
    for metric in METRICS:
        values = aggregate[metric]
        metric_lines.append(f"| {metric} | {values['mean']:.6f} | {values['std']:.6f} |")
    markdown = f"""# M6 Multi-Seed Summary

- Seeds: `{', '.join(str(seed) for seed in seeds)}`
- Protocol: historical M6 validation-F1 checkpoint and validation-only blend/threshold per seed

| Metric | Mean | Sample SD |
|---|---:|---:|
{chr(10).join(metric_lines)}
"""
    (args.output_dir / "m6_multiseed_summary.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
