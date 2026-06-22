#!/usr/bin/env python3
"""Summarize val/test metrics using the validation-selected threshold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _pick_threshold_item(metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    sweep = metrics.get("threshold_sweep") or []
    if not sweep:
        return {}
    return min(sweep, key=lambda item: abs(float(item["threshold"]) - float(threshold)))


def _row(split: str, metrics: dict[str, Any], val_threshold: float) -> dict[str, Any]:
    at_val = _pick_threshold_item(metrics, val_threshold)
    return {
        "split": split,
        "metric_scope": "validation" if split == "val" else "test_at_val_threshold",
        "val_selected_threshold": val_threshold,
        "threshold_used": at_val.get("threshold"),
        "f1_at_0_5": metrics.get("f1_at_0_5", metrics.get("f1")),
        "mcc_at_0_5": metrics.get("mcc_at_0_5", metrics.get("mcc")),
        "f1_at_val_threshold": at_val.get("f1"),
        "precision_at_val_threshold": at_val.get("precision"),
        "recall_at_val_threshold": at_val.get("recall"),
        "mcc_at_val_threshold": at_val.get("mcc"),
        "auroc": metrics.get("auroc"),
        "pr_auc": metrics.get("pr_auc"),
        "diagnostic_best_threshold": metrics.get("best_threshold"),
        "diagnostic_f1_best_threshold": metrics.get("f1_best_threshold"),
        "diagnostic_best_mcc_threshold": metrics.get("best_mcc_threshold"),
        "diagnostic_mcc_best_threshold": metrics.get("mcc_best_threshold"),
        "top_5pct_precision_micro": metrics.get("top_5pct_precision_micro"),
        "top_5pct_enrichment_micro": metrics.get("top_5pct_enrichment_micro"),
        "top_10pct_precision_micro": metrics.get("top_10pct_precision_micro"),
        "top_10pct_enrichment_micro": metrics.get("top_10pct_enrichment_micro"),
        "top_20pct_precision_micro": metrics.get("top_20pct_precision_micro"),
        "top_20pct_enrichment_micro": metrics.get("top_20pct_enrichment_micro"),
        "loss": metrics.get("loss"),
        "skipped_nonfinite": metrics.get("skipped_nonfinite"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--val-threshold", type=float, default=None)
    args = parser.parse_args()

    data = json.loads(args.eval_json.read_text())
    splits = data.get("splits", {})
    if "val" not in splits:
        raise ValueError("eval JSON must contain a val split")
    val_threshold = args.val_threshold
    if val_threshold is None:
        val_threshold = float(splits["val"]["best_threshold"])

    rows = [_row(split, metrics, val_threshold) for split, metrics in splits.items()]
    summary = {
        "eval_json": str(args.eval_json),
        "checkpoint": data.get("checkpoint"),
        "config": data.get("config"),
        "val_selected_threshold": val_threshold,
        "rows": rows,
    }

    output_json = args.output_json or args.eval_json.with_name(args.eval_json.stem + "_strict_summary.json")
    output_csv = args.output_csv or args.eval_json.with_name(args.eval_json.stem + "_strict_summary.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "strict_summary", "output_json": str(output_json), "output_csv": str(output_csv), "val_selected_threshold": val_threshold}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
