#!/usr/bin/env python3
"""Collect strict val/test summaries into one test-at-validation-threshold table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_COLUMNS = [
    "f1_at_val_threshold",
    "precision_at_val_threshold",
    "recall_at_val_threshold",
    "mcc_at_val_threshold",
    "auroc",
    "pr_auc",
    "top_5pct_precision_micro",
    "top_10pct_precision_micro",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _run_name(summary_csv: Path, row: dict[str, str]) -> str:
    json_path = summary_csv.with_suffix(".json")
    if json_path.exists():
        data = _read_json(json_path)
        checkpoint = data.get("checkpoint") or ""
        if checkpoint:
            return str(Path(checkpoint).parent.name)
    return row.get("run_name") or summary_csv.parent.name


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def collect(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_csv(path):
            if row.get("split") != "test":
                continue
            out: dict[str, Any] = {
                "run_name": _run_name(path, row),
                "summary_csv": str(path),
                "metric_scope": row.get("metric_scope", "test_at_val_threshold"),
                "val_selected_threshold": _float_or_none(row.get("val_selected_threshold")),
            }
            for key in METRIC_COLUMNS:
                out[key] = _float_or_none(row.get(key))
            rows.append(out)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        # Drop seed suffixes only when the run naming follows the project convention.
        name = str(row["run_name"])
        model = name
        for token in ("_seed42_", "_seed43_", "_seed44_"):
            if token in model:
                model = model.replace(token, "_seedXX_")
        grouped.setdefault(model, []).append(row)

    summary: list[dict[str, Any]] = []
    for model, vals in sorted(grouped.items()):
        item: dict[str, Any] = {"model_group": model, "n": len(vals)}
        for key in METRIC_COLUMNS:
            xs = [float(v[key]) for v in vals if v.get(key) is not None]
            item[f"{key}_mean"] = mean(xs) if xs else None
            item[f"{key}_std"] = pstdev(xs) if len(xs) > 1 else 0.0
        summary.append(item)
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/strict_eval"))
    args = parser.parse_args()

    rows, summary = collect(args.summary_csv)
    write_csv(args.output_dir / "strict_test_at_val_threshold_rows.csv", rows)
    write_csv(args.output_dir / "strict_test_at_val_threshold_summary.csv", summary)
    print(json.dumps({"n_rows": len(rows), "n_groups": len(summary), "output_dir": str(args.output_dir)}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
