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
    "top_20pct_precision_micro",
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


def _read_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out[Path(row["strict_output_csv"]).name] = row
    return out


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def collect(paths: list[Path], manifest: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = _read_manifest(manifest)
    rows: list[dict[str, Any]] = []
    for path in paths:
        manifest_row = manifest_rows.get(path.name, {})
        for row in _read_csv(path):
            if row.get("split") != "test":
                continue
            run_name = manifest_row.get("run_name") or _run_name(path, row)
            model_short = manifest_row.get("model_short") or run_name
            metric_note = manifest_row.get("metric_scope_note") or "strict"
            out: dict[str, Any] = {
                "run_name": run_name,
                "model_short": model_short,
                "metric_scope_note": metric_note,
                "summary_csv": str(path),
                "metric_scope": row.get("metric_scope", "test_at_val_threshold"),
                "val_selected_threshold": _float_or_none(row.get("val_selected_threshold")),
            }
            for key in METRIC_COLUMNS:
                out[key] = _float_or_none(row.get(key))
            rows.append(out)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model_short"]), str(row["metric_scope_note"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (model, metric_note), vals in sorted(grouped.items()):
        item: dict[str, Any] = {"model_short": model, "metric_scope_note": metric_note, "n": len(vals)}
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
    parser.add_argument("--manifest", type=Path, default=Path("results/strict_eval/strict_eval_manifest.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/strict_eval"))
    args = parser.parse_args()

    rows, summary = collect(args.summary_csv, args.manifest)
    write_csv(args.output_dir / "strict_test_at_val_threshold_rows.csv", rows)
    write_csv(args.output_dir / "strict_test_at_val_threshold_summary.csv", summary)
    print(json.dumps({"n_rows": len(rows), "n_groups": len(summary), "output_dir": str(args.output_dir)}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
