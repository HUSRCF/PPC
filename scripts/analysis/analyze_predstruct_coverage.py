#!/usr/bin/env python3
"""Summarize predicted-structure feature coverage buckets from sample tags."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


BUCKETS = [
    ("eq0", lambda x: x == 0.0),
    ("gt0_lt0p5", lambda x: 0.0 < x < 0.5),
    ("ge0p5", lambda x: x >= 0.5),
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    out: list[dict[str, Any]] = []
    for name, predicate in BUCKETS:
        subset = [r for r in rows if predicate(_float(r.get("predseq_coverage_residue")))]
        n = len(subset)
        positives = [int(_float(r.get("n_positive"))) for r in subset]
        residues = [int(_float(r.get("n_residues"))) for r in subset]
        pos_frac = [_float(r.get("positive_fraction")) for r in subset]
        out.append(
            {
                "coverage_bucket": name,
                "n_complexes": n,
                "n_residues": sum(residues),
                "n_positive": sum(positives),
                "mean_positive_fraction": mean(pos_frac) if pos_frac else None,
                "usable_for_predstruct_scalar": sum(1 for r in subset if _bool(r.get("usable_for_predstruct_scalar"))),
                "mean_coverage": mean([_float(r.get("predseq_coverage_residue")) for r in subset]) if subset else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-tags", type=Path, default=Path("exports/ppc_metadata_release_20260619/tags/sample_tags.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/predstruct_coverage_buckets.csv"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize(args.sample_tags)
    write_csv(args.output_csv, rows)
    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(output_json), "rows": rows}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
