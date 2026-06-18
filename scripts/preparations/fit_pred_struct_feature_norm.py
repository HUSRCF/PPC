#!/usr/bin/env python3
"""Fit normalization statistics for predicted-structure features on train only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DEFAULT_NUMERIC_KEYS = [
    "pred_plddt",
    "pred_membrane_z",
    "pred_membrane_abs_z",
    "pred_dssp_rsa",
    "pred_dssp_sasa",
    "pred_contact_edge_dist",
]

DEFAULT_CATEGORICAL_KEYS = [
    "pred_tmdet_topology",
    "pred_cctop_topology",
    "pred_dssp_ss",
]


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().cpu().float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        batch_count = int(values.numel())
        batch_mean = float(values.mean().item())
        batch_m2 = float(((values - batch_mean) ** 2).sum().item())
        batch_min = float(values.min().item())
        batch_max = float(values.max().item())
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total
        self.min = min(self.min, batch_min)
        self.max = max(self.max, batch_max)

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "var": None,
                "min": None,
                "max": None,
            }
        var = self.m2 / max(self.count - 1, 1)
        std = var**0.5
        return {
            "count": self.count,
            "mean": self.mean,
            "std": std,
            "var": var,
            "min": self.min,
            "max": self.max,
        }


def read_ids(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, float, bool)) for x in value):
        return torch.tensor(value)
    return None


def update_categorical(counter: Counter[str], value: Any) -> None:
    if isinstance(value, torch.Tensor):
        for item in value.detach().cpu().reshape(-1).tolist():
            counter[str(int(item))] += 1
    elif isinstance(value, (list, tuple)):
        for item in value:
            counter[str(item)] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("features/pred_struct_v1/manifest.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("features/pred_struct_v1/normalization/train_stats.json"))
    parser.add_argument("--numeric-keys", default=",".join(DEFAULT_NUMERIC_KEYS))
    parser.add_argument("--categorical-keys", default=",".join(DEFAULT_CATEGORICAL_KEYS))
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    split_ids = read_ids(args.split_dir / f"{args.split}_ids.txt")
    numeric_keys = [key.strip() for key in args.numeric_keys.split(",") if key.strip()]
    categorical_keys = [key.strip() for key in args.categorical_keys.split(",") if key.strip()]
    stats = {key: RunningStats() for key in numeric_keys}
    categorical = {key: Counter() for key in categorical_keys}

    rows = read_manifest(args.manifest)
    selected = [row for row in rows if row.get("status") == "OK" and (row.get("pdb_id") or "").lower() in split_ids]
    missing_paths = 0
    load_errors: list[dict[str, str]] = []
    n_loaded = 0

    for idx, row in enumerate(selected, start=1):
        feature_path = Path(row.get("feature_path") or "")
        if not feature_path.exists():
            missing_paths += 1
            continue
        try:
            feature = torch.load(feature_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            load_errors.append({"feature_path": str(feature_path), "error": repr(exc)})
            continue
        n_loaded += 1
        for key in numeric_keys:
            tensor = as_tensor(feature.get(key))
            if tensor is not None:
                stats[key].update(tensor)
        for key in categorical_keys:
            update_categorical(categorical[key], feature.get(key))
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(json.dumps({"event": "progress", "seen": idx, "selected": len(selected), "loaded": n_loaded}), flush=True)

    output = {
        "semantics": "fit on train split only; apply these stats to train/val/test without refitting",
        "manifest": str(args.manifest),
        "split_dir": str(args.split_dir),
        "split": args.split,
        "n_manifest_rows": len(rows),
        "n_split_complexes": len(split_ids),
        "n_selected_rows": len(selected),
        "n_loaded_features": n_loaded,
        "missing_paths": missing_paths,
        "n_load_errors": len(load_errors),
        "load_errors_head": load_errors[:10],
        "numeric": {key: value.to_dict() for key, value in stats.items()},
        "categorical_counts": {key: dict(counter.most_common()) for key, counter in categorical.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    tmp_path.replace(args.output)
    print(json.dumps({"event": "wrote_norm_stats", "output": str(args.output), "n_loaded_features": n_loaded}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
