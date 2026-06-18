#!/usr/bin/env python3
"""Fit PLC-v4 spatial scalar normalization stats on train split only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_features_v4 import fit_normalization_stats, get_spatial_scalar_feature_names_v4


def read_ids(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("features/pred_struct_complete_v1/manifest.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("features/pred_struct_complete_v1/normalization/train_spatial_scalar_stats.json"))
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    split_ids = read_ids(args.split_dir / f"{args.split}_ids.txt")
    rows = read_manifest(args.manifest)
    selected = [row for row in rows if row.get("status") == "OK" and (row.get("pdb_id") or "").lower() in split_ids]

    raw_list: list[np.ndarray] = []
    missing_paths = 0
    load_errors: list[dict[str, str]] = []
    for idx, row in enumerate(selected, start=1):
        path = Path(row.get("feature_path") or "")
        if not path.exists():
            missing_paths += 1
            continue
        try:
            obj: dict[str, Any] = torch.load(path, map_location="cpu")
            raw = obj["pred_spatial_scalar_raw_features"]
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().numpy()
            raw = np.asarray(raw, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != 89:
                raise ValueError(f"bad raw scalar shape {raw.shape}")
            raw_list.append(raw)
        except Exception as exc:  # noqa: BLE001
            load_errors.append({"feature_path": str(path), "error": repr(exc)})
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(json.dumps({"event": "progress", "seen": idx, "selected": len(selected), "loaded": len(raw_list)}), flush=True)

    if not raw_list:
        raise SystemExit("No valid raw spatial scalar features loaded")

    stats = fit_normalization_stats(raw_list, get_spatial_scalar_feature_names_v4())
    stats.update(
        {
            "semantics": "fit on train split only for pred_spatial_scalar_raw_features; apply to train/val/test without refitting",
            "manifest": str(args.manifest),
            "split_dir": str(args.split_dir),
            "split": args.split,
            "n_manifest_rows": len(rows),
            "n_split_complexes": len(split_ids),
            "n_selected_rows": len(selected),
            "n_loaded_features": len(raw_list),
            "n_residues": int(sum(x.shape[0] for x in raw_list)),
            "missing_paths": missing_paths,
            "n_load_errors": len(load_errors),
            "load_errors_head": load_errors[:10],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    tmp_path.replace(args.output)
    print(json.dumps({"event": "wrote_stats", "output": str(args.output), "n_loaded_features": len(raw_list), "n_residues": stats["n_residues"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
