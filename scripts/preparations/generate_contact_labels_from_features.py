#!/usr/bin/env python3
"""Generate residue-level protein-protein contact labels from PPC features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _feature_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _atom_is_heavy(name: Any) -> bool:
    atom = _norm_text(name).upper()
    return bool(atom) and not atom.startswith("H") and not atom.startswith("D")


def _aabb_min_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_min = left.min(axis=0)
    left_max = left.max(axis=0)
    right_min = right.min(axis=0)
    right_max = right.max(axis=0)
    delta = np.maximum(0.0, np.maximum(left_min - right_max, right_min - left_max))
    return float(np.sqrt(np.sum(delta * delta)))


def _build_chain_atoms(
    all_atom_coords: list[Any],
    all_atom_names: list[Any],
    chain_ids: list[str],
    heavy_only: bool,
) -> dict[str, dict[str, Any]]:
    chain_atoms: dict[str, dict[str, Any]] = OrderedDict()
    for res_idx, (coords, atom_names, chain_id) in enumerate(zip(all_atom_coords, all_atom_names, chain_ids)):
        arr = np.asarray(coords, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        names = list(atom_names)
        if heavy_only:
            keep = np.array([_atom_is_heavy(name) for name in names], dtype=bool)
            arr = arr[keep]
        if arr.shape[0] == 0:
            continue
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        if arr.shape[0] == 0:
            continue
        bucket = chain_atoms.setdefault(chain_id, {"coords": [], "residue_rows": []})
        bucket["coords"].append(arr)
        bucket["residue_rows"].append(np.full(arr.shape[0], res_idx, dtype=np.int64))

    out: dict[str, dict[str, Any]] = OrderedDict()
    for chain_id, bucket in chain_atoms.items():
        coords_cat = np.concatenate(bucket["coords"], axis=0)
        rows_cat = np.concatenate(bucket["residue_rows"], axis=0)
        out[chain_id] = {"coords": coords_cat, "residue_rows": rows_cat}
    return out


def _label_one(
    feature_path: Path,
    output_root: Path,
    cutoff: float,
    heavy_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    pdb_id = _feature_id(feature_path)
    out_path = output_root / pdb_id / f"{pdb_id}_labels.pt"
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "status": "ERROR",
        "feature_path": str(feature_path),
        "label_path": str(out_path),
        "n_residues": 0,
        "n_chains": 0,
        "n_positive": 0,
        "n_negative": 0,
        "positive_fraction": 0.0,
        "n_chain_pairs_checked": 0,
        "n_chain_pairs_skipped_aabb": 0,
        "error": "",
    }
    try:
        if out_path.exists() and not overwrite:
            data = _torch_load(out_path)
            labels = torch.as_tensor(data["labels"], dtype=torch.long)
            row.update(
                {
                    "status": "SKIP",
                    "n_residues": int(labels.numel()),
                    "n_chains": int(data.get("n_chains", 0)),
                    "n_positive": int(labels.sum().item()),
                    "n_negative": int((labels == 0).sum().item()),
                    "positive_fraction": float(labels.float().mean().item()) if labels.numel() else 0.0,
                }
            )
            return row

        feature = _torch_load(feature_path)
        n_res = int(feature["ca_coords"].shape[0])
        chain_ids = [_norm_text(x) for x in _as_list(feature["chain_ids"])]
        labels = torch.zeros(n_res, dtype=torch.long)
        chain_atoms = _build_chain_atoms(
            feature["all_atom_coords"],
            feature["all_atom_names"],
            chain_ids,
            heavy_only=heavy_only,
        )
        chain_keys = list(chain_atoms.keys())
        checked = 0
        skipped = 0
        positive_rows: set[int] = set()

        for left_key, right_key in combinations(chain_keys, 2):
            left = chain_atoms[left_key]
            right = chain_atoms[right_key]
            if left["coords"].shape[0] == 0 or right["coords"].shape[0] == 0:
                continue
            if _aabb_min_distance(left["coords"], right["coords"]) > cutoff:
                skipped += 1
                continue
            checked += 1
            left_tree = cKDTree(left["coords"])
            right_tree = cKDTree(right["coords"])
            neighbors = left_tree.query_ball_tree(right_tree, r=cutoff)
            for left_atom_idx, right_atom_indices in enumerate(neighbors):
                if not right_atom_indices:
                    continue
                positive_rows.add(int(left["residue_rows"][left_atom_idx]))
                right_rows = right["residue_rows"][np.asarray(right_atom_indices, dtype=np.int64)]
                positive_rows.update(int(x) for x in np.unique(right_rows))

        if positive_rows:
            labels[torch.tensor(sorted(positive_rows), dtype=torch.long)] = 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / f".{pdb_id}_{uuid.uuid4().hex}.tmp"
        result = {
            "labels": labels,
            "pdb_id": pdb_id,
            "source_feature_path": str(feature_path),
            "cutoff_angstrom": float(cutoff),
            "heavy_only": bool(heavy_only),
            "label_definition": "positive if any heavy atom is within cutoff of a heavy atom from a different chain",
            "chain_ids": chain_ids,
            "residue_indices": _as_list(feature["residue_indices"]),
            "insertion_codes": [_norm_text(x) for x in _as_list(feature["insertion_codes"])],
            "residue_names": [_norm_text(x) for x in _as_list(feature["residue_names"])],
            "n_residues": n_res,
            "n_chains": len(set(chain_ids)),
            "n_positive": int(labels.sum().item()),
            "n_negative": int((labels == 0).sum().item()),
            "n_chain_pairs_checked": checked,
            "n_chain_pairs_skipped_aabb": skipped,
        }
        torch.save(result, tmp_path)
        os.replace(tmp_path, out_path)
        n_pos = int(labels.sum().item())
        row.update(
            {
                "status": "OK",
                "n_residues": n_res,
                "n_chains": len(set(chain_ids)),
                "n_positive": n_pos,
                "n_negative": int((labels == 0).sum().item()),
                "positive_fraction": float(n_pos / n_res) if n_res else 0.0,
                "n_chain_pairs_checked": checked,
                "n_chain_pairs_skipped_aabb": skipped,
            }
        )
        return row
    except Exception as exc:
        row["error"] = repr(exc)
        return row


def _worker(task: tuple[str, str, float, bool, bool]) -> dict[str, Any]:
    feature_path, output_root, cutoff, heavy_only, overwrite = task
    return _label_one(Path(feature_path), Path(output_root), cutoff, heavy_only, overwrite)


def _discover_features(features_root: Path, id_list: Path | None) -> list[Path]:
    if id_list is None:
        return sorted(features_root.glob("*/*_protein.pt"))
    ids = [
        line.strip().lower()
        for line in id_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    paths: list[Path] = []
    for pdb_id in ids:
        path = features_root / pdb_id / f"{pdb_id}_protein.pt"
        if path.exists():
            paths.append(path)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--id-list", default=None, type=Path)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--include-hydrogen", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    feature_paths = _discover_features(args.features_root, args.id_list)
    if args.max is not None:
        feature_paths = feature_paths[: args.max]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_root / "manifest.csv")

    tasks = [
        (str(path), str(args.output_root), args.cutoff, not args.include_hydrogen, args.overwrite)
        for path in feature_paths
    ]
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            rows.append(_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["pdb_id"])

    fieldnames = list(rows[0].keys()) if rows else ["pdb_id", "status", "error"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["status"] for row in rows)
    ok_rows = [row for row in rows if row["status"] in {"OK", "SKIP"}]
    summary = {
        "n_total": len(rows),
        "statuses": dict(statuses),
        "n_labeled_or_skipped": len(ok_rows),
        "n_with_positive": sum(1 for row in ok_rows if int(row["n_positive"]) > 0),
        "n_without_positive": sum(1 for row in ok_rows if int(row["n_positive"]) == 0),
        "total_residues": sum(int(row["n_residues"]) for row in ok_rows),
        "total_positive": sum(int(row["n_positive"]) for row in ok_rows),
        "total_negative": sum(int(row["n_negative"]) for row in ok_rows),
        "cutoff": args.cutoff,
        "heavy_only": not args.include_hydrogen,
        "manifest": str(manifest_path),
        "examples_error": [
            {"pdb_id": row["pdb_id"], "error": row["error"]}
            for row in rows
            if row["status"] == "ERROR"
        ][:20],
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if statuses.get("ERROR", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
