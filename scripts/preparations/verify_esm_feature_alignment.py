#!/usr/bin/env python3
"""Verify ESM feature files align row-by-row to PPC protein_v4 features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return list(value)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _norm_int(value: Any) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return int(value)


def _pdb_id_from_esm(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_esm2"):
        return stem[: -len("_esm2")]
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _locate_feature(features_root: Path, pdb_id: str) -> Path | None:
    candidates = (
        features_root / pdb_id / f"{pdb_id}_protein.pt",
        features_root / pdb_id.lower() / f"{pdb_id.lower()}_protein.pt",
        features_root / pdb_id.upper() / f"{pdb_id.upper()}_protein.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _first_diff(a: list[Any], b: list[Any]) -> int:
    for idx, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def _scan_one(args: tuple[str, str]) -> dict[str, Any]:
    esm_path = Path(args[0])
    features_root = Path(args[1])
    pdb_id = _pdb_id_from_esm(esm_path)
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "status": "ERROR",
        "esm_path": str(esm_path),
        "feature_path": "",
        "model_name": "",
        "embedding_dim": "",
        "n_feature": "",
        "n_esm": "",
        "chain_equal": "",
        "residue_index_equal": "",
        "insertion_code_equal": "",
        "residue_name_equal": "",
        "first_diff_field": "",
        "first_diff_pos": "",
        "feature_value": "",
        "esm_value": "",
        "error": "",
    }
    try:
        feature_path = _locate_feature(features_root, pdb_id)
        if feature_path is None:
            row["error"] = "feature_not_found"
            return row
        row["feature_path"] = str(feature_path)
        feature = _torch_load(feature_path)
        esm = _torch_load(esm_path)

        embeddings = esm["embeddings"]
        row["model_name"] = str(esm.get("model_name", ""))
        row["embedding_dim"] = int(embeddings.shape[1])
        n_feature = int(feature["ca_coords"].shape[0])
        n_esm = int(embeddings.shape[0])
        row["n_feature"] = n_feature
        row["n_esm"] = n_esm

        feature_chain = [_norm_text(x) for x in _as_list(feature.get("chain_ids"))]
        esm_chain = [_norm_text(x) for x in _as_list(esm.get("chain_ids", esm.get("chain_id")))]
        feature_resseq = [_norm_int(x) for x in _as_list(feature.get("residue_indices"))]
        esm_resseq = [_norm_int(x) for x in _as_list(esm.get("residue_indices", esm.get("residue_index")))]
        feature_icode = [_norm_text(x) for x in _as_list(feature.get("insertion_codes"))]
        esm_icode = [_norm_text(x) for x in _as_list(esm.get("insertion_codes", esm.get("insertion_code")))]
        feature_resname = [_norm_text(x).upper() for x in _as_list(feature.get("residue_names"))]
        esm_resname = [_norm_text(x).upper() for x in _as_list(esm.get("residue_names_3", esm.get("residue_name_3")))]

        checks = {
            "chain": feature_chain == esm_chain,
            "residue_index": feature_resseq == esm_resseq,
            "insertion_code": feature_icode == esm_icode,
            "residue_name": feature_resname == esm_resname,
        }
        row["chain_equal"] = int(checks["chain"])
        row["residue_index_equal"] = int(checks["residue_index"])
        row["insertion_code_equal"] = int(checks["insertion_code"])
        row["residue_name_equal"] = int(checks["residue_name"])

        if n_feature != n_esm:
            row["status"] = "FAIL"
            row["first_diff_field"] = "length"
        elif all(checks.values()):
            row["status"] = "OK"
        else:
            row["status"] = "FAIL"
            pairs = {
                "chain": (feature_chain, esm_chain),
                "residue_index": (feature_resseq, esm_resseq),
                "insertion_code": (feature_icode, esm_icode),
                "residue_name": (feature_resname, esm_resname),
            }
            for field, ok in checks.items():
                if ok:
                    continue
                a, b = pairs[field]
                diff = _first_diff(a, b)
                row["first_diff_field"] = field
                row["first_diff_pos"] = diff
                row["feature_value"] = repr(a[diff]) if 0 <= diff < len(a) else ""
                row["esm_value"] = repr(b[diff]) if 0 <= diff < len(b) else ""
                break
        return row
    except Exception as exc:
        row["error"] = repr(exc)
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--esm-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tag", default="esm_alignment")
    args = parser.parse_args()

    esm_paths = sorted(args.esm_root.glob("*/*_esm2.pt"))
    tasks = [(str(path), str(args.features_root)) for path in esm_paths]
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            rows.append(_scan_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_scan_one, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["pdb_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.tag}.csv"
    summary_path = args.output_dir / f"{args.tag}_summary.json"
    fieldnames = list(rows[0].keys()) if rows else ["pdb_id", "status", "error"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_total": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "n_error": sum(1 for row in rows if row.get("error")),
        "csv_path": str(csv_path),
        "examples_fail": [
            {
                "pdb_id": row["pdb_id"],
                "first_diff_field": row["first_diff_field"],
                "first_diff_pos": row["first_diff_pos"],
                "feature_value": row["feature_value"],
                "esm_value": row["esm_value"],
                "error": row["error"],
            }
            for row in rows
            if row["status"] != "OK"
        ][:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["statuses"].get("FAIL", 0) == 0 and summary["n_error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
