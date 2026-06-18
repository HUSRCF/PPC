#!/usr/bin/env python3
"""Scan PPC protein feature residue metadata and insertion-code alignment.

This script treats ``features/protein_v4/pt`` as the alignment source used by
downstream ESM extraction.  It checks that all per-residue arrays have the same
length, that residue keys are unique, and optionally compares those keys against
CA residues in the fixed PDB files.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


FEATURE_TENSOR_KEYS = (
    "physchem_features",
    "spatial_scalar_features",
    "spatial_vector_features",
    "ca_coords",
)


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


def _norm_resseq(value: Any) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return int(value)


def _feature_pdb_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _locate_pdb(pdb_root: Path | None, pdb_id: str) -> Path | None:
    if pdb_root is None:
        return None
    candidates = (
        pdb_root / f"{pdb_id}.pdb",
        pdb_root / f"{pdb_id.lower()}.pdb",
        pdb_root / f"{pdb_id.upper()}.pdb",
        pdb_root / pdb_id / f"{pdb_id}.pdb",
        pdb_root / pdb_id.lower() / f"{pdb_id.lower()}.pdb",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_pdb_ca_keys(pdb_path: Path) -> tuple[list[tuple[str, int, str]], list[str]]:
    keys: list[tuple[str, int, str]] = []
    resnames: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    with pdb_path.open("rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                chain = line[21].strip()
                resseq = int(line[22:26])
                icode = line[26].strip()
                resname = line[17:20].strip()
            except Exception:
                parts = line.split()
                if len(parts) < 6:
                    continue
                resname = parts[3]
                chain = parts[4]
                resseq = int(parts[5])
                icode = ""
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
            resnames.append(resname)
    return keys, resnames


def _first_diff(
    a: list[tuple[str, int, str]],
    b: list[tuple[str, int, str]],
) -> int:
    for idx, (ka, kb) in enumerate(zip(a, b)):
        if ka != kb:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def _scan_one(args: tuple[str, str | None]) -> dict[str, Any]:
    feature_path = Path(args[0])
    pdb_root = Path(args[1]) if args[1] else None
    pdb_id = _feature_pdb_id(feature_path)

    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "feature_path": str(feature_path),
        "status": "ERROR",
        "error": "",
        "internal_errors": "",
        "n_residues_declared": "",
        "n_from_physchem_features": "",
        "n_from_spatial_scalar_features": "",
        "n_from_spatial_vector_features": "",
        "n_from_ca_coords": "",
        "n_from_all_atom_coords": "",
        "n_from_residue_names": "",
        "n_from_chain_ids": "",
        "n_from_residue_indices": "",
        "n_from_insertion_codes": "",
        "n_chains": "",
        "n_insertion_residues": "",
        "n_duplicate_feature_keys": "",
        "chain_lengths_json": "{}",
        "insertion_residues_json": "[]",
        "pdb_path": "",
        "pdb_ca_len": "",
        "pdb_key_equal": "",
        "first_diff_pos": "",
        "first_diff_feature_key": "",
        "first_diff_pdb_key": "",
        "missing_in_pdb_count": "",
        "extra_in_pdb_count": "",
    }

    try:
        data = _torch_load(feature_path)
        lengths: dict[str, int] = {}
        for key in FEATURE_TENSOR_KEYS:
            value = data.get(key)
            if value is not None:
                lengths[key] = int(value.shape[0])
                row[f"n_from_{key}"] = lengths[key]

        all_atom_coords = data.get("all_atom_coords")
        if all_atom_coords is not None:
            lengths["all_atom_coords"] = len(all_atom_coords)
            row["n_from_all_atom_coords"] = len(all_atom_coords)

        residue_names = [_norm_text(x) for x in _as_list(data.get("residue_names"))]
        chain_ids = [_norm_text(x) for x in _as_list(data.get("chain_ids"))]
        residue_indices = [_norm_resseq(x) for x in _as_list(data.get("residue_indices"))]
        insertion_codes = [_norm_text(x) for x in _as_list(data.get("insertion_codes"))]
        lengths["residue_names"] = len(residue_names)
        lengths["chain_ids"] = len(chain_ids)
        lengths["residue_indices"] = len(residue_indices)
        lengths["insertion_codes"] = len(insertion_codes)
        row["n_from_residue_names"] = len(residue_names)
        row["n_from_chain_ids"] = len(chain_ids)
        row["n_from_residue_indices"] = len(residue_indices)
        row["n_from_insertion_codes"] = len(insertion_codes)

        n_declared = data.get("n_residues")
        if n_declared is not None:
            n_declared = int(n_declared)
            row["n_residues_declared"] = n_declared
            lengths["n_residues"] = n_declared

        internal_errors: list[str] = []
        nonempty_lengths = {k: v for k, v in lengths.items() if v is not None}
        unique_lengths = sorted(set(nonempty_lengths.values()))
        if len(unique_lengths) != 1:
            internal_errors.append(f"length_mismatch:{nonempty_lengths}")
        n = unique_lengths[0] if len(unique_lengths) == 1 else max(nonempty_lengths.values())

        feature_keys = list(zip(chain_ids, residue_indices, insertion_codes))
        key_counts = Counter(feature_keys)
        duplicate_count = sum(count - 1 for count in key_counts.values() if count > 1)
        if duplicate_count:
            internal_errors.append(f"duplicate_feature_keys:{duplicate_count}")
        row["n_duplicate_feature_keys"] = duplicate_count

        chain_lengths: OrderedDict[str, int] = OrderedDict()
        for chain in chain_ids:
            chain_lengths[chain] = chain_lengths.get(chain, 0) + 1
        row["n_chains"] = len(chain_lengths)
        row["chain_lengths_json"] = json.dumps(chain_lengths, ensure_ascii=False, separators=(",", ":"))
        row["n_insertion_residues"] = sum(1 for code in insertion_codes if code)
        row["insertion_residues_json"] = json.dumps(
            [
                {
                    "feature_row": idx,
                    "chain_id": chain,
                    "residue_index": resseq,
                    "insertion_code": icode,
                    "residue_name": resname,
                }
                for idx, (chain, resseq, icode, resname) in enumerate(
                    zip(chain_ids, residue_indices, insertion_codes, residue_names)
                )
                if icode
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        row["internal_errors"] = ";".join(internal_errors)

        pdb_key_equal: bool | None = None
        pdb_path = _locate_pdb(pdb_root, pdb_id)
        if pdb_root is not None:
            if pdb_path is None:
                internal_errors.append("pdb_not_found")
                row["internal_errors"] = ";".join(internal_errors)
            else:
                row["pdb_path"] = str(pdb_path)
                pdb_keys, _ = _parse_pdb_ca_keys(pdb_path)
                row["pdb_ca_len"] = len(pdb_keys)
                pdb_key_equal = feature_keys == pdb_keys
                row["pdb_key_equal"] = int(pdb_key_equal)
                if not pdb_key_equal:
                    diff_pos = _first_diff(feature_keys, pdb_keys)
                    row["first_diff_pos"] = diff_pos
                    if 0 <= diff_pos < len(feature_keys):
                        row["first_diff_feature_key"] = repr(feature_keys[diff_pos])
                    if 0 <= diff_pos < len(pdb_keys):
                        row["first_diff_pdb_key"] = repr(pdb_keys[diff_pos])
                    feature_set = set(feature_keys)
                    pdb_set = set(pdb_keys)
                    row["missing_in_pdb_count"] = len(feature_set - pdb_set)
                    row["extra_in_pdb_count"] = len(pdb_set - feature_set)

        if internal_errors:
            row["status"] = "FAIL"
        elif pdb_key_equal is False:
            row["status"] = "FAIL"
        elif n <= 0:
            row["status"] = "FAIL"
            row["internal_errors"] = "empty_feature"
        else:
            row["status"] = "OK"
        return row
    except Exception as exc:
        row["error"] = repr(exc)
        return row


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


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in rows)
    insertion_rows = [row for row in rows if int(row.get("n_insertion_residues") or 0) > 0]
    pdb_mismatch = [row for row in rows if str(row.get("pdb_key_equal")) == "0"]
    internal_fail = [row for row in rows if row.get("internal_errors")]
    errors = [row for row in rows if row.get("error")]
    return {
        "n_total": len(rows),
        "statuses": dict(statuses),
        "n_with_insertion_codes": len(insertion_rows),
        "total_insertion_residues": sum(int(row.get("n_insertion_residues") or 0) for row in rows),
        "n_pdb_key_mismatch": len(pdb_mismatch),
        "n_internal_error": len(internal_fail),
        "n_exception": len(errors),
        "examples_with_insertion_codes": [
            {
                "pdb_id": row["pdb_id"],
                "n_insertion_residues": row["n_insertion_residues"],
                "chain_lengths_json": row["chain_lengths_json"],
                "insertion_residues_json": row.get("insertion_residues_json", "[]"),
            }
            for row in insertion_rows[:20]
        ],
        "examples_pdb_key_mismatch": [
            {
                "pdb_id": row["pdb_id"],
                "pdb_ca_len": row["pdb_ca_len"],
                "n_from_ca_coords": row["n_from_ca_coords"],
                "first_diff_pos": row["first_diff_pos"],
                "first_diff_feature_key": row["first_diff_feature_key"],
                "first_diff_pdb_key": row["first_diff_pdb_key"],
                "missing_in_pdb_count": row["missing_in_pdb_count"],
                "extra_in_pdb_count": row["extra_in_pdb_count"],
                "internal_errors": row["internal_errors"],
                "error": row["error"],
            }
            for row in pdb_mismatch[:20]
        ],
        "examples_internal_error": [
            {
                "pdb_id": row["pdb_id"],
                "internal_errors": row["internal_errors"],
                "error": row["error"],
            }
            for row in internal_fail[:20]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--pdb-root", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-list", default=None, type=Path)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tag", default="scan")
    args = parser.parse_args()

    feature_paths = _discover_features(args.features_root, args.id_list)
    if args.max is not None:
        feature_paths = feature_paths[: args.max]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdb_root_str = str(args.pdb_root) if args.pdb_root else None
    tasks = [(str(path), pdb_root_str) for path in feature_paths]
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

    csv_path = args.output_dir / f"{args.tag}_residue_metadata.csv"
    insertion_csv_path = args.output_dir / f"{args.tag}_insertion_residues.csv"
    json_path = args.output_dir / f"{args.tag}_summary.json"
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["pdb_id", "status", "error"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with insertion_csv_path.open("w", newline="") as handle:
        fieldnames_ins = ("pdb_id", "feature_row", "chain_id", "residue_index", "insertion_code", "residue_name")
        writer = csv.DictWriter(handle, fieldnames=fieldnames_ins)
        writer.writeheader()
        for row in rows:
            for item in json.loads(row.get("insertion_residues_json") or "[]"):
                writer.writerow({"pdb_id": row["pdb_id"], **item})

    summary = _summarize(rows)
    summary.update(
        {
            "features_root": str(args.features_root),
            "pdb_root": str(args.pdb_root) if args.pdb_root else None,
            "csv_path": str(csv_path),
            "insertion_csv_path": str(insertion_csv_path),
        }
    )
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["statuses"].get("FAIL", 0) == 0 and summary["n_exception"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
