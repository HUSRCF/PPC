#!/usr/bin/env python3
"""Export per-chain FASTA records from PPC protein feature files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
}


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


def _safe_token(value: str) -> str:
    if not value:
        return "blank"
    out: list[str] = []
    for char in value:
        if char.isalnum():
            out.append(char)
        elif char in {"_", "-", "."}:
            out.append(char)
        else:
            out.append("_" + format(ord(char), "x"))
    return "".join(out) or "blank"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _wrap_fasta(sequence: str, width: int) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def _chain_rows(feature_path: Path, min_length: int, max_unknown_frac: float) -> dict[str, Any]:
    pdb_id = _feature_id(feature_path).lower()
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "feature_path": str(feature_path),
        "status": "ERROR",
        "error": "",
        "chains": [],
    }
    try:
        data = _torch_load(feature_path)
        chain_ids = [_norm_text(x) for x in _as_list(data["chain_ids"])]
        residue_names = [_norm_text(x).upper() for x in _as_list(data["residue_names"])]
        if len(chain_ids) != len(residue_names):
            raise ValueError(f"chain_ids length {len(chain_ids)} != residue_names length {len(residue_names)}")

        chains: OrderedDict[str, list[str]] = OrderedDict()
        for chain_id, residue_name in zip(chain_ids, residue_names):
            chains.setdefault(chain_id, []).append(AA3_TO_1.get(residue_name, "X"))

        seen_seq_ids: set[str] = set()
        out_rows: list[dict[str, Any]] = []
        for chain_index, (chain_id, letters) in enumerate(chains.items()):
            sequence = "".join(letters)
            n_residues = len(sequence)
            n_unknown = sequence.count("X")
            unknown_frac = float(n_unknown / n_residues) if n_residues else 1.0
            if n_residues < min_length or unknown_frac > max_unknown_frac:
                continue
            base_seq_id = f"{pdb_id}__{_safe_token(chain_id)}"
            seq_id = base_seq_id
            suffix = 2
            while seq_id in seen_seq_ids:
                seq_id = f"{base_seq_id}_{suffix}"
                suffix += 1
            seen_seq_ids.add(seq_id)
            out_rows.append(
                {
                    "seq_id": seq_id,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "chain_index": chain_index,
                    "n_residues": n_residues,
                    "n_unknown": n_unknown,
                    "unknown_fraction": unknown_frac,
                    "sequence": sequence,
                    "feature_path": str(feature_path),
                }
            )
        row["chains"] = out_rows
        row["status"] = "OK"
        return row
    except Exception as exc:
        row["error"] = repr(exc)
        return row


def _worker(task: tuple[str, int, float]) -> dict[str, Any]:
    feature_path, min_length, max_unknown_frac = task
    return _chain_rows(Path(feature_path), min_length, max_unknown_frac)


def _discover_features(features_root: Path, id_list: Path | None) -> list[Path]:
    if id_list is None:
        return sorted(features_root.glob("*/*_protein.pt"))
    ids = [
        line.strip().lower()
        for line in id_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return [features_root / pdb_id / f"{pdb_id}_protein.pt" for pdb_id in ids if (features_root / pdb_id / f"{pdb_id}_protein.pt").exists()]


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--output-fasta", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument("--id-list", default=None, type=Path)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-unknown-frac", type=float, default=1.0)
    parser.add_argument("--line-width", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    feature_paths = _discover_features(args.features_root, args.id_list)
    if args.max is not None:
        feature_paths = feature_paths[: args.max]

    tasks = [(str(path), args.min_length, args.max_unknown_frac) for path in feature_paths]
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            results.append(_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["pdb_id"])

    chain_rows: list[dict[str, Any]] = []
    fasta_chunks: list[str] = []
    for item in results:
        for chain in item["chains"]:
            chain_rows.append(chain)
            header = (
                f">{chain['seq_id']} pdb_id={chain['pdb_id']} "
                f"chain_id={chain['chain_id'] or '<blank>'} n_residues={chain['n_residues']}"
            )
            fasta_chunks.append(header + "\n" + _wrap_fasta(chain["sequence"], args.line_width))

    _write_text_atomic(args.output_fasta, "\n".join(fasta_chunks) + ("\n" if fasta_chunks else ""))

    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = args.metadata_csv.parent / f".{args.metadata_csv.name}.{uuid.uuid4().hex}.tmp"
    fieldnames = [
        "seq_id",
        "pdb_id",
        "chain_id",
        "chain_index",
        "n_residues",
        "n_unknown",
        "unknown_fraction",
        "feature_path",
    ]
    with tmp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chain in chain_rows:
            writer.writerow({key: chain[key] for key in fieldnames})
    os.replace(tmp_csv, args.metadata_csv)

    n_error = sum(1 for item in results if item["status"] != "OK")
    summary = {
        "features_root": str(args.features_root),
        "output_fasta": str(args.output_fasta),
        "metadata_csv": str(args.metadata_csv),
        "n_feature_files": len(feature_paths),
        "n_feature_ok": len(feature_paths) - n_error,
        "n_feature_error": n_error,
        "n_chains": len(chain_rows),
        "n_residues": int(sum(int(row["n_residues"]) for row in chain_rows)),
        "n_unknown": int(sum(int(row["n_unknown"]) for row in chain_rows)),
        "examples_error": [
            {"pdb_id": item["pdb_id"], "error": item["error"]}
            for item in results
            if item["status"] != "OK"
        ][:10],
    }
    summary_path = args.summary_json or args.metadata_csv.with_suffix(".summary.json")
    _write_text_atomic(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if n_error == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
