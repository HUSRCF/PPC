#!/usr/bin/env python3
"""Prepare ScanNet MSA run manifests for the PyPropel strict benchmark.

This manifest is intentionally ESMFold-unique-only: no GPSite/native chain PDB
fallback is used. Each run row corresponds to one exact unique sequence so that
ScanNet's HHblits/MSA cache is naturally de-duplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def index_pdbs(pdb_dirs: list[Path]) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for pdb_dir in pdb_dirs:
        for path in pdb_dir.glob("*.pdb"):
            by_stem.setdefault(path.stem, path)
    return by_stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--strict-unique", required=True, type=Path)
    parser.add_argument("--pdb-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    metadata = read_csv(args.metadata)
    unique_rows = read_csv(args.strict_unique)
    pdb_by_stem = index_pdbs(args.pdb_dir)

    seq_to_unique: dict[str, dict[str, str]] = {}
    for row in unique_rows:
        for seq_id in row["chain_ids"].split(";"):
            if seq_id:
                seq_to_unique[seq_id.lower()] = row

    run_to_chains: dict[str, list[str]] = defaultdict(list)
    chain_rows: list[dict[str, object]] = []
    run_meta: dict[str, dict[str, object]] = {}
    missing_unique: list[str] = []
    missing_pdb: list[str] = []

    for row in metadata:
        seq_id = row["seq_id"].lower()
        unique = seq_to_unique.get(seq_id)
        if unique is None:
            missing_unique.append(seq_id)
            continue
        unique_id = unique.get("old_unique_id", "")
        pdb_path = pdb_by_stem.get(unique_id)
        if pdb_path is None:
            missing_pdb.append(seq_id)
            continue

        run_to_chains[unique_id].append(seq_id)
        run_meta.setdefault(
            unique_id,
            {
                "input_id": unique_id,
                "pdb_path": str(pdb_path),
                "source": "esmfold_unique",
                "length": unique["length"],
                "first_seq_id": seq_id,
                "seq_sha256": unique["seq_sha256"],
            },
        )
        chain_rows.append(
            {
                "seq_id": seq_id,
                "pdb_id": row.get("pdb_id", ""),
                "chain_id": row.get("chain_id", ""),
                "n_residues": row.get("n_residues", ""),
                "n_positive": row.get("n_positive", ""),
                "input_id": unique_id,
                "has_pdb": 1,
                "input_source": "esmfold_unique",
                "pdb_path": str(pdb_path),
                "seq_sha256": unique["seq_sha256"],
            }
        )

    run_rows: list[dict[str, object]] = []
    for input_id in sorted(run_to_chains):
        seq_ids = sorted(set(run_to_chains[input_id]))
        row = dict(run_meta[input_id])
        row["n_seq_ids_needed"] = len(seq_ids)
        row["seq_ids"] = ",".join(seq_ids)
        run_rows.append(row)

    write_csv(
        args.out_dir / "strict_chain_input_manifest.csv",
        [
            "seq_id",
            "pdb_id",
            "chain_id",
            "n_residues",
            "n_positive",
            "input_id",
            "has_pdb",
            "input_source",
            "pdb_path",
            "seq_sha256",
        ],
        chain_rows,
    )
    write_tsv(
        args.out_dir / "strict_chain_input_manifest.tsv",
        [
            "seq_id",
            "pdb_id",
            "chain_id",
            "n_residues",
            "n_positive",
            "input_id",
            "has_pdb",
            "input_source",
            "pdb_path",
            "seq_sha256",
        ],
        chain_rows,
    )
    write_tsv(
        args.out_dir / "run_list.tsv",
        ["input_id", "pdb_path", "source", "length", "n_seq_ids_needed", "first_seq_id", "seq_sha256", "seq_ids"],
        run_rows,
    )
    summary = {
        "metadata_rows": len(metadata),
        "mapped_chain_rows": len(chain_rows),
        "run_rows": len(run_rows),
        "pdb_dirs": [str(path) for path in args.pdb_dir],
        "missing_unique_count": len(missing_unique),
        "missing_unique_examples": missing_unique[:30],
        "missing_pdb_count": len(missing_pdb),
        "missing_pdb_examples": missing_pdb[:30],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not (missing_unique or missing_pdb) else 2


if __name__ == "__main__":
    raise SystemExit(main())
