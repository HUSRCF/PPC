#!/usr/bin/env python3
"""Prepare ScanNet noMSA run manifests for the strict PPC benchmark.

The preferred input is one ESMFold PDB per unique sequence. Some strict chains are
not present in that ESMFold union; for those, use the existing GPSite per-chain
PDB files so ScanNet can still cover the full strict chain set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def load_label_sequences(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    residues_by_seq: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            residues_by_seq[row["seq_id"].lower()].append(row["residue"])
    return {seq_id: "".join(residues) for seq_id, residues in residues_by_seq.items()}


def index_union_pdbs(union_pdb_dir: Path) -> tuple[dict[str, Path], dict[tuple[int, str], str]]:
    by_id: dict[str, Path] = {}
    by_length_hash: dict[tuple[int, str], str] = {}
    for pdb_path in union_pdb_dir.glob("*.pdb"):
        unique_id = pdb_path.stem
        by_id[unique_id] = pdb_path
        parts = unique_id.split("_")
        if len(parts) >= 4 and parts[-2].startswith("L"):
            try:
                length = int(parts[-2][1:])
            except ValueError:
                continue
            by_length_hash[(length, parts[-1])] = unique_id
    return by_id, by_length_hash


def index_fallback_pdbs(fallback_dirs: list[Path]) -> dict[str, Path]:
    by_pdb_chain: dict[str, Path] = {}
    for fallback_dir in fallback_dirs:
        for pdb_path in fallback_dir.glob("**/*.pdb"):
            by_pdb_chain[pdb_path.stem] = pdb_path
    return by_pdb_chain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--seqid-to-unique", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--union-pdb-dir", required=True, type=Path)
    parser.add_argument("--fallback-pdb-dir", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    metadata_rows = read_csv(args.metadata)
    seq_rows = read_csv(args.seqid_to_unique)
    label_sequences = load_label_sequences(args.labels)
    union_by_id, union_by_length_hash = index_union_pdbs(args.union_pdb_dir)
    fallback_by_pdb_chain = index_fallback_pdbs(args.fallback_pdb_dir)

    seq_to_unique: dict[str, dict[str, str]] = {}
    duplicate_seqids: list[str] = []
    for row in seq_rows:
        key = row["seq_id"].lower()
        if key in seq_to_unique:
            duplicate_seqids.append(key)
        seq_to_unique[key] = row

    chain_rows: list[dict[str, object]] = []
    missing_inputs: list[str] = []
    inferred_unique: list[str] = []
    run_to_seqids: dict[str, list[str]] = defaultdict(list)
    run_meta: dict[str, dict[str, object]] = {}
    source_counts: dict[str, int] = defaultdict(int)

    for meta in metadata_rows:
        seq_id = meta["seq_id"]
        key = seq_id.lower()
        seq_row = seq_to_unique.get(key)
        unique_id = seq_row["unique_id"] if seq_row is not None else ""
        unique_length = int(seq_row["length"]) if seq_row is not None else 0
        input_id = ""
        pdb_path: Path | None = None
        source = ""

        if unique_id and unique_id in union_by_id:
            input_id = unique_id
            pdb_path = union_by_id[unique_id]
            source = "esmfold_unique"
        else:
            sequence = label_sequences.get(key)
            if sequence:
                seq_hash12 = hashlib.sha256(sequence.encode()).hexdigest()[:12]
                inferred_id = union_by_length_hash.get((len(sequence), seq_hash12))
                if inferred_id is not None:
                    unique_id = inferred_id
                    unique_length = len(sequence)
                    input_id = unique_id
                    pdb_path = union_by_id[unique_id]
                    source = "esmfold_unique_inferred_from_labels"
                    inferred_unique.append(seq_id)

        if pdb_path is None:
            fallback_key = f"{meta['pdb_id']}__{meta['chain_id']}"
            fallback_path = fallback_by_pdb_chain.get(fallback_key)
            if fallback_path is not None:
                input_id = seq_id
                pdb_path = fallback_path
                source = "gpsite_chain_pdb_fallback"
                unique_id = unique_id or ""
                unique_length = unique_length or int(meta["n_residues"])

        if pdb_path is None:
            missing_inputs.append(seq_id)
            continue

        run_to_seqids[input_id].append(seq_id)
        source_counts[source] += 1
        run_meta.setdefault(
            input_id,
            {
                "input_id": input_id,
                "pdb_path": str(pdb_path),
                "source": source,
                "length": unique_length or int(meta["n_residues"]),
                "first_seq_id": seq_id,
            },
        )
        chain_rows.append(
            {
                "seq_id": seq_id,
                "pdb_id": meta.get("pdb_id", ""),
                "chain_id": meta.get("chain_id", ""),
                "n_residues": meta.get("n_residues", ""),
                "n_positive": meta.get("n_positive", ""),
                "input_id": input_id,
                "input_source": source,
                "unique_id": unique_id,
                "unique_length": unique_length or "",
                "pdb_path": str(pdb_path),
            }
        )

    run_rows: list[dict[str, object]] = []
    for input_id in sorted(run_to_seqids):
        seq_ids = sorted(run_to_seqids[input_id])
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
            "input_source",
            "unique_id",
            "unique_length",
            "pdb_path",
        ],
        chain_rows,
    )
    write_tsv(
        args.out_dir / "run_list.tsv",
        ["input_id", "pdb_path", "source", "length", "n_seq_ids_needed", "first_seq_id", "seq_ids"],
        run_rows,
    )
    # Backward-compatible name for older runner invocations.
    write_tsv(
        args.out_dir / "unique_run_list.tsv",
        ["input_id", "pdb_path", "source", "length", "n_seq_ids_needed", "first_seq_id", "seq_ids"],
        run_rows,
    )

    summary = {
        "metadata_rows": len(metadata_rows),
        "mapped_chain_rows": len(chain_rows),
        "run_rows": len(run_rows),
        "duplicate_seqids_in_unique_map": len(duplicate_seqids),
        "inferred_unique_from_labels": len(inferred_unique),
        "source_counts_by_chain": dict(sorted(source_counts.items())),
        "missing_input_count": len(missing_inputs),
        "missing_input_examples": missing_inputs[:20],
        "inferred_unique_examples": inferred_unique[:20],
        "outputs": {
            "strict_chain_input_manifest": str(args.out_dir / "strict_chain_input_manifest.csv"),
            "run_list": str(args.out_dir / "run_list.tsv"),
        },
    }
    (args.out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if missing_inputs:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
