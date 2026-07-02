#!/usr/bin/env python3
"""Build PyPropel-QC-filtered strict benchmark manifests.

Filtering uses QC dataset keys (pdb_id/prot, chain_id/chain), not seq_id strings,
because strict labels may add suffixes such as _2 while the QC table does not.
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
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-dataset", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--all-strict-unique", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--len-lt", type=int, default=0, help="Optional strict length cutoff on QC len_seq")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    qc_by_key: dict[tuple[str, str], dict[str, str]] = {}
    with args.qc_dataset.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if args.len_lt and int(float(row["len_seq"])) >= args.len_lt:
                continue
            qc_by_key[(row["prot"].lower(), row["chain"])] = row

    metadata = read_csv(args.metadata)
    kept_meta: list[dict[str, object]] = []
    dropped_meta: list[dict[str, object]] = []
    allowed_seq_ids: set[str] = set()
    for row in metadata:
        key = (row["pdb_id"].lower(), row["chain_id"])
        qc = qc_by_key.get(key)
        if qc is None:
            dropped_meta.append(row)
            continue
        out = dict(row)
        for col in ["len_seq", "nchain", "rez", "met1", "mthm", "tm_type", "seq_id"]:
            out[f"qc_{col}"] = qc.get(col, "")
        kept_meta.append(out)
        allowed_seq_ids.add(row["seq_id"])

    labels_by_seq: dict[str, list[dict[str, str]]] = defaultdict(list)
    kept_label_rows: list[dict[str, str]] = []
    with args.labels.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["seq_id"] in allowed_seq_ids:
                kept_label_rows.append(row)
                labels_by_seq[row["seq_id"]].append(row)

    old_unique_by_sha: dict[str, dict[str, str]] = {}
    with args.all_strict_unique.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            old_unique_by_sha[row["seq_sha256"]] = row

    unique_by_sha: dict[str, dict[str, object]] = {}
    for seq_id, rows in labels_by_seq.items():
        sequence = "".join(row["residue"] for row in rows)
        seq_sha = hashlib.sha256(sequence.encode()).hexdigest()
        entry = unique_by_sha.setdefault(
            seq_sha,
            {
                "seq_sha256": seq_sha,
                "length": len(sequence),
                "chain_ids": [],
                "sequence": sequence,
                "old_unique_id": old_unique_by_sha.get(seq_sha, {}).get("old_unique_id", ""),
                "has_existing_pdb": old_unique_by_sha.get(seq_sha, {}).get("has_existing_pdb", ""),
            },
        )
        entry["chain_ids"].append(seq_id)

    unique_rows: list[dict[str, object]] = []
    for seq_sha, row in sorted(unique_by_sha.items(), key=lambda item: (int(item[1]["length"]), item[0])):
        chain_ids = sorted(row["chain_ids"])
        unique_rows.append(
            {
                "seq_sha256": seq_sha,
                "length": row["length"],
                "n_chain_ids": len(chain_ids),
                "chain_ids": ";".join(chain_ids),
                "old_unique_id": row.get("old_unique_id", ""),
                "has_existing_pdb": row.get("has_existing_pdb", ""),
                "sequence": row["sequence"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_fields = list(metadata[0].keys()) + ["qc_len_seq", "qc_nchain", "qc_rez", "qc_met1", "qc_mthm", "qc_tm_type", "qc_seq_id"]
    write_csv(args.out_dir / "test_chain_metadata.csv", meta_fields, kept_meta)
    write_csv(args.out_dir / "dropped_test_chain_metadata.csv", list(metadata[0].keys()), dropped_meta)
    write_tsv(args.out_dir / "test_chain_labels.tsv", ["seq_id", "position", "residue", "label"], kept_label_rows)
    write_tsv(
        args.out_dir / "all_strict_unique.tsv",
        ["seq_sha256", "length", "n_chain_ids", "chain_ids", "old_unique_id", "has_existing_pdb", "sequence"],
        unique_rows,
    )
    (args.out_dir / "allowed_seq_ids.txt").write_text("\n".join(sorted(allowed_seq_ids)) + "\n")

    summary = {
        "name": args.name,
        "len_lt": args.len_lt or None,
        "qc_dataset": str(args.qc_dataset),
        "input_metadata_rows": len(metadata),
        "kept_chain_rows": len(kept_meta),
        "dropped_chain_rows": len(dropped_meta),
        "kept_label_rows": len(kept_label_rows),
        "kept_unique_sequences": len(unique_rows),
        "kept_residues": len(kept_label_rows),
        "kept_positive": sum(int(row["label"]) for row in kept_label_rows),
        "outputs": {
            "metadata": str(args.out_dir / "test_chain_metadata.csv"),
            "labels": str(args.out_dir / "test_chain_labels.tsv"),
            "unique": str(args.out_dir / "all_strict_unique.tsv"),
            "allowed_seq_ids": str(args.out_dir / "allowed_seq_ids.txt"),
            "dropped_metadata": str(args.out_dir / "dropped_test_chain_metadata.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
