#!/usr/bin/env python3
"""Build PyPropel-QC chain allowlists for an existing complex-level split.

The pairwise split is a PDB/complex-level split.  PyPropel's tutorial QC is
chain-level, so downstream residue benchmarks should use only chains that pass
the selected TMKit/PDBTM QC table.  This script keeps the existing PDB split
assignment unchanged, writes chain-level allowlists, and audits cross-split
MMseqs hits among the retained chains.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def read_ids(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(text)
    os.replace(tmp, path)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_qc_dataset(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = row.get("seq_id", "")
            if not seq_id:
                continue
            row["prot"] = (row.get("prot") or "").lower()
            rows[seq_id] = row
    return rows


def read_chain_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = row.get("seq_id", "")
            if not seq_id:
                continue
            row["pdb_id"] = (row.get("pdb_id") or "").lower()
            rows[seq_id] = row
    return rows


def parse_hit(parts: list[str]) -> tuple[str, str, float, int, float, float] | None:
    if len(parts) < 6:
        return None
    try:
        pident = float(parts[2])
        alnlen = int(float(parts[3]))
        qcov = float(parts[4])
        tcov = float(parts[5])
    except ValueError:
        return None
    if pident > 1.0:
        pident /= 100.0
    return parts[0], parts[1], pident, alnlen, qcov, tcov


def audit_pairwise_hits(
    hits_path: Path,
    chain_to_split: dict[str, str],
    chain_to_pdb: dict[str, str],
    min_seq_id: float,
    coverage: float,
    min_aln_len: int,
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    examples: list[dict[str, Any]] = []
    max_cross_pident = 0.0
    n_rows = n_filtered = n_missing = 0
    with hits_path.open() as handle:
        for line in handle:
            parsed = parse_hit(line.rstrip("\n").split("\t"))
            if parsed is None:
                continue
            n_rows += 1
            query, target, pident, alnlen, qcov, tcov = parsed
            if pident < min_seq_id or alnlen < min_aln_len or qcov < coverage or tcov < coverage:
                continue
            n_filtered += 1
            qsplit = chain_to_split.get(query)
            tsplit = chain_to_split.get(target)
            if qsplit is None or tsplit is None:
                n_missing += 1
                continue
            if qsplit == tsplit:
                continue
            key = "__".join(sorted((qsplit, tsplit)))
            counts[key] += 1
            max_cross_pident = max(max_cross_pident, pident)
            if len(examples) < 50:
                examples.append(
                    {
                        "query": query,
                        "target": target,
                        "query_split": qsplit,
                        "target_split": tsplit,
                        "query_pdb": chain_to_pdb.get(query, ""),
                        "target_pdb": chain_to_pdb.get(target, ""),
                        "pident": pident,
                        "alnlen": alnlen,
                        "qcov": qcov,
                        "tcov": tcov,
                    }
                )
    return {
        "n_pairwise_hit_rows": n_rows,
        "n_hit_rows_after_filters": n_filtered,
        "n_hit_rows_with_chain_outside_allowlist": n_missing,
        "cross_split_hit_counts": dict(sorted(counts.items())),
        "n_cross_split_hits": int(sum(counts.values())),
        "max_cross_split_pident": max_cross_pident,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--qc-dataset", required=True, type=Path)
    parser.add_argument("--chain-metadata", required=True, type=Path)
    parser.add_argument("--pairwise-hits", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--min-aln-len", type=int, default=0)
    parser.add_argument("--mode-name", default="tmk_no_len_limit")
    args = parser.parse_args()

    split_to_pdbs = {split: read_ids(args.split_dir / f"{split}_ids.txt") for split in SPLITS}
    all_pdbs = sorted({pdb_id for ids in split_to_pdbs.values() for pdb_id in ids})
    pdb_to_split = {pdb_id: split for split, ids in split_to_pdbs.items() for pdb_id in ids}
    qc_rows = read_qc_dataset(args.qc_dataset)
    chain_rows = read_chain_metadata(args.chain_metadata)

    retained: list[dict[str, Any]] = []
    excluded_in_split = 0
    excluded_examples: list[dict[str, Any]] = []
    qc_seq_ids = set(qc_rows)
    for seq_id, chain in sorted(chain_rows.items()):
        pdb_id = chain["pdb_id"]
        split = pdb_to_split.get(pdb_id)
        if split is None:
            continue
        qc = qc_rows.get(seq_id)
        if qc is None:
            excluded_in_split += 1
            if len(excluded_examples) < 50:
                excluded_examples.append(
                    {
                        "seq_id": seq_id,
                        "pdb_id": pdb_id,
                        "chain_id": chain.get("chain_id", ""),
                        "reason": "not_in_qc_dataset",
                    }
                )
            continue
        retained.append(
            {
                "split": split,
                "seq_id": seq_id,
                "pdb_id": pdb_id,
                "chain_id": chain.get("chain_id", ""),
                "n_residues": chain.get("n_residues", qc.get("len_seq", "")),
                "len_seq": qc.get("len_seq", ""),
                "nchain": qc.get("nchain", ""),
                "rez": qc.get("rez", ""),
                "met1": qc.get("met1", ""),
                "mthm": qc.get("mthm", ""),
                "tm_type": qc.get("tm_type", ""),
                "bio_name": qc.get("bio_name", ""),
                "seq_equal_xml": qc.get("seq_equal_xml", ""),
                "feature_path": chain.get("feature_path", qc.get("feature_path", "")),
            }
        )

    chain_to_split = {row["seq_id"]: row["split"] for row in retained}
    chain_to_pdb = {row["seq_id"]: row["pdb_id"] for row in retained}
    leakage = audit_pairwise_hits(
        args.pairwise_hits,
        chain_to_split,
        chain_to_pdb,
        args.min_seq_id,
        args.coverage,
        args.min_aln_len,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "seq_id",
        "pdb_id",
        "chain_id",
        "n_residues",
        "len_seq",
        "nchain",
        "rez",
        "met1",
        "mthm",
        "tm_type",
        "bio_name",
        "seq_equal_xml",
        "feature_path",
    ]
    write_rows(args.output_dir / "chain_manifest.csv", retained, fieldnames)

    split_stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = [row for row in retained if row["split"] == split]
        ids = [row["seq_id"] for row in rows]
        write_text_atomic(args.output_dir / f"{split}_chain_ids.txt", "\n".join(ids) + ("\n" if ids else ""))
        write_rows(args.output_dir / f"{split}_chain_manifest.csv", rows, fieldnames)
        split_stats[split] = {
            "n_pdb_ids": len(split_to_pdbs[split]),
            "n_retained_chains": len(rows),
            "n_retained_pdb_ids": len({row["pdb_id"] for row in rows}),
            "n_residues": sum(int(float(row["len_seq"] or 0)) for row in rows),
        }
    write_text_atomic(
        args.output_dir / "all_chain_ids.txt",
        "\n".join(row["seq_id"] for row in retained) + ("\n" if retained else ""),
    )

    summary = {
        "mode": args.mode_name,
        "decision": "kept_existing_complex_split; applied chain-level PyPropel QC allowlist",
        "reason_no_resplit": (
            "The existing split is already component-disjoint under the pairwise SI graph. "
            "Filtering chains by the PyPropel QC dataset is a subset operation and cannot "
            "create new cross-split edges; the pairwise audit below verifies this."
        ),
        "paths": {
            "split_dir": str(args.split_dir),
            "qc_dataset": str(args.qc_dataset),
            "chain_metadata": str(args.chain_metadata),
            "pairwise_hits": str(args.pairwise_hits),
            "output_dir": str(args.output_dir),
        },
        "thresholds": {
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "cov_mode_interpretation": "qcov>=coverage and tcov>=coverage",
            "min_aln_len": args.min_aln_len,
        },
        "input_counts": {
            "n_split_pdb_ids": len(all_pdbs),
            "n_chain_metadata_rows_in_split_pdbs": len(retained) + excluded_in_split,
            "n_qc_dataset_rows": len(qc_rows),
            "n_qc_dataset_rows_in_split_pdbs": len(retained),
            "n_excluded_chain_rows_in_split_pdbs": excluded_in_split,
        },
        "split_stats": split_stats,
        "pairwise_si_audit": leakage,
        "excluded_examples": excluded_examples,
    }
    write_text_atomic(args.output_dir / "chain_filter_summary.json", json.dumps(summary, indent=2) + "\n")
    write_text_atomic(
        args.output_dir / "README.md",
        "\n".join(
            [
                "# PyPropel Chain-Filtered Split",
                "",
                "This directory keeps the existing PDB-level pairwise-SI split and adds",
                "chain-level allowlists from the PyPropel/TMKit no-length-limit QC dataset.",
                "",
                "Use `*_chain_ids.txt` or `*_chain_manifest.csv` for residue-level benchmarks.",
                "The original `train_ids.txt`, `val_ids.txt`, and `test_ids.txt` remain PDB ids.",
                "",
                "See `chain_filter_summary.json` for counts and the strict pairwise SI audit.",
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
