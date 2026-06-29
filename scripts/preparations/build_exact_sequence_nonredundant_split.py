#!/usr/bin/env python3
"""Deduplicate a chain-level split by exact amino-acid sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import pandas as pd


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--qc-dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.chain_manifest)
    dataset = pd.read_csv(args.qc_dataset, usecols=["seq_id", "seq"])
    seq_map = dict(
        zip(
            dataset["seq_id"].astype(str),
            dataset["seq"].astype(str).str.replace(r"\s+", "", regex=True).str.upper(),
        )
    )
    manifest["seq"] = manifest["seq_id"].map(seq_map)
    missing = int(manifest["seq"].isna().sum())
    if missing:
        raise ValueError(f"Missing sequences for {missing} manifest rows")
    manifest["seq_sha1"] = manifest["seq"].map(lambda seq: hashlib.sha1(seq.encode()).hexdigest()[:16])
    manifest["seq_len_from_seq"] = manifest["seq"].str.len()

    cross_split = []
    for seq_hash, rows in manifest.groupby("seq_sha1", sort=True):
        splits = sorted(rows["split"].unique())
        if len(splits) > 1:
            cross_split.append(
                {
                    "seq_sha1": seq_hash,
                    "splits": splits,
                    "seq_ids": rows["seq_id"].head(20).tolist(),
                }
            )
    if cross_split:
        raise ValueError("Exact duplicate sequences cross splits: " + json.dumps(cross_split[:10], indent=2))

    split_order = {"train": 0, "val": 1, "test": 2}
    rep_rows = []
    map_rows = []
    for seq_hash, rows in manifest.groupby("seq_sha1", sort=True):
        rows = rows.sort_values(
            ["split", "pdb_id", "chain_id", "seq_id"],
            key=lambda col: col.map(split_order) if col.name == "split" else col,
        )
        rep = rows.iloc[0].copy()
        member_ids = rows["seq_id"].tolist()
        rep["duplicate_count"] = len(member_ids)
        rep["member_seq_ids"] = ";".join(member_ids)
        rep_rows.append(rep)
        for _, row in rows.iterrows():
            map_rows.append(
                {
                    "seq_sha1": seq_hash,
                    "representative_seq_id": rep["seq_id"],
                    "member_seq_id": row["seq_id"],
                    "split": row["split"],
                    "pdb_id": row["pdb_id"],
                    "chain_id": row["chain_id"],
                    "is_representative": int(row["seq_id"] == rep["seq_id"]),
                }
            )

    rep_df = (
        pd.DataFrame(rep_rows)
        .sort_values(
            ["split", "pdb_id", "chain_id", "seq_id"],
            key=lambda col: col.map(split_order) if col.name == "split" else col,
        )
        .reset_index(drop=True)
    )
    map_df = pd.DataFrame(map_rows).sort_values(
        ["representative_seq_id", "is_representative", "member_seq_id"],
        ascending=[True, False, True],
    )
    out_cols = [col for col in rep_df.columns if col != "seq"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(rep_df[out_cols], args.output_dir / "nr_chain_manifest.csv")
    atomic_csv(map_df, args.output_dir / "nr_sequence_reverse_map.csv")
    atomic_text(args.output_dir / "all_chain_ids.txt", "\n".join(rep_df["seq_id"]) + "\n")

    summary = {
        "mode": "exact_sequence_nonredundant",
        "source_chain_manifest": str(args.chain_manifest),
        "sequence_source": str(args.qc_dataset),
        "dedup_key": "uppercase whitespace-stripped exact seq from qc dataset; seq_sha1 is sha1(seq)[:16]",
        "representative_rule": "sort by split, pdb_id, chain_id, seq_id; take first",
        "input_chains": int(len(manifest)),
        "unique_sequences": int(len(rep_df)),
        "removed_duplicate_chains": int(len(manifest) - len(rep_df)),
        "cross_split_exact_duplicate_groups": 0,
        "split_stats": {},
        "duplicate_count_distribution": {
            str(k): int(v) for k, v in rep_df["duplicate_count"].value_counts().sort_index().items()
        },
        "top_duplicate_groups": rep_df.sort_values("duplicate_count", ascending=False)[
            ["seq_id", "split", "pdb_id", "chain_id", "seq_sha1", "seq_len_from_seq", "duplicate_count", "member_seq_ids"]
        ]
        .head(30)
        .to_dict(orient="records"),
    }
    for split in ["train", "val", "test"]:
        sub = rep_df[rep_df["split"] == split]
        atomic_text(args.output_dir / f"{split}_chain_ids.txt", "\n".join(sub["seq_id"]) + ("\n" if len(sub) else ""))
        atomic_csv(sub[out_cols], args.output_dir / f"{split}_chain_manifest.csv")
        summary["split_stats"][split] = {
            "input_chains": int((manifest["split"] == split).sum()),
            "unique_sequences": int(len(sub)),
            "removed_duplicate_chains": int((manifest["split"] == split).sum() - len(sub)),
            "n_residues_representatives": int(pd.to_numeric(sub["len_seq"], errors="coerce").fillna(0).sum()),
        }

    atomic_text(args.output_dir / "nr_summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_text(
        args.output_dir / "README.md",
        "\n".join(
            [
                "# Exact-Sequence Nonredundant PyPropel Split",
                "",
                "This directory deduplicates the PyPropel no-length-limit chain-filtered split by exact amino-acid sequence.",
                "",
                "Use `*_chain_ids.txt` for representative chains and `nr_sequence_reverse_map.csv` to map each representative back to all duplicate chains.",
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
