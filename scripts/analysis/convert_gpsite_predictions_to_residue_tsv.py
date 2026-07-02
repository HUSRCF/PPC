#!/usr/bin/env python3
"""Convert GPSite chain prediction TSV into the common residue TSV format."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def _load_manifest(path: Path) -> tuple[set[str], dict[str, str], dict[str, list[str]], dict[str, int]]:
    allowed: set[str] = set()
    exact_to_seq: dict[str, str] = {}
    base_to_seq: dict[str, list[str]] = defaultdict(list)
    lengths: dict[str, int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = row["seq_id"].strip().lower()
            pdb_id = row["pdb_id"].strip().lower()
            chain_id_exact = row["chain_id"].strip()
            chain_id = chain_id_exact.lower()
            allowed.add(seq_id)
            base_to_seq[f"{pdb_id}__{chain_id}"].append(seq_id)
            exact_to_seq[f"{pdb_id}__{chain_id_exact}"] = seq_id
            if row.get("qc_seq_id"):
                exact_to_seq[row["qc_seq_id"].strip()] = seq_id
            lengths[seq_id] = int(row["n_residues"])
    return allowed, exact_to_seq, base_to_seq, lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpsite-predictions", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()

    allowed, exact_to_seq, base_to_seq, expected_lengths = _load_manifest(args.manifest)
    seen_rows: dict[str, int] = defaultdict(int)
    skipped = 0
    ambiguous_base = 0
    rows_written = 0

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(args.gpsite_predictions) as in_handle, args.out_tsv.open("w", newline="") as out_handle:
        reader = csv.DictReader(in_handle, delimiter="\t")
        writer = csv.DictWriter(out_handle, fieldnames=["seq_id", "position", "residue", "score"], delimiter="\t")
        writer.writeheader()
        for row in reader:
            sample_id_exact = row.get("sample_id", "").strip()
            sample_id = sample_id_exact.lower()
            seq_id = exact_to_seq.get(sample_id_exact)
            if seq_id is None and sample_id in allowed:
                seq_id = sample_id
            if seq_id is None:
                base = f"{row.get('pdb_id', '').strip().lower()}__{row.get('chain_id', '').strip().lower()}"
                candidates = base_to_seq.get(base, [])
                if len(candidates) == 1:
                    seq_id = candidates[0]
                elif len(candidates) > 1:
                    ambiguous_base += 1
                    skipped += 1
                    continue
            if seq_id is None:
                skipped += 1
                continue
            position = int(row["row_idx_0based"]) + 1
            writer.writerow(
                {
                    "seq_id": seq_id,
                    "position": position,
                    "residue": row.get("aa", ""),
                    "score": row["score_protein_binding"],
                }
            )
            seen_rows[seq_id] += 1
            rows_written += 1

    missing = sorted(allowed - set(seen_rows))
    length_mismatch = {
        seq_id: {"expected": expected_lengths[seq_id], "observed": seen_rows.get(seq_id, 0)}
        for seq_id in sorted(allowed)
        if seen_rows.get(seq_id, 0) not in (0, expected_lengths[seq_id])
    }
    summary = {
        "gpsite_predictions": str(args.gpsite_predictions),
        "manifest": str(args.manifest),
        "out_tsv": str(args.out_tsv),
        "allowed_chains": len(allowed),
        "covered_chains": len(seen_rows),
        "missing_chains": len(missing),
        "missing_examples": missing[:50],
        "rows_written": rows_written,
        "skipped_rows": skipped,
        "ambiguous_base_rows": ambiguous_base,
        "length_mismatch_count": len(length_mismatch),
        "length_mismatch_examples": dict(list(length_mismatch.items())[:20]),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
