#!/usr/bin/env python3
"""Build a strict-native ScanNet manifest from strict sequence hashes.

This intentionally does not use the old chai1 seqid_to_unique.csv as a strict
mapping. It maps strict unique sequences to either an existing old ESMFold PDB by
old_unique_id or a strict-missing ESMFold PDB by missing_uid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def index_pdbs(dirs: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in dirs:
        if not root.exists():
            continue
        for pattern in ("**/*.pdb", "**/*.cif"):
            for path in root.glob(pattern):
                out[path.stem] = path
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-strict-unique", required=True, type=Path)
    parser.add_argument("--missing-unique", required=True, type=Path)
    parser.add_argument("--old-pdb-dir", action="append", default=[], type=Path)
    parser.add_argument("--strict-missing-pdb-dir", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    all_unique = read_tsv(args.all_strict_unique)
    missing_unique = read_tsv(args.missing_unique)
    missing_by_sha = {row["seq_sha256"]: row for row in missing_unique}
    old_pdb_by_id = index_pdbs(args.old_pdb_dir)
    missing_pdb_by_id = index_pdbs(args.strict_missing_pdb_dir)

    unique_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    chain_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    source_counts = Counter()
    chain_source_counts = Counter()

    for index, row in enumerate(all_unique, start=1):
        seq_sha = row["seq_sha256"]
        old_unique_id = row.get("old_unique_id", "")
        missing_row = missing_by_sha.get(seq_sha)
        missing_uid = missing_row["missing_uid"] if missing_row is not None else ""
        chain_ids = [x for x in row["chain_ids"].split(";") if x]
        length = int(row["length"])

        input_id = ""
        pdb_path: Path | None = None
        source = ""
        if old_unique_id and old_unique_id in old_pdb_by_id:
            input_id = old_unique_id
            pdb_path = old_pdb_by_id[old_unique_id]
            source = "old_unique_existing_pdb"
        elif missing_uid and missing_uid in missing_pdb_by_id:
            input_id = missing_uid
            pdb_path = missing_pdb_by_id[missing_uid]
            source = "strict_missing_esmfold_pdb"
        elif old_unique_id:
            input_id = old_unique_id
            source = "missing_old_unique_pdb"
        elif missing_uid:
            input_id = missing_uid
            source = missing_row.get("reason", "missing_strict_esmfold_pdb") if missing_row else "missing_strict_esmfold_pdb"
        else:
            input_id = f"strict_unmapped_{index:04d}_L{length}_{seq_sha[:12]}"
            source = "missing_no_unique_id"

        has_pdb = pdb_path is not None
        source_counts[source] += 1
        chain_source_counts[source] += len(chain_ids)
        unique_out = {
            "strict_unique_index": index,
            "input_id": input_id,
            "source": source,
            "has_pdb": int(has_pdb),
            "pdb_path": str(pdb_path) if pdb_path is not None else "",
            "seq_sha256": seq_sha,
            "length": length,
            "n_chain_ids": len(chain_ids),
            "chain_ids": ";".join(chain_ids),
            "old_unique_id": old_unique_id,
            "missing_uid": missing_uid,
            "sequence": row["sequence"],
        }
        unique_rows.append(unique_out)
        if has_pdb:
            run_rows.append(
                {
                    "input_id": input_id,
                    "pdb_path": str(pdb_path),
                    "source": source,
                    "length": length,
                    "n_chain_ids": len(chain_ids),
                    "chain_ids": ";".join(chain_ids),
                }
            )
        else:
            missing_rows.append(unique_out)

        for chain_id in chain_ids:
            chain_rows.append(
                {
                    "seq_id": chain_id,
                    "input_id": input_id,
                    "source": source,
                    "has_pdb": int(has_pdb),
                    "pdb_path": str(pdb_path) if pdb_path is not None else "",
                    "seq_sha256": seq_sha,
                    "length": length,
                    "old_unique_id": old_unique_id,
                    "missing_uid": missing_uid,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.out_dir / "strict_unique_manifest.tsv",
        [
            "strict_unique_index",
            "input_id",
            "source",
            "has_pdb",
            "pdb_path",
            "seq_sha256",
            "length",
            "n_chain_ids",
            "chain_ids",
            "old_unique_id",
            "missing_uid",
            "sequence",
        ],
        unique_rows,
    )
    write_tsv(
        args.out_dir / "run_list.tsv",
        ["input_id", "pdb_path", "source", "length", "n_chain_ids", "chain_ids"],
        run_rows,
    )
    write_tsv(
        args.out_dir / "chain_input_manifest.tsv",
        ["seq_id", "input_id", "source", "has_pdb", "pdb_path", "seq_sha256", "length", "old_unique_id", "missing_uid"],
        chain_rows,
    )
    write_tsv(
        args.out_dir / "missing_pdb_unique.tsv",
        [
            "strict_unique_index",
            "input_id",
            "source",
            "has_pdb",
            "pdb_path",
            "seq_sha256",
            "length",
            "n_chain_ids",
            "chain_ids",
            "old_unique_id",
            "missing_uid",
            "sequence",
        ],
        missing_rows,
    )

    summary = {
        "strict_unique_total": len(unique_rows),
        "strict_unique_with_pdb": len(run_rows),
        "strict_unique_missing_pdb": len(missing_rows),
        "strict_chain_total": len(chain_rows),
        "strict_chain_with_pdb": sum(int(r["has_pdb"]) for r in chain_rows),
        "strict_chain_missing_pdb": sum(1 - int(r["has_pdb"]) for r in chain_rows),
        "source_counts_unique": dict(sorted(source_counts.items())),
        "source_counts_chain": dict(sorted(chain_source_counts.items())),
        "outputs": {
            "strict_unique_manifest": str(args.out_dir / "strict_unique_manifest.tsv"),
            "run_list": str(args.out_dir / "run_list.tsv"),
            "chain_input_manifest": str(args.out_dir / "chain_input_manifest.tsv"),
            "missing_pdb_unique": str(args.out_dir / "missing_pdb_unique.tsv"),
        },
    }
    (args.out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not missing_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
