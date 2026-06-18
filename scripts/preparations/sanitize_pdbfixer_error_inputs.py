#!/usr/bin/env python3
"""Create sanitized PDB inputs for PDBFixer error cases.

This is intended for PDBTM files where PDBFixer fails before it can remove
heterogens because malformed HETATM records break the PDB parser. The raw files
are left untouched.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DROP_RECORDS = ("HETATM", "CONECT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("features/pdbfixer/pdbfixer_report.csv"),
        help="PDBFixer CSV report containing error rows.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/pdbtm/cplx"),
        help="Directory containing raw PDBTM PDB files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features/pdbfixer/sanitized_nohetconect_input"),
        help="Directory for sanitized PDB files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("features/pdbfixer/sanitized_nohetconect_manifest.csv"),
        help="CSV manifest for generated sanitized files.",
    )
    parser.add_argument("--status", default="error", help="Report status to sanitize.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sanitized files.")
    return parser.parse_args()


def load_pdb_ids(report_path: Path, status: str) -> list[str]:
    with report_path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return sorted({row["pdb_id"] for row in rows if row.get("status") == status})


def sanitize_one(raw_path: Path, output_path: Path, overwrite: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "pdb_id": raw_path.stem,
        "input_path": str(raw_path),
        "output_path": str(output_path),
        "status": "error",
        "message": "",
        "lines_total": 0,
        "lines_written": 0,
        "hetatm_removed": 0,
        "conect_removed": 0,
        "atom_lines_written": 0,
    }

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        row["status"] = "skip"
        row["message"] = "output exists"
        return row

    if not raw_path.exists():
        row["message"] = "raw input missing"
        return row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with raw_path.open(errors="replace") as src, tmp_path.open("w") as dst:
        for line in src:
            row["lines_total"] = int(row["lines_total"]) + 1
            record = line[:6]
            if record == "HETATM":
                row["hetatm_removed"] = int(row["hetatm_removed"]) + 1
                continue
            if record == "CONECT":
                row["conect_removed"] = int(row["conect_removed"]) + 1
                continue
            dst.write(line)
            row["lines_written"] = int(row["lines_written"]) + 1
            if record == "ATOM  ":
                row["atom_lines_written"] = int(row["atom_lines_written"]) + 1

    tmp_path.replace(output_path)
    row["status"] = "ok"
    return row


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "status",
        "message",
        "input_path",
        "output_path",
        "lines_total",
        "lines_written",
        "hetatm_removed",
        "conect_removed",
        "atom_lines_written",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    pdb_ids = load_pdb_ids(args.report, args.status)
    if not pdb_ids:
        raise RuntimeError(f"No rows with status={args.status!r} found in {args.report}")

    rows = [
        sanitize_one(args.raw_dir / f"{pdb_id}.pdb", args.output_dir / f"{pdb_id}.pdb", args.overwrite)
        for pdb_id in pdb_ids
    ]
    write_manifest(args.manifest, rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1

    print(f"Report: {args.report.resolve()}")
    print(f"Raw input: {args.raw_dir.resolve()}")
    print(f"Sanitized output: {args.output_dir.resolve()}")
    print(f"Manifest: {args.manifest.resolve()}")
    print(f"Requested IDs: {len(pdb_ids)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
