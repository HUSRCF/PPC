#!/usr/bin/env python3
"""Batch-fix PDBTM PDB files with PDBFixer.

The raw PDBTM files are left untouched. Fixed files are written to a separate
directory with the same file names.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/pdbtm/cplx"),
        help="Input directory containing raw PDBTM .pdb files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pdbtm/cplx_pdbfixer"),
        help="Output directory for fixed PDB files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("features/pdbfixer/pdbfixer_report.csv"),
        help="CSV report path.",
    )
    parser.add_argument("--pattern", default="*.pdb", help="Input file glob pattern.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixed PDB files.")
    parser.add_argument(
        "--add-missing-residues",
        action="store_true",
        help="Allow PDBFixer to model missing residues. Off by default to avoid changing contact labels.",
    )
    parser.add_argument(
        "--keep-terminal-missing-residues",
        action="store_true",
        help="When adding missing residues, also keep terminal missing residues.",
    )
    parser.add_argument(
        "--keep-heterogens",
        action="store_true",
        help="Keep heterogens instead of removing them.",
    )
    parser.add_argument("--keep-water", action="store_true", help="Keep water when removing heterogens.")
    parser.add_argument("--add-hydrogens", action="store_true", help="Add hydrogens after fixing atoms.")
    parser.add_argument("--ph", type=float, default=7.0, help="pH used when adding hydrogens.")
    return parser.parse_args()


def topology_counts(topology: Any) -> tuple[int, int, int]:
    chains = sum(1 for _ in topology.chains())
    residues = sum(1 for _ in topology.residues())
    atoms = sum(1 for _ in topology.atoms())
    return chains, residues, atoms


def trim_terminal_missing_residues(fixer: Any) -> None:
    chains = list(fixer.topology.chains())
    keys = list(fixer.missingResidues.keys())
    for key in keys:
        chain_idx, residue_idx = key
        residues = list(chains[chain_idx].residues())
        if residue_idx == 0 or residue_idx == len(residues):
            del fixer.missingResidues[key]


def fix_one(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    input_path = Path(task[0])
    output_path = Path(task[1])
    options = task[2]
    started = time.time()

    row: dict[str, Any] = {
        "pdb_id": input_path.stem,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "error",
        "message": "",
        "seconds": 0.0,
        "chains_before": "",
        "residues_before": "",
        "atoms_before": "",
        "chains_after": "",
        "residues_after": "",
        "atoms_after": "",
        "missing_residues_found": "",
        "missing_residues_added": "",
        "nonstandard_residues": "",
        "missing_atoms": "",
        "missing_terminals": "",
    }

    if output_path.exists() and output_path.stat().st_size > 0 and not options["overwrite"]:
        row["status"] = "skip"
        row["message"] = "output exists"
        row["seconds"] = round(time.time() - started, 3)
        return row

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fixer = PDBFixer(filename=str(input_path))
        chains_before, residues_before, atoms_before = topology_counts(fixer.topology)
        row["chains_before"] = chains_before
        row["residues_before"] = residues_before
        row["atoms_before"] = atoms_before

        fixer.findMissingResidues()
        missing_residues_found = sum(len(v) for v in fixer.missingResidues.values())
        row["missing_residues_found"] = missing_residues_found
        if not options["add_missing_residues"]:
            fixer.missingResidues = {}
        elif not options["keep_terminal_missing_residues"]:
            trim_terminal_missing_residues(fixer)
        row["missing_residues_added"] = sum(len(v) for v in fixer.missingResidues.values())

        fixer.findNonstandardResidues()
        row["nonstandard_residues"] = len(fixer.nonstandardResidues)
        fixer.replaceNonstandardResidues()

        if not options["keep_heterogens"]:
            fixer.removeHeterogens(keepWater=options["keep_water"])

        fixer.findMissingAtoms()
        row["missing_atoms"] = sum(len(v) for v in fixer.missingAtoms.values())
        row["missing_terminals"] = sum(len(v) for v in getattr(fixer, "missingTerminals", {}).values())
        fixer.addMissingAtoms()

        if options["add_hydrogens"]:
            fixer.addMissingHydrogens(options["ph"])

        chains_after, residues_after, atoms_after = topology_counts(fixer.topology)
        row["chains_after"] = chains_after
        row["residues_after"] = residues_after
        row["atoms_after"] = atoms_after

        tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("w") as handle:
                PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        row["status"] = "ok"
        row["message"] = ""
    except Exception as exc:  # noqa: BLE001 - report per-file failures and continue.
        row["status"] = "error"
        row["message"] = str(exc)[:500]

    row["seconds"] = round(time.time() - started, 3)
    return row


def write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "status",
        "message",
        "seconds",
        "input_path",
        "output_path",
        "chains_before",
        "residues_before",
        "atoms_before",
        "chains_after",
        "residues_after",
        "atoms_after",
        "missing_residues_found",
        "missing_residues_added",
        "nonstandard_residues",
        "missing_atoms",
        "missing_terminals",
    ]
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(input_dir.glob(args.pattern))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir}/{args.pattern}")

    workers = max(1, min(args.workers, cpu_count(), len(files)))
    options = {
        "overwrite": args.overwrite,
        "add_missing_residues": args.add_missing_residues,
        "keep_terminal_missing_residues": args.keep_terminal_missing_residues,
        "keep_heterogens": args.keep_heterogens,
        "keep_water": args.keep_water,
        "add_hydrogens": args.add_hydrogens,
        "ph": args.ph,
    }
    tasks = [(str(path), str(output_dir / path.name), options) for path in files]

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Report: {args.report.resolve()}")
    print(f"Files: {len(files)}")
    print(f"Workers: {workers}")
    print("Policy: replace nonstandard residues, remove heterogens/water, add missing atoms")
    print(f"Add missing residues: {args.add_missing_residues}")
    print(f"Add hydrogens: {args.add_hydrogens}")

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            row = fix_one(task)
            rows.append(row)
            print(f"[{idx}/{len(tasks)}] {row['pdb_id']} {row['status']} {row['message']}")
    else:
        with Pool(processes=workers) as pool:
            for idx, row in enumerate(pool.imap_unordered(fix_one, tasks), start=1):
                rows.append(row)
                if idx % 25 == 0 or row["status"] == "error":
                    ok = sum(1 for item in rows if item["status"] == "ok")
                    err = sum(1 for item in rows if item["status"] == "error")
                    skip = sum(1 for item in rows if item["status"] == "skip")
                    print(f"[{idx}/{len(tasks)}] ok={ok} error={err} skip={skip}")

    rows.sort(key=lambda row: row["pdb_id"])
    write_report(args.report.resolve(), rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    err = sum(1 for row in rows if row["status"] == "error")
    skip = sum(1 for row in rows if row["status"] == "skip")
    print("Summary")
    print(f"  ok: {ok}")
    print(f"  error: {err}")
    print(f"  skip: {skip}")
    print(f"  report: {args.report.resolve()}")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
