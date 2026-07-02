#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_sequence(value: str) -> str:
    return "".join(part.strip() for part in value.split(",") if part.strip())


def read_prepared_ids(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"uniprot_id", "sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            rows.append((row["uniprot_id"], parse_sequence(row["sequence"])))
    return rows


def read_embd(path: Path, expected_seq: str) -> np.ndarray:
    residues: list[str] = []
    vectors: list[list[float]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{line_no}: missing residue separator ':'")
            aa, rest = line.split(":", 1)
            residues.append(aa.strip())
            values = [float(x) for x in rest.strip().split()]
            if len(values) != 1024:
                raise ValueError(f"{path}:{line_no}: expected 1024 floats, got {len(values)}")
            vectors.append(values)
    observed = "".join(residues)
    if observed != expected_seq:
        raise ValueError(f"{path}: sequence mismatch observed_len={len(observed)} expected_len={len(expected_seq)}")
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert residue-level .embd text files to PIPENN-EMB NPZ.")
    ap.add_argument("--prepared-csv", required=True)
    ap.add_argument("--embd-dir", required=True)
    ap.add_argument("--output-npz", required=True)
    ap.add_argument("--qc-tsv", default=None)
    args = ap.parse_args()

    embd_dir = Path(args.embd_dir)
    arrays: dict[str, np.ndarray] = {}
    qc_rows: list[tuple[str, int, str, str]] = []
    for seq_id, seq in read_prepared_ids(Path(args.prepared_csv)):
        path = embd_dir / f"{seq_id}.embd"
        try:
            arr = read_embd(path, seq)
            arrays[seq_id] = arr
            qc_rows.append((seq_id, len(seq), "OK", str(arr.shape)))
        except Exception as exc:
            qc_rows.append((seq_id, len(seq), "ERROR", str(exc)))
            raise

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)

    if args.qc_tsv:
        qc = Path(args.qc_tsv)
        qc.parent.mkdir(parents=True, exist_ok=True)
        with qc.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["seq_id", "length", "status", "detail"])
            writer.writerows(qc_rows)

    n_res = sum(arr.shape[0] for arr in arrays.values())
    print(f"Wrote {len(arrays)} arrays / {n_res} residues to {out}")


if __name__ == "__main__":
    main()
