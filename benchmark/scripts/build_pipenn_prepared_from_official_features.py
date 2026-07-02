#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
MMSEQS_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
PIPENN_MISSING = 0.11111111

BASE_FEATURE_COLUMNS = [
    "domain",
    "sequence",
    "normalized_length",
    "normalized_abs_surf_acc",
    "rel_surf_acc",
    "prob_sheet",
    "prob_helix",
    "prob_coil",
]
PSSM_COLUMNS = [f"pssm_{aa}" for aa in AA_ORDER]
WM_BASE_COLUMNS = [
    "normalized_abs_surf_acc",
    "rel_surf_acc",
    "prob_sheet",
    "prob_helix",
    "prob_coil",
] + PSSM_COLUMNS
WINDOW_COLUMNS = [f"{w}_wm_{col}" for w in (3, 5, 7, 9) for col in WM_BASE_COLUMNS]
PIPENN_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PSSM_COLUMNS + WINDOW_COLUMNS
OUTPUT_COLUMNS = PIPENN_FEATURE_COLUMNS + ["p_interface", "Rlength", "uniprot_id"]


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    cur_id: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records[cur_id] = "".join(chunks)
            cur_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if cur_id is not None:
        records[cur_id] = "".join(chunks)
    if not records:
        raise ValueError(f"No FASTA records found: {path}")
    return records


def read_labels(path: Path) -> dict[str, list[int]]:
    by_id: dict[str, list[int]] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"pdb_id", "row_index", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            pid = row["pdb_id"]
            idx = int(row["row_index"])
            lab = int(row["label"])
            by_id.setdefault(pid, [])
            if idx != len(by_id[pid]):
                raise ValueError(f"{path}: non-contiguous row_index for {pid}: got {idx}, expected {len(by_id[pid])}")
            by_id[pid].append(lab)
    return by_id


def normalize_netsurfp_id(value: str) -> str:
    return str(value).strip().split()[0]


def read_netsurfp(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    aliases = {
        "id": ["id", "name", "protein", "protein_id", "uniprot_id"],
        "seq": ["seq", "residue", "aa", "sequence"],
        "rsa": ["rsa", "rel_surf_acc", "relative_sasa"],
        "asa": ["asa", "abs_surf_acc", "sasa"],
        "p[q3_H]": ["p[q3_H]", "prob_helix", "pH", "helix"],
        "p[q3_E]": ["p[q3_E]", "prob_sheet", "pE", "sheet"],
        "p[q3_C]": ["p[q3_C]", "prob_coil", "pC", "coil"],
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for cand in candidates:
            if cand in df.columns:
                rename[cand] = canonical
                break
        else:
            raise ValueError(f"{path} lacks NetSurfP column for {canonical}; columns={list(df.columns)}")
    df = df.rename(columns=rename)
    df["id"] = df["id"].map(normalize_netsurfp_id)
    grouped = {pid: sub.reset_index(drop=True) for pid, sub in df.groupby("id", sort=False)}
    return grouped


def sigmoid_pssm(x: int) -> float:
    return round(float(1.0 / (1.0 + math.exp(-int(x)))), 6)


def read_mmseqs_pssm(path: Path, fasta_ids: list[str]) -> dict[str, pd.DataFrame]:
    blocks: list[list[list[str]]] = []
    cur: list[list[str]] | None = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Query profile of sequence"):
                if cur is not None:
                    blocks.append(cur)
                cur = []
                continue
            if line.startswith("Pos"):
                continue
            if cur is not None and re.match(r"^\d+\s+", line):
                cur.append(re.split(r"\s+", line))
    if cur is not None:
        blocks.append(cur)
    if len(blocks) != len(fasta_ids):
        raise ValueError(f"PSSM block count {len(blocks)} != FASTA record count {len(fasta_ids)}")

    out: dict[str, pd.DataFrame] = {}
    for pid, rows in zip(fasta_ids, blocks):
        values: dict[str, list[float]] = {f"pssm_{aa}": [] for aa in MMSEQS_AA_ORDER}
        cns: list[str] = []
        for row in rows:
            if len(row) < 22:
                raise ValueError(f"{pid}: malformed MMseqs PSSM row: {row}")
            cns.append(row[1])
            for aa, val in zip(MMSEQS_AA_ORDER, row[2:22]):
                values[f"pssm_{aa}"].append(sigmoid_pssm(int(float(val))))
        df = pd.DataFrame(values)
        df["__cns"] = cns
        out[pid] = df
    return out


def centered_window(values: pd.Series, window: int) -> list[float]:
    rolled = values.rolling(window=window, center=True, min_periods=window).mean()
    return [PIPENN_MISSING if pd.isna(x) else float(x) for x in rolled]


def csv_seq(values: list[object]) -> str:
    return ",".join(str(x) for x in values)


def build_one(
    pid: str,
    seq: str,
    labels: list[int],
    nsp: pd.DataFrame,
    pssm: pd.DataFrame,
    *,
    strict_sequence_check: bool,
) -> dict[str, object]:
    length = len(seq)
    if len(labels) != length:
        raise ValueError(f"{pid}: labels length {len(labels)} != sequence length {length}")
    if len(nsp) != length:
        raise ValueError(f"{pid}: NetSurfP rows {len(nsp)} != sequence length {length}")
    if len(pssm) != length:
        raise ValueError(f"{pid}: PSSM rows {len(pssm)} != sequence length {length}")

    nsp_seq = "".join(str(x).strip() for x in nsp["seq"].tolist())
    pssm_seq = "".join(str(x).strip() for x in pssm["__cns"].tolist())
    if strict_sequence_check and nsp_seq.replace("X", "") != seq.replace("X", ""):
        raise ValueError(f"{pid}: NetSurfP sequence mismatch")
    if strict_sequence_check and pssm_seq.replace("X", "") != seq.replace("X", ""):
        raise ValueError(f"{pid}: PSSM consensus sequence mismatch")

    per_res = pd.DataFrame(
        {
            "domain": np.zeros(length, dtype=int),
            "sequence": list(seq),
            "normalized_length": np.full(length, min(length / 2050.0, 1.0), dtype=float),
            "normalized_abs_surf_acc": np.clip(nsp["asa"].astype(float).to_numpy() / 233.0, 0.0, 1.0),
            "rel_surf_acc": nsp["rsa"].astype(float).to_numpy(),
            "prob_sheet": nsp["p[q3_E]"].astype(float).to_numpy(),
            "prob_helix": nsp["p[q3_H]"].astype(float).to_numpy(),
            "prob_coil": nsp["p[q3_C]"].astype(float).to_numpy(),
        }
    )
    for col in PSSM_COLUMNS:
        per_res[col] = pssm[col].astype(float).to_numpy()
    window_data: dict[str, list[float]] = {}
    for window in (3, 5, 7, 9):
        for col in WM_BASE_COLUMNS:
            window_data[f"{window}_wm_{col}"] = centered_window(per_res[col], window)
    per_res = pd.concat([per_res, pd.DataFrame(window_data)], axis=1)

    row = {col: csv_seq(per_res[col].tolist()) for col in PIPENN_FEATURE_COLUMNS}
    row["p_interface"] = csv_seq([int(x) for x in labels])
    row["Rlength"] = length
    row["uniprot_id"] = pid
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Build PIPENN-1 prepared CSV from official-style NetSurfP and MMseqs PSSM outputs.")
    ap.add_argument("--raw-dir", required=True, help="Directory with <split>.fasta and <split>_residue_labels.tsv")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--netsurfp-csv", required=True)
    ap.add_argument("--mmseqs-pssm", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-strict-sequence-check", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    fasta = read_fasta(raw_dir / f"{args.split}.fasta")
    labels = read_labels(raw_dir / f"{args.split}_residue_labels.tsv")
    nsp = read_netsurfp(Path(args.netsurfp_csv))
    pssm = read_mmseqs_pssm(Path(args.mmseqs_pssm), list(fasta.keys()))

    rows = []
    for pid, seq in fasta.items():
        if pid not in labels:
            raise ValueError(f"{pid}: missing labels")
        if pid not in nsp:
            raise ValueError(f"{pid}: missing NetSurfP rows")
        if pid not in pssm:
            raise ValueError(f"{pid}: missing PSSM block")
        rows.append(
            build_one(
                pid,
                seq,
                labels[pid],
                nsp[pid],
                pssm[pid],
                strict_sequence_check=not args.no_strict_sequence_check,
            )
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out, index=False)
    n_res = sum(len(seq) for seq in fasta.values())
    print(f"Wrote {len(rows)} proteins / {n_res} residues to {out}")


if __name__ == "__main__":
    main()
