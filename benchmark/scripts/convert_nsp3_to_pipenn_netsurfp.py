#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Tien et al. empirical maximum solvent accessibility values, in A^2.
# NetSurfP-3 biolib output reports RSA but not ASA; PIPENN expects both.
MAX_ASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 223.0,
    "E": 225.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
    "X": 200.0,
}


def read_fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    if not ids:
        raise ValueError(f"No FASTA records found: {path}")
    return ids


def nsp3_safe_name(identifier: str) -> str:
    value = str(identifier).strip().replace(" ", "_")
    value = re.sub(r"(?u)[^-\w.]", "", value)
    return value[:80]


def find_csv(input_dir: Path, protein_id: str) -> Path:
    candidates = [
        input_dir / f"{protein_id}.csv",
        input_dir / f"{nsp3_safe_name(protein_id)}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(input_dir.glob(f"{protein_id}*.csv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Missing unique NSP3 CSV for {protein_id} under {input_dir}")


def pick_column(df: pd.DataFrame, canonical: str, aliases: list[str]) -> str:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for name in aliases:
        key = name.lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Missing NSP3 column for {canonical}; columns={list(df.columns)}")


def convert_one(input_dir: Path, protein_id: str) -> pd.DataFrame:
    path = find_csv(input_dir, protein_id)
    df = pd.read_csv(path)
    residue_col = pick_column(df, "residue", ["residue", "seq", "aa", "sequence"])
    rsa_col = pick_column(df, "rsa", ["rsa", "rel_surf_acc", "relative_sasa"])
    q3h_col = pick_column(df, "p[q3_H]", ["q3(h)", "p[q3_H]", "prob_helix", "helix", "pH"])
    q3e_col = pick_column(df, "p[q3_E]", ["q3(e)", "p[q3_E]", "prob_sheet", "sheet", "pE"])
    q3c_col = pick_column(df, "p[q3_C]", ["q3(c)", "p[q3_C]", "prob_coil", "coil", "pC"])

    residues = df[residue_col].astype(str).str.strip().str.upper()
    rsa = df[rsa_col].astype(float).clip(lower=0.0, upper=1.0)
    asa = [float(r) * MAX_ASA.get(aa, MAX_ASA["X"]) for aa, r in zip(residues, rsa)]
    return pd.DataFrame(
        {
            "id": protein_id,
            "seq": residues,
            "rsa": rsa,
            "asa": asa,
            "p[q3_H]": df[q3h_col].astype(float),
            "p[q3_E]": df[q3e_col].astype(float),
            "p[q3_C]": df[q3c_col].astype(float),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert NetSurfP-3 per-protein CSV output into PIPENN NetSurfP-compatible CSV.")
    ap.add_argument("--nsp3-dir", required=True, help="Directory containing NetSurfP-3 per-protein <id>.csv files.")
    ap.add_argument("--fasta", required=True, help="FASTA file whose record order should be used in the output.")
    ap.add_argument("--output", required=True, help="Output PIPENN-compatible NetSurfP CSV.")
    args = ap.parse_args()

    nsp3_dir = Path(args.nsp3_dir)
    ids = read_fasta_ids(Path(args.fasta))
    frames = [convert_one(nsp3_dir, protein_id) for protein_id in ids]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"Wrote {len(ids)} proteins / {sum(len(frame) for frame in frames)} residues to {out}")


if __name__ == "__main__":
    main()
