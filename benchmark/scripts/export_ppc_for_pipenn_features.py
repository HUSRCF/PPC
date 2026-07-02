#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "C",
    "PYL": "K",
    "ASX": "X",
    "GLX": "X",
    "UNK": "X",
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_label(label_root: Path, pdb_id: str) -> dict:
    path = label_root / pdb_id / f"{pdb_id}_labels.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def label_to_sequence(rec: dict) -> str:
    names = rec.get("residue_names") or rec.get("residue_names_3")
    if names is None:
        raise KeyError("residue_names")
    return "".join(AA3_TO_1.get(str(name).upper(), "X") for name in names)


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rid, seq in records:
            f.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--label-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    label_root = Path(args.label_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ids = read_ids(split_dir / f"{split}_ids.txt")
        fasta_records: list[tuple[str, str]] = []
        table_path = out_dir / f"{split}_residue_labels.tsv"
        with table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "pdb_id",
                    "row_index",
                    "chain_id",
                    "residue_index",
                    "insertion_code",
                    "residue_name_3",
                    "residue_name_1",
                    "label",
                ]
            )
            for pdb_id in ids:
                rec = load_label(label_root, pdb_id)
                seq = label_to_sequence(rec)
                labels = rec["labels"].detach().cpu().tolist()
                chain_ids = rec["chain_ids"]
                residue_indices = rec["residue_indices"]
                insertion_codes = rec["insertion_codes"]
                residue_names = rec["residue_names"]
                if len(seq) != len(labels):
                    raise ValueError(f"{pdb_id}: sequence length {len(seq)} != labels {len(labels)}")
                fasta_records.append((pdb_id, seq))
                for i, (aa, lab) in enumerate(zip(seq, labels)):
                    writer.writerow(
                        [
                            pdb_id,
                            i,
                            chain_ids[i],
                            residue_indices[i],
                            insertion_codes[i],
                            residue_names[i],
                            aa,
                            int(lab),
                        ]
                    )
                summary_rows.append(
                    {
                        "split": split,
                        "pdb_id": pdb_id,
                        "length": len(seq),
                        "n_positive": int(sum(labels)),
                        "n_negative": int(len(labels) - sum(labels)),
                        "n_chains": len(set(chain_ids)),
                    }
                )
        write_fasta(fasta_records, out_dir / f"{split}.fasta")

    with (out_dir / "summary.tsv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "pdb_id", "length", "n_positive", "n_negative", "n_chains"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote PIPENN feature inputs to {out_dir}")


if __name__ == "__main__":
    main()
