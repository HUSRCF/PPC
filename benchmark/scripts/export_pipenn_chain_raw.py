#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def read_chain_labels(path: Path) -> OrderedDict[str, dict[str, list]]:
    chains: OrderedDict[str, dict[str, list]] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"seq_id", "position", "residue", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            seq_id = row["seq_id"]
            pos = int(row["position"])
            rec = chains.setdefault(seq_id, {"residues": [], "labels": [], "positions": []})
            expected = len(rec["residues"]) + 1
            if pos != expected:
                raise ValueError(f"{seq_id}: non-contiguous position {pos}, expected {expected}")
            rec["positions"].append(pos)
            rec["residues"].append(row["residue"])
            rec["labels"].append(int(row["label"]))
    return chains


def write_fasta(chains: OrderedDict[str, dict[str, list]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for seq_id, rec in chains.items():
            seq = "".join(rec["residues"])
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def write_residue_labels(chains: OrderedDict[str, dict[str, list]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seq_id", "row_index", "position", "residue", "label"])
        for seq_id, rec in chains.items():
            for i, (pos, aa, label) in enumerate(zip(rec["positions"], rec["residues"], rec["labels"])):
                writer.writerow([seq_id, i, pos, aa, label])


def write_summary(chains: OrderedDict[str, dict[str, list]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seq_id", "length", "n_positive", "n_negative"])
        for seq_id, rec in chains.items():
            length = len(rec["labels"])
            n_pos = sum(rec["labels"])
            writer.writerow([seq_id, length, n_pos, length - n_pos])


def main() -> None:
    ap = argparse.ArgumentParser(description="Export chain-level PIPENN raw FASTA/labels from PPC chain label table.")
    ap.add_argument("--chain-labels", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-name", default="test")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains = read_chain_labels(Path(args.chain_labels))
    split = args.split_name

    write_fasta(chains, out_dir / f"{split}.fasta")
    write_residue_labels(chains, out_dir / f"{split}_residue_labels.tsv")
    write_summary(chains, out_dir / f"{split}_summary.tsv")

    n_res = sum(len(rec["labels"]) for rec in chains.values())
    n_pos = sum(sum(rec["labels"]) for rec in chains.values())
    max_len = max((len(rec["labels"]) for rec in chains.values()), default=0)
    print(
        f"Wrote {len(chains)} chains / {n_res} residues / {n_pos} positives "
        f"(max_len={max_len}) to {out_dir}"
    )


if __name__ == "__main__":
    main()
