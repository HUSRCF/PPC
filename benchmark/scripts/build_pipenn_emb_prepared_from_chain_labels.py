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
            rec = chains.setdefault(seq_id, {"residues": [], "labels": []})
            expected = len(rec["residues"]) + 1
            pos = int(row["position"])
            if pos != expected:
                raise ValueError(f"{seq_id}: non-contiguous position {pos}, expected {expected}")
            rec["residues"].append(row["residue"])
            rec["labels"].append(int(row["label"]))
    return chains


def csv_seq(values: list[object]) -> str:
    return ",".join(str(v) for v in values)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build minimal PIPENN-EMB prepared CSV from PPC chain labels.")
    ap.add_argument("--chain-labels", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--embedding-dir", default=None, help="Optional .embd directory; if set, emit only chains with embeddings.")
    ap.add_argument("--missing-tsv", default=None)
    args = ap.parse_args()

    chains = read_chain_labels(Path(args.chain_labels))
    emb_dir = Path(args.embedding_dir) if args.embedding_dir else None
    rows: list[dict[str, object]] = []
    missing: list[tuple[str, int, str]] = []

    for seq_id, rec in chains.items():
        length = len(rec["residues"])
        if emb_dir is not None and not (emb_dir / f"{seq_id}.embd").exists():
            missing.append((seq_id, length, "missing_embedding"))
            continue
        seq = csv_seq(rec["residues"])
        norm_len = min(length / 2050.0, 1.0)
        rows.append(
            {
                "uniprot_id": seq_id,
                "sequence": seq,
                "Rlength": length,
                "normalized_length": csv_seq([norm_len] * length),
                "any_interface": csv_seq(rec["labels"]),
            }
        )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uniprot_id", "sequence", "Rlength", "normalized_length", "any_interface"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.missing_tsv:
        miss_path = Path(args.missing_tsv)
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        with miss_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["seq_id", "length", "reason"])
            writer.writerows(missing)

    n_res = sum(int(row["Rlength"]) for row in rows)
    n_pos = 0
    for row in rows:
        n_pos += sum(int(x) for x in str(row["any_interface"]).split(",") if x)
    print(f"Wrote {len(rows)} chains / {n_res} residues / {n_pos} positives to {out}")
    if missing:
        print(f"Skipped {len(missing)} chains without embeddings")


if __name__ == "__main__":
    main()
