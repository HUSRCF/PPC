#!/usr/bin/env python3
"""Convert ScanNet per-input CSV files into strict residue prediction TSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    labels: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            labels[row["seq_id"]].append(row)
    return dict(labels)


def find_prediction_csv(root: Path, input_id: str) -> Path | None:
    direct = root / input_id
    if direct.exists():
        matches = sorted(direct.glob(f"**/predictions_{input_id}.csv"))
        if matches:
            return matches[0]
    matches = sorted(root.glob(f"**/predictions_{input_id}.csv"))
    return matches[0] if matches else None


def read_scores(path: Path) -> tuple[list[str], list[float]]:
    residues: list[str] = []
    scores: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        score_col = "Binding site probability"
        for row in reader:
            residues.append(row.get("Sequence", ""))
            scores.append(float(row[score_col]))
    return residues, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--method", default="scannet_noMSA_strict_native")
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()

    chain_rows = read_tsv(args.chain_manifest)
    labels_by_seq = load_labels(args.labels)
    rows_by_input: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in chain_rows:
        if row.get("has_pdb") == "1":
            rows_by_input[row["input_id"]].append(row)

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    missing_csv: list[str] = []
    missing_labels: list[str] = []
    length_mismatch: list[dict[str, object]] = []
    residue_mismatch: list[dict[str, object]] = []
    n_chains = 0
    n_residues = 0

    with args.out_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_id", "position", "residue", "score", "method"], delimiter="\t")
        writer.writeheader()
        for input_id in sorted(rows_by_input):
            pred_csv = find_prediction_csv(args.predictions_root, input_id)
            if pred_csv is None:
                missing_csv.append(input_id)
                continue
            pred_residues, scores = read_scores(pred_csv)
            for chain_row in rows_by_input[input_id]:
                seq_id = chain_row["seq_id"]
                label_rows = labels_by_seq.get(seq_id)
                if not label_rows:
                    missing_labels.append(seq_id)
                    continue
                if len(label_rows) != len(scores):
                    length_mismatch.append(
                        {
                            "seq_id": seq_id,
                            "input_id": input_id,
                            "label_len": len(label_rows),
                            "score_len": len(scores),
                            "pred_csv": str(pred_csv),
                        }
                    )
                    continue
                bad = [i for i, (lab, pred_res) in enumerate(zip(label_rows, pred_residues), start=1) if lab["residue"] != pred_res]
                if bad:
                    residue_mismatch.append(
                        {
                            "seq_id": seq_id,
                            "input_id": input_id,
                            "n_bad": len(bad),
                            "first_bad_position": bad[0],
                            "pred_csv": str(pred_csv),
                        }
                    )
                    continue
                for lab, score in zip(label_rows, scores):
                    writer.writerow(
                        {
                            "seq_id": seq_id,
                            "position": lab["position"],
                            "residue": lab["residue"],
                            "score": score,
                            "method": args.method,
                        }
                    )
                n_chains += 1
                n_residues += len(scores)

    summary = {
        "method": args.method,
        "predictions_root": str(args.predictions_root),
        "chain_manifest": str(args.chain_manifest),
        "labels": str(args.labels),
        "out_tsv": str(args.out_tsv),
        "input_ids_expected": len(rows_by_input),
        "missing_csv_count": len(missing_csv),
        "missing_csv_examples": missing_csv[:30],
        "missing_labels_count": len(missing_labels),
        "missing_labels_examples": missing_labels[:30],
        "length_mismatch_count": len(length_mismatch),
        "length_mismatch_examples": length_mismatch[:30],
        "residue_mismatch_count": len(residue_mismatch),
        "residue_mismatch_examples": residue_mismatch[:30],
        "n_chains_written": n_chains,
        "n_residues_written": n_residues,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not (missing_csv or length_mismatch or residue_mismatch) else 2


if __name__ == "__main__":
    raise SystemExit(main())
