#!/usr/bin/env python3
"""Compute chain-macro average precision from aligned residue predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def chain_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        raise ValueError("Cannot compute AP for an empty chain")
    if labels.shape != scores.shape:
        raise ValueError("Label/score shape mismatch")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values")
    if int(labels.sum()) == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def load_npz(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = np.load(path, allow_pickle=False)
    required = {"seq_ids", "labels", "scores"}
    if not required.issubset(payload.files):
        raise ValueError(f"{path} lacks required arrays: {sorted(required - set(payload.files))}")
    seq_ids = payload["seq_ids"]
    labels = payload["labels"]
    scores = payload["scores"]
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError(f"Invalid flat label/score arrays in {path}")
    if "offsets" in payload.files:
        offsets = payload["offsets"]
    elif "lengths" in payload.files:
        lengths = payload["lengths"].astype(np.int64, copy=False)
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
    else:
        raise ValueError(f"{path} has neither offsets nor lengths")
    if offsets.ndim != 1 or len(offsets) != len(seq_ids) + 1:
        raise ValueError(f"Invalid offsets in {path}")
    if np.any(offsets[1:] < offsets[:-1]) or int(offsets[0]) != 0 or int(offsets[-1]) != labels.size:
        raise ValueError(f"Offsets do not monotonically span residues in {path}")

    chains: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, seq_id_value in enumerate(seq_ids):
        seq_id = str(seq_id_value)
        if seq_id in chains:
            raise ValueError(f"Duplicate seq_id {seq_id!r} in {path}")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        chains[seq_id] = (labels[start:stop], scores[start:stop])
    return chains


def read_tsv(path: Path, value_column: str) -> dict[str, list[tuple[int, str, str]]]:
    chains: dict[str, list[tuple[int, str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"seq_id", "position", "residue", value_column}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            seq_id = row["seq_id"]
            chains.setdefault(seq_id, []).append((int(row["position"]), row["residue"], row[value_column]))
    return chains


def load_tsv_pair(scores_path: Path, labels_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    score_rows = read_tsv(scores_path, "score")
    label_rows = read_tsv(labels_path, "label")
    if set(score_rows) != set(label_rows):
        missing_scores = sorted(set(label_rows) - set(score_rows))[:10]
        extra_scores = sorted(set(score_rows) - set(label_rows))[:10]
        raise ValueError(f"Chain-set mismatch: missing_scores={missing_scores}, extra_scores={extra_scores}")

    chains: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seq_id in sorted(label_rows):
        scores = score_rows[seq_id]
        labels = label_rows[seq_id]
        if len(scores) != len(labels):
            raise ValueError(f"Length mismatch for {seq_id}: scores={len(scores)}, labels={len(labels)}")
        positions = [row[0] for row in labels]
        if len(set(positions)) != len(positions) or positions != sorted(positions):
            raise ValueError(f"Label positions must be unique and increasing for {seq_id}")
        score_values: list[float] = []
        label_values: list[int] = []
        for score_row, label_row in zip(scores, labels):
            score_position, score_residue, score = score_row
            label_position, label_residue, label = label_row
            if score_position != label_position or score_residue != label_residue:
                raise ValueError(
                    f"Residue alignment mismatch for {seq_id}: "
                    f"score=({score_position},{score_residue}) label=({label_position},{label_residue})"
                )
            score_values.append(float(score))
            label_values.append(int(label))
        chains[seq_id] = (
            np.asarray(label_values, dtype=np.uint8),
            np.asarray(score_values, dtype=np.float32),
        )
    return chains


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scores-npz", type=Path)
    inputs.add_argument("--scores-tsv", type=Path)
    parser.add_argument("--labels-tsv", type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-per-chain", required=True, type=Path)
    args = parser.parse_args()

    if args.scores_npz is not None:
        if args.labels_tsv is not None:
            parser.error("--labels-tsv is only valid with --scores-tsv")
        chains = load_npz(args.scores_npz)
        source = {"scores_npz": str(args.scores_npz)}
    else:
        if args.labels_tsv is None:
            parser.error("--labels-tsv is required with --scores-tsv")
        chains = load_tsv_pair(args.scores_tsv, args.labels_tsv)
        source = {"scores_tsv": str(args.scores_tsv), "labels_tsv": str(args.labels_tsv)}

    rows: list[dict[str, object]] = []
    for seq_id in sorted(chains):
        labels, scores = chains[seq_id]
        n_positive = int(labels.sum())
        rows.append(
            {
                "seq_id": seq_id,
                "n_residues": int(labels.size),
                "n_positive": n_positive,
                "n_negative": int(labels.size - n_positive),
                "average_precision": chain_ap(labels, scores),
            }
        )

    macro_ap = float(np.mean([float(row["average_precision"]) for row in rows]))
    summary = {
        "method": args.method,
        "definition": "unweighted mean of per-chain average precision; all-negative chains contribute AP=0",
        "chain_ap_macro": macro_ap,
        "n_chains": len(rows),
        "n_residues": sum(int(row["n_residues"]) for row in rows),
        "n_positive": sum(int(row["n_positive"]) for row in rows),
        "n_all_negative_chains": sum(int(row["n_positive"]) == 0 for row in rows),
        "n_all_positive_chains": sum(int(row["n_negative"]) == 0 for row in rows),
        "source": source,
    }
    args.output_per_chain.parent.mkdir(parents=True, exist_ok=True)
    with args.output_per_chain.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
