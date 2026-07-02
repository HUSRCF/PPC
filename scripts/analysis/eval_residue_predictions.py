#!/usr/bin/env python3
"""Evaluate residue-level site predictions against strict labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_curve, roc_auc_score


def load_labels(path: Path) -> dict[tuple[str, int], int]:
    labels: dict[tuple[str, int], int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            labels[(row["seq_id"], int(row["position"]))] = int(row["label"])
    return labels


def confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = y_score >= threshold
    yb = y_true.astype(bool)
    tp = int((pred & yb).sum())
    fp = int((pred & ~yb).sum())
    tn = int((~pred & ~yb).sum())
    fn = int((~pred & yb).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc = float(matthews_corrcoef(y_true, pred.astype(int))) if len(np.unique(y_true)) > 1 else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def top_fraction_metrics(by_chain: dict[str, list[tuple[int, float]]], frac: float) -> dict[str, float]:
    total_tp = total_k = total_pos = 0
    chain_precisions = []
    chain_recalls = []
    chain_enrichments = []
    for values in by_chain.values():
        n = len(values)
        if n == 0:
            continue
        k = max(1, int(math.ceil(n * frac)))
        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
        top = sorted_values[:k]
        pos = sum(label for label, _ in values)
        tp = sum(label for label, _ in top)
        total_tp += tp
        total_k += k
        total_pos += pos
        precision = tp / k if k else 0.0
        recall = tp / pos if pos else 0.0
        base = pos / n if n else 0.0
        enrichment = precision / base if base else 0.0
        chain_precisions.append(precision)
        chain_recalls.append(recall)
        chain_enrichments.append(enrichment)
    micro_precision = total_tp / total_k if total_k else 0.0
    micro_recall = total_tp / total_pos if total_pos else 0.0
    micro_base = total_pos / sum(len(v) for v in by_chain.values()) if by_chain else 0.0
    return {
        f"top_{int(frac * 100)}pct_precision_micro": micro_precision,
        f"top_{int(frac * 100)}pct_recall_micro": micro_recall,
        f"top_{int(frac * 100)}pct_enrichment_micro": micro_precision / micro_base if micro_base else 0.0,
        f"top_{int(frac * 100)}pct_precision_chain_macro": float(np.mean(chain_precisions)) if chain_precisions else 0.0,
        f"top_{int(frac * 100)}pct_recall_chain_macro": float(np.mean(chain_recalls)) if chain_recalls else 0.0,
        f"top_{int(frac * 100)}pct_enrichment_chain_macro": float(np.mean(chain_enrichments)) if chain_enrichments else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    y_true = []
    y_score = []
    by_chain: dict[str, list[tuple[int, float]]] = defaultdict(list)
    missing_labels = []
    with args.predictions.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row["seq_id"], int(row["position"]))
            if key not in labels:
                missing_labels.append(key)
                continue
            label = labels[key]
            score = float(row["score"])
            y_true.append(label)
            y_score.append(score)
            by_chain[row["seq_id"]].append((label, score))

    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    thresholds = [round(i / 100, 2) for i in range(1, 100)]
    sweep = [confusion(yt, ys, t) for t in thresholds]
    at_05 = confusion(yt, ys, 0.5)
    best_f1 = max(sweep, key=lambda row: row["f1"])
    best_mcc = max(sweep, key=lambda row: row["mcc"])
    result = {
        "method": args.method,
        "predictions_tsv": str(args.predictions),
        "label_table_tsv": str(args.labels),
        "n_chains": len(by_chain),
        "n_residues": int(len(yt)),
        "n_positive": int(yt.sum()),
        "positive_rate_micro": float(yt.mean()) if len(yt) else 0.0,
        "tp": at_05["tp"],
        "fp": at_05["fp"],
        "tn": at_05["tn"],
        "fn": at_05["fn"],
        "precision": at_05["precision"],
        "recall": at_05["recall"],
        "f1": at_05["f1"],
        "mcc": at_05["mcc"],
        "threshold_sweep": sweep,
        "threshold_0_5": 0.5,
        "f1_at_0_5": at_05["f1"],
        "mcc_at_0_5": at_05["mcc"],
        "best_threshold": best_f1["threshold"],
        "f1_best_threshold": best_f1["f1"],
        "precision_best_threshold": best_f1["precision"],
        "recall_best_threshold": best_f1["recall"],
        "mcc_at_best_f1_threshold": best_f1["mcc"],
        "best_mcc_threshold": best_mcc["threshold"],
        "mcc_best_threshold": best_mcc["mcc"],
        "f1_at_best_mcc_threshold": best_mcc["f1"],
        "auroc": float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else None,
        "pr_auc": float(average_precision_score(yt, ys)) if len(np.unique(yt)) > 1 else None,
        "missing_label_count": len(missing_labels),
        "missing_label_examples": [list(x) for x in missing_labels[:20]],
    }
    for frac in (0.05, 0.10, 0.20):
        result.update(top_fraction_metrics(by_chain, frac))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
