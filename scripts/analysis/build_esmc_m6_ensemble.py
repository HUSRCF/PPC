#!/usr/bin/env python3
"""Select an M0/M2 logit ensemble on validation and evaluate once on test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scores(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = np.load(path, allow_pickle=False)
    seq_ids = payload["seq_ids"]
    offsets = payload["offsets"]
    labels = payload["labels"]
    scores = payload["scores"]
    if offsets.ndim != 1 or len(offsets) != len(seq_ids) + 1:
        raise ValueError(f"Invalid offsets in {path}")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"Offsets are not monotonic in {path}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != labels.size:
        raise ValueError(f"Offsets do not span labels in {path}")
    if labels.shape != scores.shape:
        raise ValueError(f"Label/score shape mismatch in {path}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, seq_id_value in enumerate(seq_ids):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        seq_id = str(seq_id_value)
        if seq_id in result:
            raise ValueError(f"Duplicate seq_id {seq_id!r} in {path}")
        result[seq_id] = (labels[start:stop], scores[start:stop])
    return result


def align(
    left: dict[str, tuple[np.ndarray, np.ndarray]],
    right: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[str], list[int], np.ndarray, np.ndarray, np.ndarray]:
    if set(left) != set(right):
        raise ValueError("Prediction chain sets differ")
    seq_ids = sorted(left)
    lengths: list[int] = []
    labels: list[np.ndarray] = []
    left_scores: list[np.ndarray] = []
    right_scores: list[np.ndarray] = []
    for seq_id in seq_ids:
        left_labels, left_chain_scores = left[seq_id]
        right_labels, right_chain_scores = right[seq_id]
        if not np.array_equal(left_labels, right_labels):
            raise ValueError(f"Label mismatch for {seq_id}")
        if left_chain_scores.shape != right_chain_scores.shape:
            raise ValueError(f"Score length mismatch for {seq_id}")
        lengths.append(int(left_labels.size))
        labels.append(left_labels.astype(np.uint8, copy=False))
        left_scores.append(left_chain_scores.astype(np.float64, copy=False))
        right_scores.append(right_chain_scores.astype(np.float64, copy=False))
    return seq_ids, lengths, np.concatenate(labels), np.concatenate(left_scores), np.concatenate(right_scores)


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def confusion_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = scores >= threshold
    positives = labels == 1
    tp = int(np.count_nonzero(predictions & positives))
    fp = int(np.count_nonzero(predictions & ~positives))
    fn = int(np.count_nonzero(~predictions & positives))
    tn = int(np.count_nonzero(~predictions & ~positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "accuracy": (tp + tn) / labels.size,
    }


def sweep_thresholds(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> list[dict[str, float | int]]:
    # Bucket assignment is exact for the supplied threshold grid and avoids a
    # full O(N * thresholds) comparison matrix.
    buckets = np.searchsorted(thresholds, scores, side="right")
    positives = np.bincount(buckets, weights=labels, minlength=len(thresholds) + 1)
    totals = np.bincount(buckets, minlength=len(thresholds) + 1)
    negatives = totals - positives
    tp = np.cumsum(positives[::-1])[::-1][1:]
    fp = np.cumsum(negatives[::-1])[::-1][1:]
    total_positive = float(labels.sum())
    total_negative = float(labels.size - labels.sum())
    rows: list[dict[str, float | int]] = []
    for index, threshold in enumerate(thresholds):
        tp_i = int(tp[index])
        fp_i = int(fp[index])
        fn_i = int(total_positive - tp_i)
        tn_i = int(total_negative - fp_i)
        precision = tp_i / (tp_i + fp_i) if tp_i + fp_i else 0.0
        recall = tp_i / (tp_i + fn_i) if tp_i + fn_i else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        denominator = math.sqrt((tp_i + fp_i) * (tp_i + fn_i) * (tn_i + fp_i) * (tn_i + fn_i))
        mcc = (tp_i * tn_i - fp_i * fn_i) / denominator if denominator else 0.0
        rows.append({"threshold": float(threshold), "f1": f1, "mcc": mcc})
    return rows


def chain_metrics(labels: np.ndarray, scores: np.ndarray, lengths: list[int], threshold: float) -> dict[str, float | int]:
    top_precision: dict[int, list[float]] = {20: [], 10: [], 5: []}
    top_hits: dict[int, int] = {20: 0, 10: 0, 5: 0}
    top_totals: dict[int, int] = {20: 0, 10: 0, 5: 0}
    raw_ratio_errors: list[float] = []
    thresholded_ratio_errors: list[float] = []
    offset = 0
    for length in lengths:
        chain_labels = labels[offset : offset + length]
        chain_scores = scores[offset : offset + length]
        offset += length
        true_ratio = float(chain_labels.mean())
        raw_ratio_errors.append(abs(float(chain_scores.mean()) - true_ratio))
        thresholded_ratio_errors.append(abs(float(np.mean(chain_scores >= threshold)) - true_ratio))
        order = np.argsort(-chain_scores, kind="stable")
        for denominator, values in top_precision.items():
            k = max(1, length // denominator)
            hits = int(chain_labels[order[:k]].sum())
            values.append(hits / k)
            top_hits[denominator] += hits
            top_totals[denominator] += k
    result: dict[str, float | int] = {
        "chain_effect_site_ratio_mae_raw": float(np.mean(raw_ratio_errors)),
        "chain_effect_site_ratio_mae_thresholded": float(np.mean(thresholded_ratio_errors)),
    }
    for denominator in (20, 10, 5):
        result[f"L/{denominator}_ACC_macro"] = float(np.mean(top_precision[denominator]))
        result[f"L/{denominator}_ACC_micro"] = top_hits[denominator] / top_totals[denominator]
        result[f"L/{denominator}_topk_total"] = top_totals[denominator]
        result[f"L/{denominator}_hits_total"] = top_hits[denominator]
    return result


def evaluate(labels: np.ndarray, scores: np.ndarray, lengths: list[int], threshold: float) -> dict[str, object]:
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    exact_f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_f1_index = int(np.nanargmax(exact_f1))
    best_f1_threshold = 1.0 if best_f1_index == len(pr_thresholds) else float(pr_thresholds[best_f1_index])
    mcc_sweep = sweep_thresholds(labels, scores, np.linspace(0.0, 1.0, 1001))
    best_mcc = max(mcc_sweep, key=lambda row: (row["mcc"], row["f1"], -abs(row["threshold"] - 0.5)))
    result: dict[str, object] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(auc(recall[::-1], precision[::-1])),
        "average_precision": float(average_precision_score(labels, scores)),
        "selected_threshold": confusion_metrics(labels, scores, threshold),
        "threshold_0_5": confusion_metrics(labels, scores, 0.5),
        "threshold_0_6": confusion_metrics(labels, scores, 0.6),
        "test_oracle_diagnostics": {
            "best_F1": float(exact_f1[best_f1_index]),
            "best_F1_threshold": best_f1_threshold,
            "best_MCC": float(best_mcc["mcc"]),
            "best_MCC_threshold": float(best_mcc["threshold"]),
        },
    }
    result.update(chain_metrics(labels, scores, lengths, threshold))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-val", required=True, type=Path)
    parser.add_argument("--m2-val", required=True, type=Path)
    parser.add_argument("--m0-test", required=True, type=Path)
    parser.add_argument("--m2-test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    val_ids, val_lengths, val_labels, m0_val, m2_val = align(load_scores(args.m0_val), load_scores(args.m2_val))
    test_ids, test_lengths, test_labels, m0_test, m2_test = align(load_scores(args.m0_test), load_scores(args.m2_test))
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    alphas = np.round(np.arange(0.00, 1.001, 0.05), 2)
    m0_val_logits = logit(m0_val)
    m2_val_logits = logit(m2_val)

    candidates: list[dict[str, float]] = []
    for alpha in alphas:
        scores = sigmoid((1.0 - alpha) * m0_val_logits + alpha * m2_val_logits)
        for metric in sweep_thresholds(val_labels, scores, thresholds):
            candidates.append({"alpha_m2": float(alpha), **metric})
    selected = max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["mcc"],
            -abs(row["alpha_m2"] - 0.5),
            -abs(row["threshold"] - 0.5),
        ),
    )
    alpha = selected["alpha_m2"]
    threshold = selected["threshold"]
    val_scores = sigmoid((1.0 - alpha) * m0_val_logits + alpha * m2_val_logits)
    test_scores = sigmoid((1.0 - alpha) * logit(m0_test) + alpha * logit(m2_test))
    summary = {
        "method": "m6_validation_selected_m0_m2_logit_ensemble",
        "selection_policy": "alpha and threshold selected only on validation F1; tie break MCC, alpha proximity to 0.5, threshold proximity to 0.5",
        "alpha_grid": [float(value) for value in alphas],
        "threshold_grid": [float(value) for value in thresholds],
        "selected_alpha_m0": 1.0 - alpha,
        "selected_alpha_m2": alpha,
        "selected_validation_threshold": threshold,
        "validation": evaluate(val_labels, val_scores, val_lengths, threshold),
        "test": evaluate(test_labels, test_scores, test_lengths, threshold),
        "coverage": {
            "validation_chains": len(val_ids),
            "validation_residues": int(val_labels.size),
            "test_chains": len(test_ids),
            "test_residues": int(test_labels.size),
        },
        "sources": {
            "m0_val": {"path": str(args.m0_val), "sha256": sha256_file(args.m0_val)},
            "m2_val": {"path": str(args.m2_val), "sha256": sha256_file(args.m2_val)},
            "m0_test": {"path": str(args.m0_test), "sha256": sha256_file(args.m0_test)},
            "m2_test": {"path": str(args.m2_test), "sha256": sha256_file(args.m2_test)},
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m6_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "m6_test_predictions.npz",
        seq_ids=np.asarray(test_ids),
        lengths=np.asarray(test_lengths, dtype=np.int32),
        labels=test_labels,
        scores=test_scores.astype(np.float32),
    )
    test = summary["test"]
    selected_test = test["selected_threshold"]
    markdown = f"""# M6 validation-only logit ensemble

- Blend: `{1.0 - alpha:.2f} * logit(M0) + {alpha:.2f} * logit(M2)`
- Validation-selected threshold: `{threshold:.2f}`
- Test F1/MCC at frozen threshold: `{selected_test['f1']:.4f}` / `{selected_test['mcc']:.4f}`
- Test AUPRC/AP/AUROC: `{test['auprc']:.4f}` / `{test['average_precision']:.4f}` / `{test['auroc']:.4f}`
- Test F1/MCC at 0.5: `{test['threshold_0_5']['f1']:.4f}` / `{test['threshold_0_5']['mcc']:.4f}`
- Test F1/MCC at 0.6: `{test['threshold_0_6']['f1']:.4f}` / `{test['threshold_0_6']['mcc']:.4f}`
- Test oracle best F1@threshold / best MCC@threshold (diagnostic only): `{test['test_oracle_diagnostics']['best_F1']:.4f}@{test['test_oracle_diagnostics']['best_F1_threshold']:.4f}` / `{test['test_oracle_diagnostics']['best_MCC']:.4f}@{test['test_oracle_diagnostics']['best_MCC_threshold']:.3f}`
- L/20, L/10, L/5 macro accuracy: `{test['L/20_ACC_macro']:.4f}`, `{test['L/10_ACC_macro']:.4f}`, `{test['L/5_ACC_macro']:.4f}`
- Chain effect-site-ratio MAE (raw mean score / thresholded): `{test['chain_effect_site_ratio_mae_raw']:.4f}` / `{test['chain_effect_site_ratio_mae_thresholded']:.4f}`

Alpha and threshold were selected exclusively on validation. Test was evaluated once with both frozen.
"""
    (args.output_dir / "m6_summary.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
