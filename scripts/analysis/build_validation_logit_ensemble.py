#!/usr/bin/env python3
"""Build a validation-locked two-model logit ensemble.

The blend weight is selected by validation chain-macro average precision. The
operating threshold is selected only after the blend is locked, using validation
F1 with MCC as a tie-breaker. Test scores never participate in either decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class PackedScores:
    seq_ids: list[str]
    lengths: np.ndarray
    labels: np.ndarray
    left: np.ndarray
    right: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scores(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = np.load(path, allow_pickle=False)
    required = {"seq_ids", "labels", "scores"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"Missing arrays in {path}: {missing}")
    seq_ids = payload["seq_ids"]
    labels = payload["labels"]
    scores = payload["scores"]
    if "offsets" in payload.files:
        offsets = payload["offsets"].astype(np.int64, copy=False)
    elif "lengths" in payload.files:
        lengths = payload["lengths"].astype(np.int64, copy=False)
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    else:
        raise ValueError(f"{path} must contain offsets or lengths")
    if offsets.ndim != 1 or offsets.size != seq_ids.size + 1:
        raise ValueError(f"Invalid offsets in {path}")
    if np.any(offsets[1:] < offsets[:-1]) or int(offsets[0]) != 0:
        raise ValueError(f"Non-monotonic offsets in {path}")
    if labels.shape != scores.shape or int(offsets[-1]) != labels.size:
        raise ValueError(f"Packed array shape mismatch in {path}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, seq_id_value in enumerate(seq_ids):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        seq_id = str(seq_id_value)
        if seq_id in result:
            raise ValueError(f"Duplicate seq_id {seq_id!r} in {path}")
        result[seq_id] = (
            labels[start:stop].astype(np.uint8, copy=False),
            scores[start:stop].astype(np.float64, copy=False),
        )
    return result


def align(left_path: Path, right_path: Path) -> PackedScores:
    left = load_scores(left_path)
    right = load_scores(right_path)
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))[:10]
        only_right = sorted(set(right) - set(left))[:10]
        raise ValueError(f"Prediction chain sets differ: only_left={only_left}, only_right={only_right}")
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
        labels.append(left_labels)
        left_scores.append(left_chain_scores)
        right_scores.append(right_chain_scores)
    return PackedScores(
        seq_ids=seq_ids,
        lengths=np.asarray(lengths, dtype=np.int32),
        labels=np.concatenate(labels),
        left=np.concatenate(left_scores),
        right=np.concatenate(right_scores),
    )


def parse_grid(text: str) -> np.ndarray:
    text = text.strip()
    if ":" in text:
        start, stop, step = (float(value) for value in text.split(":"))
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid grid {text!r}")
        count = int(math.floor((stop - start) / step + 1.0e-9)) + 1
        values = start + np.arange(count, dtype=np.float64) * step
    else:
        values = np.asarray([float(value) for value in text.split(",") if value.strip()], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid grid {text!r}")
    return np.unique(np.round(values, 12))


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1.0e-7, 1.0 - 1.0e-7)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def blend(left_logits: np.ndarray, right_logits: np.ndarray, alpha_right: float) -> np.ndarray:
    return sigmoid((1.0 - alpha_right) * left_logits + alpha_right * right_logits)


def offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))


def chain_macro_ap(labels: np.ndarray, scores: np.ndarray, lengths: np.ndarray) -> dict[str, float | int]:
    values: list[float] = []
    all_negative = 0
    all_positive = 0
    offset = 0
    for length_value in lengths:
        length = int(length_value)
        chain_labels = labels[offset : offset + length]
        chain_scores = scores[offset : offset + length]
        offset += length
        positives = int(chain_labels.sum())
        if positives == 0:
            values.append(0.0)
            all_negative += 1
        else:
            values.append(float(average_precision_score(chain_labels, chain_scores)))
        if positives == length:
            all_positive += 1
    return {
        "chain_ap_macro": float(np.mean(values)) if values else 0.0,
        "chain_ap_n_chains": len(values),
        "chain_ap_all_negative": all_negative,
        "chain_ap_all_positive": all_positive,
    }


def pooled_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "auprc": float(auc(recall[::-1], precision[::-1])),
        "auroc": float(roc_auc_score(labels, scores)),
    }


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
        "threshold": float(threshold),
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


def select_threshold(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, float | int]:
    rows = [confusion_metrics(labels, scores, float(threshold)) for threshold in thresholds]
    return max(rows, key=lambda row: (row["f1"], row["mcc"], -abs(row["threshold"] - 0.5)))


def chain_task_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    top_precision: dict[int, list[float]] = {20: [], 10: [], 5: []}
    top_hits: dict[int, int] = {20: 0, 10: 0, 5: 0}
    top_totals: dict[int, int] = {20: 0, 10: 0, 5: 0}
    raw_ratio_errors: list[float] = []
    thresholded_ratio_errors: list[float] = []
    offset = 0
    for length_value in lengths:
        length = int(length_value)
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
    return result


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        **pooled_ranking_metrics(labels, scores),
        **chain_macro_ap(labels, scores, lengths),
        "selected_threshold": confusion_metrics(labels, scores, threshold),
        "threshold_0_5": confusion_metrics(labels, scores, 0.5),
        "threshold_0_6": confusion_metrics(labels, scores, 0.6),
    }
    result.update(chain_task_metrics(labels, scores, lengths, threshold))
    return result


def validation_candidate(
    labels: np.ndarray,
    scores: np.ndarray,
    lengths: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, object]:
    threshold_metrics = select_threshold(labels, scores, thresholds)
    return evaluate(labels, scores, lengths, float(threshold_metrics["threshold"]))


def save_predictions(path: Path, packed: PackedScores, scores: np.ndarray) -> None:
    np.savez_compressed(
        path,
        seq_ids=np.asarray(packed.seq_ids),
        lengths=packed.lengths,
        offsets=offsets_from_lengths(packed.lengths),
        labels=packed.labels,
        scores=scores.astype(np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-val", required=True, type=Path)
    parser.add_argument("--right-val", required=True, type=Path)
    parser.add_argument("--left-test", required=True, type=Path)
    parser.add_argument("--right-test", required=True, type=Path)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--alpha-grid", default="0:1:0.05")
    parser.add_argument("--threshold-grid", default="0.01:0.99:0.01")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    reserved_names = {"selected", "unweighted"}
    if args.left_name == args.right_name or args.left_name in reserved_names or args.right_name in reserved_names:
        raise ValueError(
            f"left-name and right-name must be distinct and cannot use {sorted(reserved_names)}"
        )

    val = align(args.left_val, args.right_val)
    test = align(args.left_test, args.right_test)
    alphas = parse_grid(args.alpha_grid)
    thresholds = parse_grid(args.threshold_grid)
    if np.any((alphas < 0.0) | (alphas > 1.0)):
        raise ValueError("alpha-grid values must lie in [0, 1]")
    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("threshold-grid values must lie in [0, 1]")

    val_left_logits = logit(val.left)
    val_right_logits = logit(val.right)
    alpha_rows: list[dict[str, float]] = []
    for alpha in alphas:
        scores = blend(val_left_logits, val_right_logits, float(alpha))
        alpha_rows.append(
            {
                "alpha_right": float(alpha),
                **pooled_ranking_metrics(val.labels, scores),
                **chain_macro_ap(val.labels, scores, val.lengths),
            }
        )
    selected_alpha_row = max(
        alpha_rows,
        key=lambda row: (
            row["chain_ap_macro"],
            row["average_precision"],
            -abs(row["alpha_right"] - 0.5),
        ),
    )
    selected_alpha = float(selected_alpha_row["alpha_right"])

    val_scores = {
        args.left_name: val.left,
        args.right_name: val.right,
        "unweighted": blend(val_left_logits, val_right_logits, 0.5),
        "selected": blend(val_left_logits, val_right_logits, selected_alpha),
    }
    test_left_logits = logit(test.left)
    test_right_logits = logit(test.right)
    test_scores = {
        args.left_name: test.left,
        args.right_name: test.right,
        "unweighted": blend(test_left_logits, test_right_logits, 0.5),
        "selected": blend(test_left_logits, test_right_logits, selected_alpha),
    }

    validation: dict[str, object] = {}
    test_results: dict[str, object] = {}
    frozen_thresholds: dict[str, float] = {}
    for name, scores in val_scores.items():
        validation[name] = validation_candidate(val.labels, scores, val.lengths, thresholds)
        threshold = float(validation[name]["selected_threshold"]["threshold"])
        frozen_thresholds[name] = threshold
        test_results[name] = evaluate(test.labels, test_scores[name], test.lengths, threshold)

    selected_val = validation["selected"]
    unweighted_val = validation["unweighted"]
    right_val = validation[args.right_name]
    promotion_checks = {
        "selected_alpha_is_learned": 0.0 < selected_alpha < 1.0 and not math.isclose(selected_alpha, 0.5),
        "macro_ap_gt_unweighted": selected_val["chain_ap_macro"] > unweighted_val["chain_ap_macro"],
        "macro_ap_gt_right": selected_val["chain_ap_macro"] > right_val["chain_ap_macro"],
        "pooled_ap_gt_unweighted": selected_val["average_precision"] > unweighted_val["average_precision"],
        "pooled_ap_gt_right": selected_val["average_precision"] > right_val["average_precision"],
        "f1_not_below_unweighted": selected_val["selected_threshold"]["f1"] >= unweighted_val["selected_threshold"]["f1"],
        "f1_not_below_right": selected_val["selected_threshold"]["f1"] >= right_val["selected_threshold"]["f1"],
        "mcc_not_below_unweighted": selected_val["selected_threshold"]["mcc"] >= unweighted_val["selected_threshold"]["mcc"],
        "mcc_not_below_right": selected_val["selected_threshold"]["mcc"] >= right_val["selected_threshold"]["mcc"],
    }
    learned_blend_promoted = all(promotion_checks.values())
    fallback_method = max(
        (args.left_name, args.right_name, "unweighted"),
        key=lambda name: (
            validation[name]["chain_ap_macro"],
            validation[name]["average_precision"],
            validation[name]["selected_threshold"]["f1"],
        ),
    )
    selected_method = "selected" if learned_blend_promoted else fallback_method

    summary = {
        "method": "validation_locked_two_model_logit_ensemble",
        "selection_policy": (
            "alpha selected by validation chain-macro AP; pooled validation AP tie-break; "
            "threshold selected afterward by validation F1 with MCC tie-break; test hidden"
        ),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "alpha_grid": [float(value) for value in alphas],
        "threshold_grid": [float(value) for value in thresholds],
        "selected_alpha_left": 1.0 - selected_alpha,
        "selected_alpha_right": selected_alpha,
        "alpha_sweep": alpha_rows,
        "frozen_thresholds": frozen_thresholds,
        "promotion_checks": promotion_checks,
        "learned_blend_promoted": learned_blend_promoted,
        "fallback_variant": fallback_method,
        "recommended_variant": selected_method,
        "validation": validation,
        "test": test_results,
        "coverage": {
            "validation_chains": len(val.seq_ids),
            "validation_residues": int(val.labels.size),
            "test_chains": len(test.seq_ids),
            "test_residues": int(test.labels.size),
        },
        "sources": {
            "left_val": {"path": str(args.left_val), "sha256": sha256_file(args.left_val)},
            "right_val": {"path": str(args.right_val), "sha256": sha256_file(args.right_val)},
            "left_test": {"path": str(args.left_test), "sha256": sha256_file(args.left_test)},
            "right_test": {"path": str(args.right_test), "sha256": sha256_file(args.right_test)},
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_logit_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    save_predictions(args.output_dir / "selected_test_predictions.npz", test, test_scores["selected"])
    save_predictions(args.output_dir / "unweighted_test_predictions.npz", test, test_scores["unweighted"])
    save_predictions(args.output_dir / "recommended_test_predictions.npz", test, test_scores[selected_method])
    recommended = summary["test"][selected_method]
    recommended_threshold = recommended["selected_threshold"]
    markdown = f"""# Validation-Locked Two-Model Logit Ensemble

- Left / right: `{args.left_name}` / `{args.right_name}`
- Validation-selected blend: `{1.0 - selected_alpha:.2f}` left + `{selected_alpha:.2f}` right in logit space
- Learned blend promotion: `{learned_blend_promoted}`; recommended variant: `{selected_method}`
- Test chain-macro AP / pooled AP / AUROC: `{recommended['chain_ap_macro']:.4f}` / `{recommended['average_precision']:.4f}` / `{recommended['auroc']:.4f}`
- Test frozen-threshold F1 / MCC: `{recommended_threshold['f1']:.4f}` / `{recommended_threshold['mcc']:.4f}` at `{recommended_threshold['threshold']:.2f}`

Alpha and threshold were selected exclusively on validation. Test did not participate in selection.
"""
    (args.output_dir / "validation_logit_ensemble_summary.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
