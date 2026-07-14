#!/usr/bin/env python3
"""Cross-fit a two-logit stacker with SI-component groups kept intact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def _best_thresholds(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find fixed-grid F1/MCC thresholds with one score sort."""

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = y_true[order].astype(np.int64, copy=False)
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_labels, dtype=np.int64)))
    thresholds = np.arange(0.01, 1.0, 0.01)
    split = np.searchsorted(sorted_scores, thresholds, side="left")
    total_positive = int(positive_prefix[-1])
    positives_below = positive_prefix[split]
    tp = total_positive - positives_below
    fn = positives_below
    predicted_positive = y_true.size - split
    fp = predicted_positive - tp
    tn = split - fn
    f1_denom = 2 * tp + fp + fn
    f1_values = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp, dtype=np.float64), where=f1_denom > 0)
    mcc_denom = (
        (tp + fp).astype(np.float64)
        * (tp + fn).astype(np.float64)
        * (tn + fp).astype(np.float64)
        * (tn + fn).astype(np.float64)
    )
    mcc_values = np.divide(
        tp * tn - fp * fn,
        np.sqrt(mcc_denom),
        out=np.zeros_like(tp, dtype=np.float64),
        where=mcc_denom > 0,
    )
    return float(round(thresholds[int(np.argmax(f1_values))], 2)), float(
        round(thresholds[int(np.argmax(mcc_values))], 2)
    )


def _chain_macro_ap(y_true: np.ndarray, scores: np.ndarray, chain_index: np.ndarray) -> tuple[float, int, int]:
    values: list[float] = []
    all_negative = 0
    for chain in np.unique(chain_index):
        selected = chain_index == chain
        labels = y_true[selected]
        if int(labels.sum()) == 0:
            values.append(0.0)
            all_negative += 1
        else:
            values.append(float(average_precision_score(labels, scores[selected])))
    return float(np.mean(values)), len(values), all_negative


def _ranking_metrics(y_true: np.ndarray, scores: np.ndarray, chain_index: np.ndarray) -> dict[str, Any]:
    chain_ap, n_chains, all_negative = _chain_macro_ap(y_true, scores, chain_index)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "chain_ap_macro": chain_ap,
        "chain_ap_n_chains": n_chains,
        "chain_ap_all_negative": all_negative,
    }


def _threshold_metrics(y_true: np.ndarray, f1_predictions: np.ndarray, mcc_predictions: np.ndarray) -> dict[str, float]:
    return {
        "f1_crossfit_threshold": _f1(y_true, f1_predictions),
        "mcc_at_crossfit_f1_threshold": float(matthews_corrcoef(y_true, f1_predictions)),
        "mcc_crossfit_threshold": float(matthews_corrcoef(y_true, mcc_predictions)),
        "f1_at_crossfit_mcc_threshold": _f1(y_true, mcc_predictions),
    }


def _chain_equal_weights(chain_index: np.ndarray) -> np.ndarray:
    counts = np.bincount(chain_index)
    weights = 1.0 / counts[chain_index].astype(np.float64)
    return weights * (weights.size / weights.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--folds-tsv", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--weighting", choices=["residue", "chain_equal"], default="residue")
    args = parser.parse_args()

    payload = np.load(args.input_npz, allow_pickle=False)
    labels = payload["labels"].astype(np.int8, copy=False)
    primary_logit = payload["primary_logit"].astype(np.float64, copy=False)
    secondary_logit = payload["secondary_logit"].astype(np.float64, copy=False)
    chain_index = payload["chain_index"].astype(np.int32, copy=False)
    component_by_chain = payload["component_ids"].astype(str)
    chain_ids = payload["chain_ids"].astype(str)
    if not (labels.size == primary_logit.size == secondary_logit.size == chain_index.size):
        raise ValueError("Aligned logit arrays have inconsistent lengths")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Labels must be binary")
    if chain_index.min() < 0 or chain_index.max() >= chain_ids.size:
        raise ValueError("chain_index is outside chain metadata bounds")
    groups = component_by_chain[chain_index]
    if np.unique(groups).size < args.n_splits:
        raise ValueError("Not enough SI components for requested folds")

    features = np.column_stack([np.clip(primary_logit, -30.0, 30.0), np.clip(secondary_logit, -30.0, 30.0)])
    weights = np.ones(labels.size, dtype=np.float64)
    if args.weighting == "chain_equal":
        weights = _chain_equal_weights(chain_index)
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_scores = np.full(labels.size, np.nan, dtype=np.float64)
    oof_f1_pred = np.zeros(labels.size, dtype=bool)
    oof_mcc_pred = np.zeros(labels.size, dtype=bool)
    primary_f1_pred = np.zeros(labels.size, dtype=bool)
    primary_mcc_pred = np.zeros(labels.size, dtype=bool)
    secondary_f1_pred = np.zeros(labels.size, dtype=bool)
    secondary_mcc_pred = np.zeros(labels.size, dtype=bool)
    fold_by_chain = np.full(chain_ids.size, -1, dtype=np.int16)
    fold_reports: list[dict[str, Any]] = []

    for fold, (train_idx, held_idx) in enumerate(splitter.split(features, labels, groups=groups)):
        train_groups = set(groups[train_idx])
        held_groups = set(groups[held_idx])
        if train_groups & held_groups:
            raise RuntimeError(f"SI component leakage in fold {fold}")
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=args.seed + fold)
        model.fit(features[train_idx], labels[train_idx], sample_weight=weights[train_idx])
        train_scores = model.predict_proba(features[train_idx])[:, 1]
        held_scores = model.predict_proba(features[held_idx])[:, 1]
        oof_scores[held_idx] = held_scores

        stack_f1_threshold, stack_mcc_threshold = _best_thresholds(labels[train_idx], train_scores)
        oof_f1_pred[held_idx] = held_scores >= stack_f1_threshold
        oof_mcc_pred[held_idx] = held_scores >= stack_mcc_threshold
        for logit, f1_out, mcc_out in (
            (primary_logit, primary_f1_pred, primary_mcc_pred),
            (secondary_logit, secondary_f1_pred, secondary_mcc_pred),
        ):
            train_base_scores = _sigmoid(logit[train_idx])
            held_base_scores = _sigmoid(logit[held_idx])
            f1_threshold, mcc_threshold = _best_thresholds(labels[train_idx], train_base_scores)
            f1_out[held_idx] = held_base_scores >= f1_threshold
            mcc_out[held_idx] = held_base_scores >= mcc_threshold

        held_chains = np.unique(chain_index[held_idx])
        fold_by_chain[held_chains] = fold
        fold_reports.append(
            {
                "fold": fold,
                "train_components": len(train_groups),
                "held_components": len(held_groups),
                "held_chains": int(held_chains.size),
                "held_residues": int(held_idx.size),
                "held_positive_residues": int(labels[held_idx].sum()),
                "intercept": float(model.intercept_[0]),
                "primary_coefficient": float(model.coef_[0, 0]),
                "secondary_coefficient": float(model.coef_[0, 1]),
                "stack_f1_threshold_from_train": stack_f1_threshold,
                "stack_mcc_threshold_from_train": stack_mcc_threshold,
            }
        )

    if np.isnan(oof_scores).any() or (fold_by_chain < 0).any():
        raise RuntimeError("Cross-fit did not cover every validation residue and chain")
    primary_scores = _sigmoid(primary_logit)
    secondary_scores = _sigmoid(secondary_logit)
    methods = {
        "primary_esmc": {
            **_ranking_metrics(labels, primary_scores, chain_index),
            **_threshold_metrics(labels, primary_f1_pred, primary_mcc_pred),
        },
        "secondary_esm2": {
            **_ranking_metrics(labels, secondary_scores, chain_index),
            **_threshold_metrics(labels, secondary_f1_pred, secondary_mcc_pred),
        },
        "grouped_oof_logit_stack": {
            **_ranking_metrics(labels, oof_scores, chain_index),
            **_threshold_metrics(labels, oof_f1_pred, oof_mcc_pred),
        },
    }
    methods["grouped_oof_logit_stack"]["delta_chain_ap_vs_primary"] = (
        methods["grouped_oof_logit_stack"]["chain_ap_macro"] - methods["primary_esmc"]["chain_ap_macro"]
    )
    methods["grouped_oof_logit_stack"]["delta_pr_auc_vs_primary"] = (
        methods["grouped_oof_logit_stack"]["pr_auc"] - methods["primary_esmc"]["pr_auc"]
    )
    methods["grouped_oof_logit_stack"]["delta_auroc_vs_primary"] = (
        methods["grouped_oof_logit_stack"]["auroc"] - methods["primary_esmc"]["auroc"]
    )
    report = {
        "scope": "validation_only_grouped_crossfit",
        "input_npz": str(args.input_npz),
        "n_splits": args.n_splits,
        "group_key": "SI component_id",
        "weighting": args.weighting,
        "chains": int(chain_ids.size),
        "components": int(np.unique(groups).size),
        "residues": int(labels.size),
        "positive_residues": int(labels.sum()),
        "test_accessed": False,
        "base_checkpoint_selection_uses_full_validation": True,
        "interpretation_limit": "Diagnostic of stacker conditional gain; not an outer-CV or final test claim.",
        "methods": methods,
        "folds": fold_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.folds_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.folds_tsv.open("w") as handle:
        handle.write("chain_id\tcomponent_id\tfold\n")
        for chain_id, component_id, fold in zip(chain_ids, component_by_chain, fold_by_chain):
            handle.write(f"{chain_id}\t{component_id}\t{int(fold)}\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
