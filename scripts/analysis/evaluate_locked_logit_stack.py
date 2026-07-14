#!/usr/bin/env python3
"""Fit a locked two-logit stacker on validation and evaluate it once on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

from evaluate_grouped_oof_logit_stack import _best_thresholds, _f1, _ranking_metrics, _sigmoid


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    payload = {key: raw[key] for key in raw.files}
    required = {
        "labels",
        "primary_logit",
        "secondary_logit",
        "chain_index",
        "chain_ids",
        "component_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path}: missing arrays {missing}")
    n_residues = int(payload["labels"].size)
    for key in ("primary_logit", "secondary_logit", "chain_index"):
        if payload[key].size != n_residues:
            raise ValueError(f"{path}: {key} length does not match labels")
    raw_labels = payload["labels"]
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError(f"{path}: labels dtype must be integral")
    if set(np.unique(raw_labels).tolist()) - {0, 1}:
        raise ValueError(f"{path}: labels must be binary")
    labels = raw_labels.astype(np.int8, copy=False)
    if not np.isfinite(payload["primary_logit"]).all() or not np.isfinite(payload["secondary_logit"]).all():
        raise ValueError(f"{path}: non-finite logits")
    n_chains = int(payload["chain_ids"].size)
    if payload["component_ids"].size != n_chains:
        raise ValueError(f"{path}: chain_ids/component_ids metadata length mismatch")
    raw_chain_index = payload["chain_index"]
    if not np.issubdtype(raw_chain_index.dtype, np.integer):
        raise ValueError(f"{path}: chain_index dtype must be integral")
    if raw_chain_index.size == 0 or np.any(raw_chain_index < 0) or np.any(raw_chain_index >= n_chains):
        raise ValueError(f"{path}: chain_index is outside metadata bounds")
    chain_index = raw_chain_index.astype(np.int64, copy=False)
    if not np.array_equal(np.unique(chain_index), np.arange(n_chains)):
        raise ValueError(f"{path}: chain_index does not exactly cover all chain metadata rows")
    chain_ids = payload["chain_ids"].astype(str)
    if np.unique(chain_ids).size != n_chains:
        raise ValueError(f"{path}: duplicate chain_ids metadata")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(path: Path, expected_scope: str, npz_path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if summary.get("scope") != expected_scope:
        raise ValueError(f"{path}: expected scope={expected_scope!r}, got {summary.get('scope')!r}")
    if not summary.get("coverage_exact"):
        raise ValueError(f"{path}: export coverage is not exact")
    if Path(summary.get("out_npz", "")).resolve() != npz_path.resolve():
        raise ValueError(f"{path}: summary is not bound to {npz_path}")
    if summary.get("out_npz_sha256") != _sha256(npz_path):
        raise ValueError(f"{path}: NPZ SHA-256 mismatch")
    if bool(summary.get("test_accessed")) != (expected_scope == "test"):
        raise ValueError(f"{path}: inconsistent test_accessed flag")
    return summary


def _threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    tp = int(np.sum((y_true == 1) & predicted))
    fp = int(np.sum((y_true == 0) & predicted))
    fn = int(np.sum((y_true == 1) & ~predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": _f1(y_true, predicted),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
    }


def _method_report(
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    val_chain_index: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_chain_index: np.ndarray,
) -> dict[str, Any]:
    f1_threshold, mcc_threshold = _best_thresholds(val_labels, val_scores)
    return {
        "validation_fit_ranking": _ranking_metrics(val_labels, val_scores, val_chain_index),
        "test_ranking": _ranking_metrics(test_labels, test_scores, test_chain_index),
        "test_at_validation_f1_threshold": _threshold_metrics(test_labels, test_scores, f1_threshold),
        "test_at_validation_mcc_threshold": _threshold_metrics(test_labels, test_scores, mcc_threshold),
        "test_at_0_5": _threshold_metrics(test_labels, test_scores, 0.5),
        "test_at_0_6": _threshold_metrics(test_labels, test_scores, 0.6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-npz", required=True, type=Path)
    parser.add_argument("--validation-summary-json", required=True, type=Path)
    parser.add_argument("--test-npz", required=True, type=Path)
    parser.add_argument("--test-summary-json", required=True, type=Path)
    parser.add_argument("--oof-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validation = _load_payload(args.validation_npz)
    test = _load_payload(args.test_npz)
    validation_summary = _load_summary(args.validation_summary_json, "val", args.validation_npz)
    test_summary = _load_summary(args.test_summary_json, "test", args.test_npz)
    for payload, summary, name in (
        (validation, validation_summary, "validation"),
        (test, test_summary, "test"),
    ):
        if int(summary.get("chains", -1)) != int(payload["chain_ids"].size):
            raise ValueError(f"{name} summary/payload chain count mismatch")
        if int(summary.get("residues", -1)) != int(payload["labels"].size):
            raise ValueError(f"{name} summary/payload residue count mismatch")
    for key in ("primary_checkpoint", "primary_epoch", "secondary_checkpoint", "secondary_epoch"):
        if validation_summary.get(key) != test_summary.get(key):
            raise ValueError(f"Validation/test provenance mismatch for {key}")
    oof = json.loads(args.oof_json.read_text())
    if oof.get("scope") != "validation_only_grouped_crossfit" or oof.get("weighting") != "residue":
        raise ValueError("Locked model requires the reviewed residue-weighted validation OOF design")
    if Path(oof.get("input_npz", "")).resolve() != args.validation_npz.resolve():
        raise ValueError("OOF design is not bound to the supplied validation NPZ")

    val_labels = validation["labels"].astype(np.int8, copy=False)
    test_labels = test["labels"].astype(np.int8, copy=False)
    val_chain_index = validation["chain_index"].astype(np.int32, copy=False)
    test_chain_index = test["chain_index"].astype(np.int32, copy=False)
    val_components = set(validation["component_ids"].astype(str))
    test_components = set(test["component_ids"].astype(str))
    if val_components & test_components:
        raise RuntimeError("SI component leakage between validation and test exports")
    if set(validation["chain_ids"].astype(str)) & set(test["chain_ids"].astype(str)):
        raise RuntimeError("Chain ID overlap between validation and test exports")

    val_features = np.column_stack(
        [
            np.clip(validation["primary_logit"].astype(np.float64), -30.0, 30.0),
            np.clip(validation["secondary_logit"].astype(np.float64), -30.0, 30.0),
        ]
    )
    test_features = np.column_stack(
        [
            np.clip(test["primary_logit"].astype(np.float64), -30.0, 30.0),
            np.clip(test["secondary_logit"].astype(np.float64), -30.0, 30.0),
        ]
    )
    stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=20260714)
    stacker.fit(val_features, val_labels)
    val_stack_scores = stacker.predict_proba(val_features)[:, 1]
    test_stack_scores = stacker.predict_proba(test_features)[:, 1]
    val_primary_scores = _sigmoid(validation["primary_logit"])
    test_primary_scores = _sigmoid(test["primary_logit"])
    val_secondary_scores = _sigmoid(validation["secondary_logit"])
    test_secondary_scores = _sigmoid(test["secondary_logit"])

    methods = {
        "primary_esmc": _method_report(
            val_labels,
            val_primary_scores,
            val_chain_index,
            test_labels,
            test_primary_scores,
            test_chain_index,
        ),
        "secondary_esm2": _method_report(
            val_labels,
            val_secondary_scores,
            val_chain_index,
            test_labels,
            test_secondary_scores,
            test_chain_index,
        ),
        "locked_logit_stack": _method_report(
            val_labels,
            val_stack_scores,
            val_chain_index,
            test_labels,
            test_stack_scores,
            test_chain_index,
        ),
    }
    for metric in ("chain_ap_macro", "pr_auc", "auroc"):
        methods["locked_logit_stack"][f"test_delta_{metric}_vs_primary"] = (
            methods["locked_logit_stack"]["test_ranking"][metric]
            - methods["primary_esmc"]["test_ranking"][metric]
        )
    report = {
        "scope": "locked_validation_fit_single_test_evaluation",
        "selection_source": "validation only",
        "test_used_for_fit_or_threshold": False,
        "validation_npz": str(args.validation_npz),
        "test_npz": str(args.test_npz),
        "oof_design": str(args.oof_json),
        "primary_checkpoint": validation_summary["primary_checkpoint"],
        "primary_epoch": validation_summary["primary_epoch"],
        "secondary_checkpoint": validation_summary["secondary_checkpoint"],
        "secondary_epoch": validation_summary["secondary_epoch"],
        "stacker": {
            "type": "LogisticRegression",
            "C": 1.0,
            "weighting": "residue",
            "intercept": float(stacker.intercept_[0]),
            "primary_coefficient": float(stacker.coef_[0, 0]),
            "secondary_coefficient": float(stacker.coef_[0, 1]),
        },
        "validation": {
            "chains": int(validation["chain_ids"].size),
            "components": len(val_components),
            "residues": int(val_labels.size),
        },
        "test": {
            "chains": int(test["chain_ids"].size),
            "components": len(test_components),
            "residues": int(test_labels.size),
        },
        "methods": methods,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
