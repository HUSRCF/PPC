#!/usr/bin/env python3
"""Evaluate M6 single-chain effect-site ratios with paired component bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


BASELINES = (
    "ScanNet_MSA_official_MMseqs_UniRef50_adapter_ESMFold",
    "PeSTo_official_i_v4_1_ESMFold",
)


def load_m6(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=False)
    seq_ids = [str(value) for value in payload["seq_ids"]]
    lengths = payload["lengths"].astype(np.int64)
    labels = payload["labels"].astype(np.uint8)
    scores = payload["scores"].astype(np.float64)
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    if int(offsets[-1]) != labels.size or labels.size != scores.size:
        raise ValueError("M6 prediction arrays have inconsistent lengths")
    rows = []
    for index, seq_id in enumerate(seq_ids):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        chain_labels = labels[start:stop]
        chain_scores = scores[start:stop]
        true_ratio = float(chain_labels.mean())
        predicted_ratio = float(chain_scores.mean())
        rows.append(
            {
                "method": "Ours_ESM2_ESMC_M6_val_selected",
                "seq_id": seq_id,
                "true_ratio": true_ratio,
                "n_residues": int(stop - start),
                "n_positive": int(chain_labels.sum()),
                "predicted_ratio": predicted_ratio,
                "ratio_error": predicted_ratio - true_ratio,
                "absolute_error": abs(predicted_ratio - true_ratio),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    true = frame["true_ratio"].to_numpy(float)
    predicted = frame["predicted_ratio"].to_numpy(float)
    error = predicted - true
    return {
        "n_chains": int(frame.shape[0]),
        "ratio_MAE_chain_macro": float(np.mean(np.abs(error))),
        "ratio_RMSE_chain_macro": float(np.sqrt(np.mean(np.square(error)))),
        "ratio_bias_chain_macro": float(np.mean(error)),
        "ratio_Pearson": float(pearsonr(true, predicted).statistic),
        "ratio_Spearman": float(spearmanr(true, predicted).statistic),
        "mean_predicted_ratio_chain_macro": float(np.mean(predicted)),
        "mean_true_ratio_chain_macro": float(np.mean(true)),
    }


def component_bootstrap(
    merged: pd.DataFrame,
    baseline: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    m6_error = merged["m6_absolute_error"].to_numpy(float)
    baseline_error = merged[f"{baseline}_absolute_error"].to_numpy(float)
    difference = m6_error - baseline_error
    component_ids = sorted(merged["component_id"].unique())
    grouped = merged.assign(difference=difference).groupby("component_id", sort=True)["difference"].agg(["sum", "count"])
    sums = grouped.loc[component_ids, "sum"].to_numpy(float)
    counts = grouped.loc[component_ids, "count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(component_ids), size=(replicates, len(component_ids)))
    deltas = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "baseline": baseline,
        "unit": "paired component_id",
        "n_components": len(component_ids),
        "replicates": replicates,
        "seed": seed,
        "m6_minus_baseline_MAE": float(difference.mean()),
        "m6_minus_baseline_MAE_ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "bootstrap_mean": float(deltas.mean()),
        "excludes_zero": bool(np.quantile(deltas, 0.975) < 0 or np.quantile(deltas, 0.025) > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m6-predictions", required=True, type=Path)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--baseline-per-chain", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m6 = load_m6(args.m6_predictions)
    manifest = pd.read_csv(args.chain_manifest, usecols=["seq_id", "component_id"])
    if manifest["seq_id"].duplicated().any():
        raise ValueError("Chain manifest has duplicate seq_id values")
    m6 = m6.merge(manifest, on="seq_id", how="left", validate="one_to_one")
    if m6["component_id"].isna().any():
        raise ValueError("M6 chain is missing component_id")

    baseline_all = pd.read_csv(args.baseline_per_chain)
    baseline_all["absolute_error"] = baseline_all["ratio_error"].abs()
    merged = m6[["seq_id", "component_id", "true_ratio", "absolute_error"]].rename(
        columns={"absolute_error": "m6_absolute_error"}
    )
    baseline_summaries: dict[str, dict[str, float | int]] = {}
    for baseline in BASELINES:
        frame = baseline_all[baseline_all["method"] == baseline].copy()
        if frame.shape[0] != m6.shape[0]:
            raise ValueError(f"{baseline}: expected {m6.shape[0]} chains, found {frame.shape[0]}")
        baseline_summaries[baseline] = metric_summary(frame)
        values = frame[["seq_id", "true_ratio", "absolute_error"]].rename(
            columns={"true_ratio": f"{baseline}_true_ratio", "absolute_error": f"{baseline}_absolute_error"}
        )
        merged = merged.merge(values, on="seq_id", how="inner", validate="one_to_one")
        if not np.allclose(merged["true_ratio"], merged[f"{baseline}_true_ratio"], atol=1e-12, rtol=0):
            raise ValueError(f"{baseline}: true ratios differ from M6 labels")

    if merged.shape[0] != m6.shape[0]:
        raise ValueError("Paired chain coverage is incomplete")
    bootstrap = {
        baseline: component_bootstrap(merged, baseline, args.replicates, args.seed)
        for baseline in BASELINES
    }
    summary = {
        "method": "Ours_ESM2_ESMC_M6_val_selected",
        "definition": "true ratio = positive residues / chain length; predicted ratio = mean residue probability",
        "m6": metric_summary(m6),
        "baselines": baseline_summaries,
        "paired_component_bootstrap": bootstrap,
        "coverage_qc": {
            "chains": int(m6.shape[0]),
            "components": int(m6["component_id"].nunique()),
            "missing_components": int(m6["component_id"].isna().sum()),
            "paired_rows": int(merged.shape[0]),
            "true_ratio_mismatches": 0,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    m6.to_csv(args.output_dir / "m6_effect_site_ratio_per_chain.csv", index=False)
    (args.output_dir / "m6_effect_site_ratio_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = []
    for method, metrics in [(summary["method"], summary["m6"]), *summary["baselines"].items()]:
        rows.append({"method": method, **metrics})
    pd.DataFrame(rows).to_csv(args.output_dir / "m6_effect_site_ratio_comparison.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
