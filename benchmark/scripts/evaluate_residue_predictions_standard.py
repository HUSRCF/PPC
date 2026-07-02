#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-per-chain")
    return p.parse_args()


def safe_float_metric(fn, y, score):
    if len(np.unique(y)) < 2:
        return None
    return float(fn(y, score))


def best_thresholds(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_f1_idx = int(np.nanargmax(f1))
    if best_f1_idx == len(thresholds):
        best_f1_thr = 1.0
    else:
        best_f1_thr = float(thresholds[best_f1_idx])

    best_mcc = -2.0
    best_mcc_thr = 0.5
    for thr in np.linspace(0.0, 1.0, 1001):
        pred = score >= thr
        mcc = matthews_corrcoef(y, pred)
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_mcc_thr = float(thr)

    return {
        "best_F1": float(f1[best_f1_idx]),
        "best_F1_threshold": best_f1_thr,
        "best_MCC": best_mcc,
        "best_MCC_threshold": best_mcc_thr,
    }


def top_l_metrics(df):
    rows = []
    for seq_id, group in df.groupby("seq_id", sort=False):
        group = group.sort_values("score", ascending=False)
        length = int(group.shape[0])
        labels = group["label"].to_numpy(dtype=int)
        row = {"seq_id": seq_id, "length": length, "positives": int(labels.sum())}
        for denom in (20, 10, 5):
            k = max(1, length // denom)
            hits = int(labels[:k].sum())
            row[f"L/{denom}_k"] = k
            row[f"L/{denom}_hits"] = hits
            row[f"L/{denom}_ACC"] = float(hits / k)
        rows.append(row)

    per_chain = pd.DataFrame(rows)
    summary = {}
    for denom in (20, 10, 5):
        hits = int(per_chain[f"L/{denom}_hits"].sum())
        total = int(per_chain[f"L/{denom}_k"].sum())
        summary[f"L/{denom}_ACC_micro"] = float(hits / total) if total else None
        summary[f"L/{denom}_ACC_macro"] = float(per_chain[f"L/{denom}_ACC"].mean())
        summary[f"L/{denom}_topk_total"] = total
        summary[f"L/{denom}_hits_total"] = hits
    return summary, per_chain


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    label_path = Path(args.labels)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path, sep="\t")
    labels = pd.read_csv(label_path, sep="\t")
    pred["position"] = pred["position"].astype(int)
    labels["position"] = labels["position"].astype(int)
    pred["score"] = pred["score"].astype(float)
    labels["label"] = labels["label"].astype(int)

    keys = ["seq_id", "position"]
    merged = labels.merge(
        pred[["seq_id", "position", "residue", "score", "method"]],
        on=keys,
        how="inner",
        suffixes=("_label", "_pred"),
    )
    residue_mismatch = merged[merged["residue_label"] != merged["residue_pred"]]
    if not residue_mismatch.empty:
        mismatch_path = out_json.with_suffix(".residue_mismatch.tsv")
        residue_mismatch.head(1000).to_csv(mismatch_path, sep="\t", index=False)
        raise SystemExit(f"Residue mismatch detected; wrote {mismatch_path}")

    y = merged["label"].to_numpy(dtype=int)
    score = merged["score"].to_numpy(dtype=float)
    pred05 = score >= 0.5
    precision, recall, _ = precision_recall_curve(y, score)

    metrics = {
        "method": args.method,
        "predictions": str(pred_path),
        "labels": str(label_path),
        "n_label_rows": int(labels.shape[0]),
        "n_prediction_rows": int(pred.shape[0]),
        "n_eval_rows": int(merged.shape[0]),
        "n_label_chains": int(labels["seq_id"].nunique()),
        "n_prediction_chains": int(pred["seq_id"].nunique()),
        "n_eval_chains": int(merged["seq_id"].nunique()),
        "label_positive_rate_eval": float(y.mean()) if len(y) else None,
        "AUROC": safe_float_metric(roc_auc_score, y, score),
        "AUPRC": float(auc(recall[::-1], precision[::-1])),
        "AP": float(average_precision_score(y, score)),
        "F1@0.5": float(f1_score(y, pred05, zero_division=0)),
        "MCC@0.5": float(matthews_corrcoef(y, pred05)),
    }
    metrics.update(best_thresholds(y, score))
    top_summary, per_chain = top_l_metrics(merged)
    metrics.update(top_summary)

    if args.out_per_chain:
        per_chain_path = Path(args.out_per_chain)
        per_chain_path.parent.mkdir(parents=True, exist_ok=True)
        per_chain.to_csv(per_chain_path, index=False)
        metrics["per_chain_topl_csv"] = str(per_chain_path)

    missing_chains = sorted(set(labels["seq_id"]) - set(pred["seq_id"]))
    metrics["n_missing_prediction_chains"] = len(missing_chains)
    metrics["missing_prediction_chains_first20"] = missing_chains[:20]

    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
