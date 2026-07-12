#!/usr/bin/env python3
"""Summarize formal M0-M5 ESM-C runs using validation-only model selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def threshold_metrics(metrics: dict, threshold: float) -> dict:
    for row in metrics.get("threshold_sweep", []):
        if abs(float(row["threshold"]) - threshold) < 1e-8:
            return row
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    run_dirs = sorted(args.runs_root.glob("contact_site_cfsi30_esmc_matrix_*_20260711_hpc2"))
    if len(run_dirs) != 6:
        raise ValueError(f"Expected six formal runs, found {len(run_dirs)}")

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        events = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        epochs = [event for event in events if event.get("event") == "epoch"]
        if len(epochs) != 30:
            raise ValueError(f"{run_dir.name}: expected 30 epochs, found {len(epochs)}")
        best = max(epochs, key=lambda event: (float(event["val"]["f1_best_threshold"]), -int(event["epoch"])))
        val = best["val"]
        test = best["test"]
        frozen = best["test_at_val_threshold"]
        at_05 = threshold_metrics(test, 0.5)
        at_06 = threshold_metrics(test, 0.6)
        config = json.loads((run_dir / "config.json").read_text())
        metadata = config.get("yaml_config", {}).get("metadata", {})
        variant = str(metadata.get("matrix_variant") or run_dir.name.split("_matrix_", 1)[1].split("_seed42", 1)[0])
        rows.append(
            {
                "variant": variant,
                "representation": metadata.get("representation", ""),
                "best_epoch_by_val": int(best["epoch"]),
                "val_f1_best": float(val["f1_best_threshold"]),
                "val_selected_threshold": float(val["best_threshold"]),
                "val_auroc": float(val["auroc"]),
                "val_auprc": float(val["pr_auc"]),
                "test_auroc": float(test["auroc"]),
                "test_auprc": float(test["pr_auc"]),
                "test_f1_at_val_threshold": float(frozen["f1"]),
                "test_mcc_at_val_threshold": float(frozen["mcc"]),
                "test_precision_at_val_threshold": float(frozen["precision"]),
                "test_recall_at_val_threshold": float(frozen["recall"]),
                "test_f1_at_0_5": float(at_05.get("f1", test.get("f1_at_0_5", 0.0))),
                "test_mcc_at_0_5": float(at_05.get("mcc", test.get("mcc_at_0_5", 0.0))),
                "test_f1_at_0_6": float(at_06.get("f1", 0.0)),
                "test_mcc_at_0_6": float(at_06.get("mcc", 0.0)),
                "test_oracle_f1": float(test["f1_best_threshold"]),
                "test_oracle_threshold": float(test["best_threshold"]),
                "test_top_l20_precision_macro": float(test.get("top_5pct_precision_macro", 0.0)),
                "test_top_l10_precision_macro": float(test.get("top_10pct_precision_macro", 0.0)),
                "train_residues_per_second": float(best["train"].get("residues_per_second", 0.0)),
                "train_data_wait_fraction": float(best["train"].get("data_wait_fraction", 0.0)),
                "run_dir": str(run_dir),
            }
        )

    rows.sort(key=lambda row: str(row["variant"]))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_prefix.with_suffix(".tsv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    header = (
        "| Variant | Best epoch | Val F1@thr | Test F1@val-thr | Test MCC@val-thr | "
        "Test AUPRC | Test AUROC | Test F1@0.5 | Test F1@0.6 | L/20 P | L/10 P |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['best_epoch_by_val']} | "
            f"{row['val_f1_best']:.4f}@{row['val_selected_threshold']:.2f} | "
            f"{row['test_f1_at_val_threshold']:.4f} | {row['test_mcc_at_val_threshold']:.4f} | "
            f"{row['test_auprc']:.4f} | {row['test_auroc']:.4f} | "
            f"{row['test_f1_at_0_5']:.4f} | {row['test_f1_at_0_6']:.4f} | "
            f"{row['test_top_l20_precision_macro']:.4f} | {row['test_top_l10_precision_macro']:.4f} |\n"
        )
    lines.append(
        "\nSelection rule: epoch and threshold are selected exclusively on validation. "
        "`test_oracle_f1` is retained only in TSV/JSON as a diagnostic and is not used for ranking.\n"
    )
    md_path.write_text("".join(lines))
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
