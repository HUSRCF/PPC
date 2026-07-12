#!/usr/bin/env python3
"""Create the 2026-07-12 benchmark summary with validation-selected M6 as Ours."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-csv", required=True, type=Path)
    parser.add_argument("--previous-sources", required=True, type=Path)
    parser.add_argument("--m6-summary", required=True, type=Path)
    parser.add_argument("--m6-ratio-summary", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    with args.previous_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if not row["method"].startswith("Ours_")]
    m6 = json.loads(args.m6_summary.read_text())
    ratio = json.loads(args.m6_ratio_summary.read_text())
    test = m6["test"]
    ours = {
        "method": "Ours_ESM2_ESMC_M6_val_selected",
        "AUROC": test["auroc"],
        "AUPRC": test["auprc"],
        "AP": test["average_precision"],
        "F1@0.5": test["threshold_0_5"]["f1"],
        "MCC@0.5": test["threshold_0_5"]["mcc"],
        "best_F1": test["test_oracle_diagnostics"]["best_F1"],
        "best_MCC": test["test_oracle_diagnostics"]["best_MCC"],
        "n_eval_chains": 2392,
        "coverage": "2392/2392",
        "source": "benchmark/reports/summary/esmc_matrix_20260712/m6/m6_summary.json",
    }
    rows.append({field: str(ours[field]) for field in fields})
    rows.sort(key=lambda row: float(row["AP"]), reverse=True)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    tsv_path = args.output_prefix.with_suffix(".tsv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    previous_sources = json.loads(args.previous_sources.read_text())
    sources = deepcopy(previous_sources)
    sources["generated"] = "2026-07-12"
    sources["previous_summary"] = str(args.previous_csv)
    sources["row_count"] = len(rows)
    sources["previous_ours_baseline"] = sources.get("ours")
    sources["ours"] = {
        "variant": "M6 validation-selected M0/M2 logit ensemble",
        "selection_rule": m6["selection_policy"],
        "selected_alpha_m0": m6["selected_alpha_m0"],
        "selected_alpha_m2": m6["selected_alpha_m2"],
        "selected_validation_threshold": m6["selected_validation_threshold"],
        "metrics": str(args.m6_summary),
        "test_predictions": str(args.m6_summary.parent / "m6_test_predictions.npz"),
        "coverage": m6["coverage"],
        "source_scores": m6["sources"],
        "common_evaluator_metrics": {
            "AUROC": test["auroc"],
            "AUPRC": test["auprc"],
            "AP": test["average_precision"],
            "F1@0.5": test["threshold_0_5"]["f1"],
            "MCC@0.5": test["threshold_0_5"]["mcc"],
            "F1@0.6": test["threshold_0_6"]["f1"],
            "MCC@0.6": test["threshold_0_6"]["mcc"],
            "best_F1": test["test_oracle_diagnostics"]["best_F1"],
            "best_F1_threshold": test["test_oracle_diagnostics"]["best_F1_threshold"],
            "best_MCC": test["test_oracle_diagnostics"]["best_MCC"],
            "best_MCC_threshold": test["test_oracle_diagnostics"]["best_MCC_threshold"],
            "L/20_ACC_macro": test["L/20_ACC_macro"],
            "L/10_ACC_macro": test["L/10_ACC_macro"],
            "L/5_ACC_macro": test["L/5_ACC_macro"],
        },
        "test_at_validation_selected_threshold": test["selected_threshold"],
        "checkpoint_selection": "M0 and M2 epochs selected by validation F1; M2 chosen as the ESM-C member by validation F1",
        "jobs": {"m0_export": 9967169, "m2_export": 9967171, "m6_merge": 9967203},
        "prediction_qc": "2392/2392 test chains and 1036256/1036256 residues; strict checkpoint state loads; label arrays identical",
    }
    sources["esmc_matrix_20260712"] = {
        "report": str(args.m6_summary.parents[1] / "ESMC_MATRIX_20260712.md"),
        "formal_table": str(args.m6_summary.parents[1] / "formal_m0_m5.tsv"),
        "checkpoint_archive": "/media/990Pro/ProtBind/PPC/benchmark/models/ours/esmc_matrix_20260712",
        "status": "M0-M5 formal A800 runs and M6 validation-only ensemble complete",
    }
    sources.setdefault("effect_site_ratio", {})["m6_raw"] = ratio["m6"]
    sources["effect_site_ratio"]["m6_paired_component_bootstrap"] = ratio["paired_component_bootstrap"]
    sources_path = args.output_prefix.with_suffix(".sources.json")
    sources_path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")

    ratio_rows = [(ratio["method"], ratio["m6"]), *ratio["baselines"].items()]
    ratio_rows.sort(key=lambda item: float(item[1]["ratio_MAE_chain_macro"]))
    table_lines = []
    for row in rows:
        table_lines.append(
            f"| {row['method']} | {float(row['AUPRC']):.4f} | {float(row['AUROC']):.4f} | "
            f"{float(row['AP']):.4f} | {float(row['F1@0.5']):.4f} | {float(row['MCC@0.5']):.4f} | "
            f"{float(row['best_F1']):.4f} | {float(row['best_MCC']):.4f} | {row['coverage']} |\n"
        )
    ratio_lines = []
    for method, metrics in ratio_rows:
        ratio_lines.append(
            f"| {method} | {metrics['ratio_MAE_chain_macro']:.4f} | {metrics['ratio_RMSE_chain_macro']:.4f} | "
            f"{metrics['ratio_bias_chain_macro']:+.4f} | {metrics['ratio_Pearson']:.4f} | {metrics['ratio_Spearman']:.4f} |\n"
        )
    scan_boot = ratio["paired_component_bootstrap"]["ScanNet_MSA_official_MMseqs_UniRef50_adapter_ESMFold"]
    pesto_boot = ratio["paired_component_bootstrap"]["PeSTo_official_i_v4_1_ESMFold"]
    markdown = f"""# Chain-Filtered Global SI30 Benchmark Summary (2026-07-12)

Split: `splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/chain_filtered_training`, test set 2,392 chains / 1,036,256 residues.

Interpretation status: **development-split point estimates, not a sealed-holdout result**. Checkpoint, representation-member, blend weight, and operating threshold for the new Ours row were selected from validation only. Test metrics were nevertheless visible during the broader development sweep.

![AUROC and AUPRC on the chain-filtered global-SI30 test set](figures/chain_filtered_global_si30_auroc_auprc_20260712.png)

| Method | AUPRC | AUROC | AP | F1@0.5 | MCC@0.5 | best F1 | best MCC | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{''.join(table_lines)}

## Ours M6

M6 is `0.45 * logit(M0 ESM2-MLC) + 0.55 * logit(M2 ESM-C MLC)`. Alpha 0.55 and threshold 0.52 were selected exclusively on 3,796 validation chains. Frozen-threshold test F1/MCC are **{test['selected_threshold']['f1']:.4f}/{test['selected_threshold']['mcc']:.4f}**. F1/MCC at the paper-style fixed thresholds are `{test['threshold_0_5']['f1']:.4f}/{test['threshold_0_5']['mcc']:.4f}` at 0.5 and `{test['threshold_0_6']['f1']:.4f}/{test['threshold_0_6']['mcc']:.4f}` at 0.6. Best-F1 and best-MCC columns in the primary table remain test-oracle diagnostics for all methods.

M6 improves over the previous Ours row by AP `{test['average_precision'] - float(previous_sources['ours']['common_evaluator_metrics']['AP']):+.4f}` and AUROC `{test['auroc'] - float(previous_sources['ours']['common_evaluator_metrics']['AUROC']):+.4f}` on the identical test universe. Full M0-M6 results, loader profiles, checkpoint hashes, and control audit are in `esmc_matrix_20260712/`.

## Single-Chain Effect-Site Ratio

Predicted ratio is the raw mean residue probability; no test calibration is fitted.

| Method | Ratio MAE | Ratio RMSE | Bias | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
{''.join(ratio_lines)}

M6 has the lowest MAE point estimate, but the paired 43-component, 10,000-replicate bootstrap does not establish significance: M6-minus-ScanNet MSA is `{scan_boot['m6_minus_baseline_MAE']:+.4f}` with 95% CI `[{scan_boot['m6_minus_baseline_MAE_ci95'][0]:+.4f}, {scan_boot['m6_minus_baseline_MAE_ci95'][1]:+.4f}]`; M6-minus-PeSTo is `{pesto_boot['m6_minus_baseline_MAE']:+.4f}` with `[{pesto_boot['m6_minus_baseline_MAE_ci95'][0]:+.4f}, {pesto_boot['m6_minus_baseline_MAE_ci95'][1]:+.4f}]`.

## Footnotes

- All structure-based rows use exact-sequence, chain-only ESMFold structures. The canonical archive is `/data/pdbtm/ESMFold` on star: 7,306 unique structures, final 20,240/20,240 chains and 6,667/6,667 exact sequences mapped, zero sequence mismatch/conflict.
- M6 uses complete 2,392-chain coverage and strict state/label alignment. M0/M2 best checkpoints are validation-selected epoch 17.
- ScanNet MSA, EquiPPIS, and PIPENN classic use the documented MMseqs2/UniRef50 adapters rather than their original heavy profile routes.
- PeSTo is an overlap-contaminated upper-bound reference; exact-PDB overlap removal was previously audited separately.
- Gated-GPS uses official five-fold weights with a reconstructed feature adapter, not the unpublished original feature stack.
- PIPENN-EMB `mean6` is a soft mean over six official base outputs, not the unavailable supervised stacking ANN.
- Seq-InSite official supports only `len(sequence) < 1024` and remains in the separate 2,343-chain matched-subset report.
- DELPHI and DeepPPISP remain absent as official full-coverage rows because the complete official DELPHI feature stack and an official DeepPPISP checkpoint are unavailable.
- The independent SI audit found zero recalled cross-split `SI>=0.30` violations (maximum 0.2826855), but is not exhaustive dynamic programming over all approximately 9.5 million pairs.

The full method-specific provenance and caveats from the 2026-07-10 report remain incorporated in the companion `.sources.json`; this refresh changes the Ours row and adds the completed ESM-C/M6 artifacts.
"""
    args.output_prefix.with_suffix(".md").write_text(markdown)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(args.output_prefix.with_suffix('.md'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
