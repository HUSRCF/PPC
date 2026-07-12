#!/usr/bin/env python3
"""Render the locked English Old-ESM2/M1-M6 representation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = (
    "m0_esm2_mlc",
    "m1_esmc_final",
    "m2_esmc_mlc_concat",
    "m3_esmc_mlc_scalar_mix",
    "m4_esm2_esmc_projected_concat",
    "m5_esm2_esmc_gated_residual",
)

DISPLAY = {
    "m0_esm2_mlc": ("Old ESM2 (M0)", "ESM2 layers 11/22/33 concatenated (3,840D)"),
    "m1_esmc_final": ("ESM-C Final (M1)", "ESM-C final block 36 (1,152D)"),
    "m2_esmc_mlc_concat": ("ESM-C MLC (M2)", "ESM-C blocks 12/24/36 concatenated (3,456D)"),
    "m3_esmc_mlc_scalar_mix": ("ESM-C Scalar Mix (M3)", "Learned scalar mix of ESM-C blocks 12/24/36"),
    "m4_esm2_esmc_projected_concat": (
        "Projected Fusion (M4)",
        "Parameter-matched projected concatenation of ESM2 and ESM-C",
    ),
    "m5_esm2_esmc_gated_residual": (
        "Gated Fusion (M5)",
        "Full ESM2/ESM-C projections with gated residual fusion",
    ),
    "m6": ("Logit Ensemble (M6)", "Validation-selected logit blend of M0 and M2"),
}

HEATMAP_COLUMNS = (
    ("test_f1_frozen", "Frozen F1"),
    ("test_mcc_frozen", "Frozen MCC"),
    ("test_auprc", "AUPRC"),
    ("chain_macro_ap", "Macro AP"),
    ("test_auroc", "AUROC"),
    ("f1_at_0_5", "F1 @ 0.5"),
    ("f1_at_0_6", "F1 @ 0.6"),
    ("oracle_f1", "Oracle F1*"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(source_dir: Path) -> tuple[list[dict[str, object]], dict[str, Path]]:
    formal_path = source_dir / "formal_m0_m5.tsv"
    macro_path = source_dir / "macro_ap" / "chain_macro_ap_summary.tsv"
    m6_path = source_dir / "m6" / "m6_summary.json"
    formal = pd.read_csv(formal_path, sep="\t").set_index("variant")
    macro = pd.read_csv(macro_path, sep="\t").set_index("model")["chain_ap_macro"].to_dict()

    rows: list[dict[str, object]] = []
    for index, variant in enumerate(VARIANT_ORDER):
        item = formal.loc[variant]
        model_id = f"M{index}"
        display_name, representation = DISPLAY[variant]
        rows.append(
            {
                "variant": model_id,
                "display_name": display_name,
                "representation": representation,
                "selected_epoch": str(int(item["best_epoch_by_val"])),
                "validation_f1": float(item["val_f1_best"]),
                "validation_threshold": float(item["val_selected_threshold"]),
                "test_f1_frozen": float(item["test_f1_at_val_threshold"]),
                "test_mcc_frozen": float(item["test_mcc_at_val_threshold"]),
                "test_auprc": float(item["test_auprc"]),
                "chain_macro_ap": float(macro[model_id]),
                "test_auroc": float(item["test_auroc"]),
                "f1_at_0_5": float(item["test_f1_at_0_5"]),
                "f1_at_0_6": float(item["test_f1_at_0_6"]),
                "oracle_f1": float(item["test_oracle_f1"]),
                "oracle_threshold": float(item["test_oracle_threshold"]),
            }
        )

    m6 = json.loads(m6_path.read_text())
    validation = m6["validation"]["selected_threshold"]
    test = m6["test"]
    display_name, representation = DISPLAY["m6"]
    rows.append(
        {
            "variant": "M6",
            "display_name": display_name,
            "representation": representation,
            "selected_epoch": "Ensemble",
            "validation_f1": float(validation["f1"]),
            "validation_threshold": float(m6["selected_validation_threshold"]),
            "test_f1_frozen": float(test["selected_threshold"]["f1"]),
            "test_mcc_frozen": float(test["selected_threshold"]["mcc"]),
            "test_auprc": float(test["auprc"]),
            "chain_macro_ap": float(macro["M6"]),
            "test_auroc": float(test["auroc"]),
            "f1_at_0_5": float(test["threshold_0_5"]["f1"]),
            "f1_at_0_6": float(test["threshold_0_6"]["f1"]),
            "oracle_f1": float(test["test_oracle_diagnostics"]["best_F1"]),
            "oracle_threshold": float(test["test_oracle_diagnostics"]["best_F1_threshold"]),
        }
    )
    return rows, {"formal_m0_m5": formal_path, "chain_macro_ap": macro_path, "m6": m6_path}


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Old ESM2 and M1-M6 Representation Matrix",
        "",
        "| Variant | Representation | Selection | Frozen F1 | Frozen MCC | AUPRC | Macro AP | AUROC | F1@0.5 | F1@0.6 | Oracle F1* |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        selection = f"{row['validation_f1']:.4f}@{row['validation_threshold']:.2f}"
        oracle = f"{row['oracle_f1']:.4f}@{row['oracle_threshold']:.4f}"
        lines.append(
            f"| {row['display_name']} | {row['representation']} | {selection} | "
            f"{row['test_f1_frozen']:.4f} | {row['test_mcc_frozen']:.4f} | "
            f"{row['test_auprc']:.4f} | {row['chain_macro_ap']:.4f} | "
            f"{row['test_auroc']:.4f} | {row['f1_at_0_5']:.4f} | "
            f"{row['f1_at_0_6']:.4f} | {oracle} |"
        )
    lines.extend(
        [
            "",
            "Selection is validation F1 and its validation threshold. Frozen F1/MCC use that threshold on test.",
            "`*` Oracle F1 is a test-only diagnostic and was not used for model or threshold selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_latex(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Old ESM2 and ESM-C representation matrix on the global-SI30 chain-filtered test set. Frozen F1 and MCC use the validation-selected threshold.}",
        r"\label{tab:esm-representation-matrix}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Variant & Frozen F1 & Frozen MCC & AUPRC & Macro AP & AUROC & F1@0.5 & F1@0.6 & Oracle F1$^{*}$ \\ ",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display_name']} & {row['test_f1_frozen']:.4f} & {row['test_mcc_frozen']:.4f} & "
            f"{row['test_auprc']:.4f} & {row['chain_macro_ap']:.4f} & {row['test_auroc']:.4f} & "
            f"{row['f1_at_0_5']:.4f} & {row['f1_at_0_6']:.4f} & {row['oracle_f1']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{0.98\textwidth}{\footnotesize $^{*}$Test-oracle diagnostic only; not used for selection.}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def draw_heatmap(rows: list[dict[str, object]], output_dir: Path) -> None:
    values = np.asarray([[float(row[key]) for key, _ in HEATMAP_COLUMNS] for row in rows])
    minima = values.min(axis=0)
    ranges = values.max(axis=0) - minima
    normalized = np.divide(values - minima, ranges, out=np.full_like(values, 0.5), where=ranges > 0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    image = ax.imshow(normalized, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(HEATMAP_COLUMNS)))
    ax.set_xticklabels([label for _, label in HEATMAP_COLUMNS], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([str(row["display_name"]) for row in rows])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(HEATMAP_COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    maxima = values.max(axis=0)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            color = "white" if normalized[row_index, col_index] < 0.48 else "black"
            weight = "bold" if np.isclose(values[row_index, col_index], maxima[col_index]) else "normal"
            ax.text(
                col_index,
                row_index,
                f"{values[row_index, col_index]:.4f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                fontweight=weight,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.02)
    colorbar.set_label("Within-column normalized score", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.text(
        0.01,
        0.012,
        "Cells show raw metrics; color is normalized independently within each column. *Oracle F1 is diagnostic only.",
        ha="left",
        va="bottom",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.235, right=0.94, top=0.98, bottom=0.25)
    fig.savefig(output_dir / "esm2_m1_m6_matrix_english.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "esm2_m1_m6_matrix_english.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_dir = project_root / "benchmark" / "reports" / "summary" / "esmc_matrix_20260712"
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, sources = load_rows(source_dir)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "esm2_m1_m6_matrix_english.tsv", sep="\t", index=False)
    write_markdown(rows, output_dir / "esm2_m1_m6_matrix_english.md")
    write_latex(rows, output_dir / "esm2_m1_m6_matrix_english.tex")
    draw_heatmap(rows, output_dir)
    provenance = {
        "scope": "Locked historical M0-M6 matrix; excludes M7 and seed-averaged M6",
        "rows": rows,
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
    }
    (output_dir / "esm2_m1_m6_matrix_english.sources.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(output_dir / "esm2_m1_m6_matrix_english.pdf")
    print(output_dir / "esm2_m1_m6_matrix_english.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
