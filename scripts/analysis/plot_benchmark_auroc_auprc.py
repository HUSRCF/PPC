#!/usr/bin/env python3
"""Plot the latest full-coverage benchmark AUPRC and AUROC comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "Ours_ESM2_ESMC_M6_val_selected": "Ours (M6 ESM2+ESM-C)",
    "ScanNet_MSA_official_MMseqs_UniRef50_adapter_ESMFold": "ScanNet MSA",
    "PeSTo_official_i_v4_1_ESMFold": "PeSTo",
    "GPSite_ESMFold_prefilled": "GPSite",
    "ScanNet_noMSA_official_ESMFold": "ScanNet noMSA",
    "GraphPPIS_fast": "GraphPPIS",
    "PIPENN_classic_dnet_official_NetSurfP3_MMseqs_UniRef50_adapter": "PIPENN classic",
    "PIPENN-EMB_rnet": "PIPENN-EMB RNet",
    "PIPENN-EMB_unet": "PIPENN-EMB U-Net",
    "PIPENN-EMB_mean6": "PIPENN-EMB mean6",
    "PIPENN-EMB_cnn_rnn": "PIPENN-EMB CNN-RNN",
    "Gated-GPS_official_5fold_reconstructed_ESMFold": "Gated-GPS",
    "PIPENN-EMB_dnet": "PIPENN-EMB DNet",
    "PIPENN-EMB_rnn": "PIPENN-EMB RNN",
    "PIPENN-EMB_ann": "PIPENN-EMB ANN",
    "EquiPPIS_official_ESMFold_MMseqs_UniRef50_adapter": "EquiPPIS",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["AP"]))
    names = [DISPLAY_NAMES.get(row["method"], row["method"]) for row in rows]
    auprc = np.asarray([float(row["AUPRC"]) for row in rows])
    auroc = np.asarray([float(row["AUROC"]) for row in rows])
    colors = []
    for row in rows:
        if row["method"].startswith("Ours_"):
            colors.append("#D55E00")
        elif row["method"].startswith("ScanNet_MSA_"):
            colors.append("#0072B2")
        else:
            colors.append("#7A8793")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 6.3), sharey=True, gridspec_kw={"wspace": 0.10})
    panels = ((axes[0], auprc, "AUPRC", 0.28), (axes[1], auroc, "AUROC", 0.52))
    for axis, values, label, lower in panels:
        bars = axis.barh(y, values, color=colors, height=0.68, edgecolor="none")
        upper = min(1.0, max(values) + 0.055)
        axis.set_xlim(lower, upper)
        axis.set_xlabel(label)
        axis.xaxis.grid(True, color="#D8DDE3", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                value + 0.006,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=7.2,
                color="#20252B",
            )
    axes[0].set_yticks(y, labels=names)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.31, right=0.99, bottom=0.10, top=0.99)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(args.output_prefix.with_suffix(".pdf"))
    print(args.output_prefix.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
