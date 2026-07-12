#!/usr/bin/env python3
"""Generate the locked validation-only M7 screening matrix."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


BASE_CONFIGS = {
    "m2": "configs/experiments/20260711_esmc_matrix_hpc2/train_m2_esmc_mlc_concat_seed42.yaml",
    "m5": "configs/experiments/20260711_esmc_matrix_hpc2/train_m5_esm2_esmc_gated_residual_seed42.yaml",
}


VARIANTS = {
    "m7a_m2_macro_select": {
        "base": "m2",
        "patch": {"training": {"loss_reduction": "residue_mean"}},
        "description": "M2 clean control with chain-macro AP checkpoint selection",
    },
    "m7b_m2_chainmean": {
        "base": "m2",
        "patch": {"training": {"loss_reduction": "chain_mean"}},
        "description": "M2 with per-chain mean cross-entropy",
    },
    "m7c_m2_chainmean_tcn2": {
        "base": "m2",
        "patch": {
            "training": {"loss_reduction": "chain_mean"},
            "model": {
                "use_tcn_context": True,
                "tcn_layers": 2,
                "tcn_kernel_size": 7,
                "tcn_dilations": [1, 2],
                "tcn_block_type": "gated_depthwise",
            },
        },
        "description": "M2 chain-balanced loss plus two gated depthwise TCN context blocks",
    },
    "m7d_m5_chainmean": {
        "base": "m5",
        "patch": {"training": {"loss_reduction": "chain_mean"}},
        "description": "M5 ESM2/ESM-C gated fusion with per-chain mean cross-entropy",
    },
    "m7e_m2_chainmean_sparse1": {
        "base": "m2",
        "patch": {
            "training": {"loss_reduction": "chain_mean"},
            "model": {
                "contact_graph_layers": 0,
                "use_sparse_graph_transformer": True,
                "sparse_graph_layers": 1,
                "sparse_graph_heads": 8,
                "sparse_graph_seq_neighbor_k": 4,
                "sparse_graph_use_contact_edges": True,
                "sparse_graph_use_seq_edges": True,
                "sparse_graph_use_global_node": True,
                "sparse_graph_use_chain_edge_type": True,
                "sparse_graph_edge_hidden": 64,
                "sparse_graph_adaptive_residual_gate": True,
            },
        },
        "description": "Conditional sparse graph-transformer replacement for the simple contact block",
    },
}


def merge(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-suffix", default="20260712_hpc2")
    parser.add_argument("--variants", default="", help="Comma-separated names; empty means all")
    args = parser.parse_args()

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    if not names:
        names = list(VARIANTS)
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown M7 variants: {unknown}; available={sorted(VARIANTS)}")

    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache = {
        name: yaml.safe_load((project_root / path).read_text())
        for name, path in BASE_CONFIGS.items()
    }

    for name in names:
        spec = VARIANTS[name]
        config = copy.deepcopy(base_cache[spec["base"]])
        merge(config, spec["patch"])
        training = config["training"]
        training.update(
            {
                "output_dir": f"runs/contact_site_cfsi30_m7screen_{name}_seed{args.seed}_e30_{args.run_suffix}",
                "seed": args.seed,
                "selection_metric": "chain_ap_macro",
                "save_metric_checkpoints": "f1_best_threshold,pr_auc,chain_ap_macro",
                "eval_test_each_epoch": False,
            }
        )
        metadata = config.setdefault("metadata", {})
        metadata.update(
            {
                "matrix_variant": name,
                "representation": spec["description"],
                "selection_policy": (
                    "checkpoint selected by validation chain-macro AP; pooled validation AP is a guardrail; "
                    "F1 threshold selected only after checkpoint lock; test hidden during training"
                ),
                "m7_screen_priority": "conditional" if name.startswith("m7e_") else "primary",
                "test_visibility": "eval_test_each_epoch=false",
            }
        )
        path = output_dir / f"train_{name}_seed{args.seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, width=120))
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
