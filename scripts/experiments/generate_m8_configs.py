#!/usr/bin/env python3
"""Generate validation-only M8 screens from the winning M7c configuration."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


BASE_CONFIG = "configs/experiments/20260712_m7_screen_hpc2/train_m7c_m2_chainmean_tcn2_seed42.yaml"

VARIANTS = {
    "m8a_tcn2_residue_mean": {
        "patch": {"training": {"loss_reduction": "residue_mean"}},
        "description": "M7c TCN2 context with residue-mean CE; isolates the context gain from chain balancing",
    },
    "m8b_tcn3_d124": {
        "patch": {
            "model": {"tcn_layers": 3, "tcn_kernel_size": 7, "tcn_dilations": [1, 2, 4]},
        },
        "description": "M7c with a third gated depthwise TCN block and dilation schedule 1/2/4",
    },
    "m8c_tcn2_kernel11": {
        "patch": {"model": {"tcn_layers": 2, "tcn_kernel_size": 11, "tcn_dilations": [1, 2]}},
        "description": "M7c with a wider kernel-11 two-block TCN receptive field",
    },
    "m8d_tcn2_ratio01": {
        "patch": {"training": {"ratio_loss_weight": 0.1}},
        "description": "M7c plus a 0.1-weight differentiable single-chain effect-site-ratio MSE objective",
    },
    "m8e_tcn2_drop035": {
        "patch": {"model": {"classifier_dropout": 0.35}},
        "description": "M7c with classifier dropout reduced from 0.45 to 0.35",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-suffix", default="20260712_hpc2")
    parser.add_argument("--variants", default="", help="Comma-separated subset; empty means all")
    args = parser.parse_args()

    names = [name.strip() for name in args.variants.split(",") if name.strip()] or list(VARIANTS)
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown M8 variants: {unknown}; available={sorted(VARIANTS)}")

    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((project_root / BASE_CONFIG).read_text())

    for name in names:
        spec = VARIANTS[name]
        config = copy.deepcopy(base)
        merge(config, spec["patch"])
        training = config["training"]
        training.update(
            {
                "output_dir": f"runs/contact_site_cfsi30_m8screen_{name}_seed{args.seed}_e30_{args.run_suffix}",
                "seed": args.seed,
                "epochs": 30,
                "selection_metric": "chain_ap_macro",
                "save_metric_checkpoints": "f1_best_threshold,pr_auc,chain_ap_macro",
                "eval_test_each_epoch": False,
                "strict_eval_gradient_isolation": True,
            }
        )
        training.setdefault("ratio_loss_weight", 0.0)
        metadata = config.setdefault("metadata", {})
        metadata.update(
            {
                "matrix_variant": name,
                "parent_experiment": "M7c M2 chain-mean TCN2 seed42 validation winner",
                "representation": spec["description"],
                "selection_policy": (
                    "checkpoint selected by validation chain-macro AP; pooled validation AP guardrail; "
                    "raw chain-ratio MAE reported diagnostically; test split is not loaded during training"
                ),
                "advance_gate": (
                    "advance only if validation chain-macro AP exceeds M7c seed42 by >=0.005 and pooled AP "
                    "does not regress by >0.003; then confirm with seeds 43/44 before one frozen test evaluation"
                ),
                "test_visibility": "eval_test_each_epoch=false; test IDs and loader remain unloaded",
                "gradient_isolation": (
                    "strict runtime checks require no parameter grads, no grad-enabled logits, and no parameter "
                    "version changes during validation"
                ),
            }
        )
        path = output_dir / f"train_{name}_seed{args.seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, width=120))
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
