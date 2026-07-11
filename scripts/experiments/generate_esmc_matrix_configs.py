#!/usr/bin/env python3
"""Generate the controlled M0-M5 ESM2/ESM-C comparison configs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SPLIT_ROOT = "features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708"
ESM2_CACHE = "features/esm2_mlc_chain_cache_20260711"
ESMC_ROOT = "features/esmc_600m_2024_12_mlc_l12_l24_l36"


def _base() -> dict:
    return {
        "data": {
            "esm_root": f"{ESM2_CACHE}/compact_pdb",
            "primary_embedding_root": f"{ESM2_CACHE}/by_chain",
            "contact_graph_root": f"{ESM2_CACHE}/by_chain",
            "label_root": "features/contact_labels_5p5A",
            "manifest": f"{SPLIT_ROOT}/chain_filtered_training/manifest.csv",
            "split_dir": f"{SPLIT_ROOT}/chain_filtered_training",
            "chain_filter_manifest": f"{SPLIT_ROOT}/pypropel_chain_filtered_no_len_limit/chain_manifest.csv",
            "sequence_feature_root": "features/esmfold_pred_struct_sequence_scalar_chain_filtered/pt",
            "require_sequence_features": True,
            "require_primary_embeddings": True,
            "require_contact_graph": True,
        },
        "model": {
            "d_esm": 3840,
            "d_seq": 90,
            "d_model": 256,
            "d_hidden": 512,
            "dropout": 0.25,
            "classifier_dropout": 0.45,
            "max_chains": 128,
            "use_chain_embedding": True,
            "use_seq_features": True,
            "use_position_features": True,
            "use_global_context": True,
            "use_contact_graph": True,
            "contact_graph_layers": 1,
            "contact_score_clip": 1.0,
            "n_transformer_layers": 0,
            "esm_layer_fusion": "concat",
            "esm_layer_count": 3,
        },
        "training": {
            "output_dir": "",
            "device": "cuda:0",
            "seed": 42,
            "epochs": 30,
            "batch_size": 64,
            "eval_batch_size": 64,
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "payload_cache_size": 4,
            "preload": False,
            "batching": "token_budget",
            "max_batch_tokens": 65536,
            "length_bucket_size": 1024,
            "max_residues": 0,
            "train_crop_mode": "none",
            "eval_crop_mode": "none",
            "max_train_samples": None,
            "max_val_samples": 0,
            "eval_max_batches": None,
            "eval_test_each_epoch": True,
            "lr": 5.0e-5,
            "warmup_steps": 0,
            "warmup_ratio": 0.1,
            "min_lr_scale": 0.02,
            "weight_decay": 0.1,
            "label_smoothing": 0.05,
            "threshold_grid": "0.01:0.99:0.01",
            "topk_fracs": "0.05,0.10",
            "selection_metric": "f1_best_threshold",
            "max_pos_weight": 20.0,
            "grad_clip_norm": 1.0,
            "grad_value_clip": 100.0,
            "adam_eps": 1.0e-7,
            "amp": True,
            "progress": True,
            "strict_ids": True,
            "require_labels": True,
            "strict_label_metadata": False,
            "strict_sequence_feature_metadata": False,
        },
        "metadata": {
            "label_cutoff_angstrom": 5.5,
            "split_policy": "global_pairwise_deeptminter_si30_chain_filtered_no_len_limit",
            "structure_source": "ESMFold exact-sequence unique-chain PDBs only",
            "bridge_policy": "no UniProt/TmAlphaFold/AFDB/PDBTM bridge",
            "contact_prior": "frozen ESM2 contact graph; top_k=16, min_score=0.05, min_seq_sep=6, bidirectional",
            "selection_policy": "checkpoint and threshold selected on validation only; test evaluated at frozen validation threshold",
            "loader_policy": "chain-level exact-sequence payloads + compact PDB metadata; batch64/tokens65536/workers4/persistent/cache4",
            "parent_hyperparameters": "hp_highdrop_wd1e1_ls5e2 seed42",
        },
    }


def _variants() -> dict[str, dict]:
    esmc_chain = f"{ESMC_ROOT}/by_chain"
    return {
        "m0_esm2_mlc": {
            "metadata": {"representation": "ESM2 layers 11/22/33 concat (3840D)"},
        },
        "m1_esmc_final": {
            "data": {"primary_embedding_root": esmc_chain},
            "model": {"d_esm": 3456, "esm_layer_fusion": "last", "esm_layer_count": 3},
            "metadata": {"representation": "ESM-C block 36 only (1152D from stored 3456D MLC)"},
        },
        "m2_esmc_mlc_concat": {
            "data": {"primary_embedding_root": esmc_chain},
            "model": {"d_esm": 3456, "esm_layer_fusion": "concat", "esm_layer_count": 3},
            "metadata": {"representation": "ESM-C blocks 12/24/36 concat (3456D)"},
        },
        "m3_esmc_mlc_scalar_mix": {
            "data": {"primary_embedding_root": esmc_chain},
            "model": {"d_esm": 3456, "esm_layer_fusion": "scalar_mix", "esm_layer_count": 3},
            "metadata": {"representation": "learned scalar mix of ESM-C blocks 12/24/36"},
        },
        "m4_esm2_esmc_projected_concat": {
            "data": {
                "prottrans_embedding_root": esmc_chain,
                "require_prottrans_embeddings": True,
            },
            "model": {
                "use_prottrans_embeddings": True,
                "d_prottrans": 3456,
                "prottrans_fusion_mode": "projected_concat",
            },
            "metadata": {
                "representation": "parameter-matched ESM2->128 + ESM-C->128 projected concat",
                "projection_parameter_policy": "approximately matched to the M0 3840->256 projection",
            },
        },
        "m5_esm2_esmc_gated_residual": {
            "data": {
                "prottrans_embedding_root": esmc_chain,
                "require_prottrans_embeddings": True,
            },
            "model": {
                "use_prottrans_embeddings": True,
                "d_prottrans": 3456,
                "prottrans_fusion_mode": "gated_residual",
                "prottrans_gate_input_mode": "full",
                "prottrans_gate_bias": -2.0,
            },
            "metadata": {"representation": "full ESM2 and ESM-C projections with gated residual fusion"},
        },
    }


def _merge(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-suffix", default="20260711_hpc2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, patch in _variants().items():
        config = copy.deepcopy(_base())
        _merge(config, patch)
        config["training"]["output_dir"] = f"runs/contact_site_cfsi30_esmc_matrix_{name}_seed42_e30_{args.run_suffix}"
        config["metadata"]["matrix_variant"] = name
        path = args.output_dir / f"train_{name}_seed42.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, width=120))
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
