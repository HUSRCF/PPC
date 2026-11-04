"""Prepare the validation-only D/E x MSA experiment configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def build_config(repo: Path, experiment: Path, loss: str, feature: str, seed: int) -> dict:
    if loss not in {"D", "E"} or feature not in {"none", "profile", "encoder"}:
        raise ValueError((loss, feature))
    cfg = json.loads((repo / "configs/experiments/20261104_m18_union_of3" / f"{loss}.json").read_text())
    cfg = copy.deepcopy(cfg)
    training = cfg["training"]
    training.update(output_dir=str(experiment / "runs" / f"{loss}_{feature}_seed{seed}"),
                    seed=seed, init_checkpoint=None, eval_only=False, epochs=50,
                    validation_only=True, eval_test_each_epoch=False,
                    refuse_existing_run=True, aux_ratio_loss_weight=0.,
                    ratio_loss_source="residue_mean")
    cfg["model"]["use_chain_ratio_head"] = False
    if feature != "none":
        training["allow_random_init_residual"] = True
        cfg["data"].update(prottrans_embedding_root=str(experiment / "features" / feature),
                           require_prottrans_embeddings=True)
        cfg["model"].update(use_prottrans_embeddings=True,
                            d_prottrans=25 if feature == "profile" else 768,
                            prottrans_layer_count=1,
                            prottrans_fusion_mode="zero_init_pre_contact_residual",
                            prottrans_residual_dim=64, prottrans_residual_hidden=64,
                            prottrans_residual_dropout=0.)
    cfg["metadata"].update(matrix_variant=f"{loss}_{feature}", matrix_seed=seed,
                            integrated_replay=False,
                            experiment_role="D/E x frozen MSA, validation-only paired-seed pilot",
                            original_results="No new result yet; separate fresh output root",
                            msa_feature=feature,
                            msa_source="Frozen unpaired UniRef90 2026_03; no templates/partner",
                            msa_fusion="Existing generic frozen-embedding carrier named prottrans; not ProtT5",
                            selection_policy_file=str(experiment / "selection_policy.json"),
                            limitation="MSA adapters add parameters; this is an augmented-system comparison, not parameter-matched causality")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment
    config_dir = experiment / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for feature in ("none", "profile", "encoder"):
        for seed in (49, 50, 51):
            for loss in ("D", "E"):
                cfg = build_config(args.repo, experiment, loss, feature, seed)
                path = config_dir / f"{loss}_{feature}_seed{seed}.json"
                path.write_text(json.dumps(cfg, indent=2) + "\n")
                jobs.append({"loss": loss, "feature": feature, "seed": seed,
                             "partition": "acd_ue" if seed == 49 else "acd_u",
                             "config": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "output": cfg["training"]["output_dir"]})
    (experiment / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    policy = {"test_evaluated": False, "from_scratch": True, "epochs": 50,
              "seeds": [49, 50, 51], "primary_checkpoint_metric": "chain_ap_macro",
              "partition_by_seed": {"49": "acd_ue", "50": "acd_u", "51": "acd_u"},
              "co_primary_report_metric": "pooled_ap",
              "secondary_metrics": ["chain_mcc_at_0_5", "chain_f1_at_0_5", "chain_ratio_mae_raw", "chain_brier_macro"],
              "promotion": "Joint macro AP and pooled AP improvement across paired seeds; MAE is diagnostic, not a tradeoff that justifies AP loss",
              "selection": "Retain every epoch and metric-specific checkpoints; no final test selection in this pilot",
              "no_test_threshold_search": True,
              "paired_loss_contrast": "D/E within each feature/seed have identical architecture, initialization and inputs",
              "feature_contrast_limitation": "Adapters add parameters and consume constructor RNG; same-seed feature arms are not identical-backbone-initialization controls",
              "normalization": "Profile has fixed bounded definitions; no validation/test fitted statistics",
              "long_sequences": "Encoder uses full-length overlap-averaged windows, never drops/truncates chains"}
    (experiment / "selection_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps({"configs": len(jobs), "root": str(experiment)}, indent=2))


if __name__ == "__main__":
    main()
