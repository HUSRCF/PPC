"""Prepare three scratch context arms, reusing the completed D_encoder control."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


VARIANTS = {
    "C_bigru": dict(post_fusion_context_mode="bigru", post_fusion_context_hidden=128),
    "B_pre_tcn": dict(msa_residual_position="pre_tcn"),
    "D_capacity_mlp": dict(post_fusion_context_mode="mlp", post_fusion_context_hidden=576),
}


def build(parent, experiment, name):
    if (parent["model"]["d_seq"] != 90 or parent["model"]["d_prottrans"] != 768
            or not parent["data"]["require_prottrans_embeddings"]
            or parent["training"]["mass_loss_type"] != "ratio_mse"
            or parent["training"]["ratio_loss_weight"] != .75):
        raise ValueError("Expected the old90D D_encoder matched control")
    cfg = copy.deepcopy(parent)
    cfg["model"].update(msa_residual_position="post_tcn", post_fusion_context_mode="none",
                        post_fusion_context_hidden=128, post_fusion_context_dropout=.1,
                        post_fusion_context_scale=.1)
    cfg["model"].update(VARIANTS[name])
    cfg["training"].update(output_dir=str(experiment / "runs" / f"{name}_seed49"), seed=49,
                           epochs=50, init_checkpoint=None, eval_only=False, validation_only=True,
                           eval_test_each_epoch=False, refuse_existing_run=True)
    cfg["metadata"].update(matrix_variant=name, experiment_role="MSA placement and post-fusion context screen",
                            initialization="from scratch; new context init preserves old backbone and CPU RNG",
                            control="Completed D_encoder_seed49, 50 epochs; macro-best13; not rerun",
                            structure_features="original90D, not B0/B1 descriptor fixes",
                            selection_policy_file=str(experiment / "selection_policy.json"),
                            limitation="Single-seed screen; matching initial backbone does not match later dropout RNG consumption")
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-config", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    args = p.parse_args()
    parent = json.loads(args.parent_config.read_text())
    (args.experiment / "configs").mkdir(parents=True, exist_ok=False)
    jobs = []
    for name in VARIANTS:
        cfg = build(parent, args.experiment, name)
        path = args.experiment / "configs" / f"{name}_seed49.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg["training"].update(validation_only=False,
                                    output_dir=str(Path(cfg["training"]["output_dir"]) / "diagnostic"))
        eval_path = path.with_name(path.stem + "_train_eval.json")
        eval_path.write_text(json.dumps(eval_cfg, indent=2) + "\n")
        jobs.append(dict(name=name, config=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         eval_config=str(eval_path), eval_sha256=hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                         output=cfg["training"]["output_dir"], partition="acd_ue"))
    (args.experiment / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    policy = dict(parent_config=str(args.parent_config), parent_sha256=hashlib.sha256(args.parent_config.read_bytes()).hexdigest(),
                  control_reused=True, seed=49, epochs=50, no_test=True, from_scratch=True,
                  primary="validation macro AP", co_primary="validation pooled AP", MAE="diagnostic only",
                  array="0-2 with no concurrency throttle; scheduler/account quotas apply",
                  feature_contract="same original90D, same frozen ESM-C MLC, MSA Transformer, ESM2 contact and union labels",
                  parameter_control="BiGRU+LN296960 versus MLP+LN296256; difference704 (0.24%)",
                  model_selection="No test evaluation; retain every epoch and metric-specific checkpoints",
                  post_training_diagnostic="Read-only train/val eval at validation-macro-selected best.pt; never evaluate test",
                  restriction="Do not infer insufficient model capacity from parameter count alone")
    (args.experiment / "selection_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps({"configs": len(jobs), "root": str(args.experiment)}), flush=True)


if __name__ == "__main__":
    main()
