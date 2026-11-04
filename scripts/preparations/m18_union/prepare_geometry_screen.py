"""Generate single-seed, D-loss architecture controls; never evaluate test."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


VARIANTS = {
    "B0_fixed90": ("fixed90", {}),
    "B1_fixed91q": ("fixed91q", {}),
    "G0_of3_mean": ("fixed91q", {}),
    "G1_edge_mlp": ("fixed91q", {"use_contact_graph": False, "geometry_mode": "edge_mlp"}),
    "G2_edge_attention": ("fixed91q", {"use_contact_graph": False, "geometry_mode": "edge_attention"}),
    "G3_gvp": ("fixed91q", {"use_contact_graph": False, "geometry_mode": "gvp"}),
    "C0_full_attention": ("fixed91q", {"use_global_context": False, "n_transformer_layers": 1}),
    "C1_linear_full": ("fixed91q", {"use_global_context": False, "n_transformer_layers": 1,
                                     "linear_attention_layers": 1}),
}


def build(repo, experiment, name):
    feature, changes = VARIANTS[name]
    cfg = json.loads((repo / "configs/experiments/20261104_m18_union_of3/D.json").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["data"]["sequence_feature_root"] = str(experiment / "features" / feature)
    cfg["model"].update(d_seq=90 if feature == "fixed90" else 91, local_mlp_blocks=2,
                        geometry_layers=1, geometry_heads=8)
    cfg["model"].update(changes)
    if name.startswith("G"):
        if name == "G0_of3_mean":
            cfg["data"]["contact_graph_root"] = str(experiment / "features/geometry")
        else:
            cfg["data"].update(geometry_feature_root=str(experiment / "features/geometry"),
                               require_geometry_features=True, require_contact_graph=False,
                               contact_graph_root=None)
    cfg["training"].update(output_dir=str(experiment / "runs" / f"{name}_seed49"), seed=49,
                           epochs=50, init_checkpoint=None, eval_only=False,
                           eval_test_each_epoch=False, validation_only=True,
                           refuse_existing_run=True, aux_ratio_loss_weight=0.,
                           prediction_provenance=str(experiment / "prediction_provenance.json"))
    cfg["metadata"].update(matrix_variant=name, architecture_screen=True, matrix_seed=49,
                            integrated_replay=False, experiment_role="single seed49 D-loss architecture screen",
                            normalization="Parent87 scalar channels retained; corrected2 fit on train only; q/100 bounded",
                            selection_policy_file=str(experiment / "selection_policy.json"),
                            original_results="New architecture run; no result before execution",
                            structure_features=feature, contact_prior=("OF3 monomer geometry" if name.startswith("G") else "ESM2 contact"),
                            limitation="Single-seed augmented-system screen; changed modules consume RNG; not parameter-matched causality")
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--experiment", required=True, type=Path)
    args = p.parse_args()
    (args.experiment / "configs").mkdir(parents=True, exist_ok=False)
    jobs = []
    for name in VARIANTS:
        cfg = build(args.repo, args.experiment, name)
        path = args.experiment / "configs" / f"{name}_seed49.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        jobs.append(dict(name=name, seed=49, config=str(path), partition="acd_ue",
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output=cfg["training"]["output_dir"]))
    (args.experiment / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    policy = dict(seed=49, epochs=50, from_scratch=True, test_evaluated=False,
                  loss="D: chain-mean unweighted CE + 0.75 residue-mean ratio MSE",
                  primary="validation chain-macro AP", co_primary="validation pooled AP",
                  diagnostic=["F1", "MCC", "ratio MAE", "Brier", "loss"],
                  promotion="Joint AP improvement; no ratio-only promotion; no new seed controls in this screen",
                  controls="B0 versus historical D isolates descriptor correction; B1 adds q; G/C versus B1 compare architectures",
                  structure="Frozen selected OF3 single monomer; no experimental assembly or partner geometry",
                  no_test_loader=True, no_resume=True, correction_columns=[8, 51])
    (args.experiment / "selection_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    parent = Path(build(args.repo, args.experiment, "B0_fixed90")["metadata"]["original_frozen_config"]).parent.parent / "prediction_provenance.json"
    provenance = json.loads(parent.read_text())
    if not isinstance(provenance.get("records"), dict) or len(provenance["records"]) != 1075:
        raise ValueError("Incomplete parent validation prediction provenance")
    provenance["geometry_screen"] = dict(
        task="single-chain exact-sequence ever-observed union 5.5A", seed=49,
        structure="OF3 maximum official confidence of 5 existing samples", labels="unchanged union v2",
        split="unchanged original pairwise SI30 components", code=str(args.repo),
        features_receipt=str(args.experiment / "features/ready.json"), validation_only=True,
        parent_provenance_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    (args.experiment / "prediction_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"configs": len(jobs), "root": str(args.experiment)}), flush=True)


if __name__ == "__main__":
    main()
