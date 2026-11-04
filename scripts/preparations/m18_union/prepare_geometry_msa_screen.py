"""Prepare a frozen-input geometry/MSA screen targeting validation pooled AP."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


VARIANTS = {
    "K0_B1_MSA": {},
    "K1_G2_MSA": {},
    "K2_G2_MSA_preTCN": {"msa_residual_position": "pre_tcn"},
    "K3_G2_MSA_twohop": {"geometry_layers": 2},
    "K4_G2_MSA_dualgraph": {"use_contact_graph": True, "contact_graph_layers": 1},
    "K5_G2_MSA_postgeo": {"msa_residual_position": "post_geometry"},
    "K6_G2_MSA_fullcontext": {"use_global_context": False, "n_transformer_layers": 1},
    "K7_G2_MSA_residue_mean": {},
    "K8_G2_MSA_pure_CE": {},
    "K9_G2_MSA_evo_edges": {"geometry_msa_edge_bias": True},
    "K10_G2_MSA_evo_edges_twohop": {"geometry_msa_edge_bias": True, "geometry_layers": 2},
    "K11_G2_MSA_twohop_pure_CE": {"geometry_layers": 2},
    "K12_G2_MSA_threehop": {"geometry_layers": 3},
    "K13_G2_MSA_threehop_pure_CE": {"geometry_layers": 3},
    "K14_G2_MSA_twohop_width128": {
        "geometry_layers": 2, "prottrans_residual_dim": 128, "prottrans_residual_hidden": 128,
    },
    "K15_G2_MSA_twohop_width256": {
        "geometry_layers": 2, "prottrans_residual_dim": 256, "prottrans_residual_hidden": 128,
    },
    "K16_G2_MSA_twohop_capacity128": {
        "geometry_layers": 2, "prottrans_residual_dim": 64, "prottrans_residual_hidden": 228,
    },
    "K17_G2_MSA_twohop_capacity256": {
        "geometry_layers": 2, "prottrans_residual_dim": 64, "prottrans_residual_hidden": 428,
    },
    "K18_G2_MSA_twohop_preTCN": {"geometry_layers": 2, "msa_residual_position": "pre_tcn"},
    "K19_G2_MSA_twohop_dropout_low": {
        "geometry_layers": 2, "dropout": .15, "classifier_dropout": .30,
    },
    "K20_G2_MSA_twohop_dropout_high": {
        "geometry_layers": 2, "dropout": .35, "classifier_dropout": .55,
    },
    "K21_G2_MSA_twohop_wd003": {"geometry_layers": 2},
    "K22_G2_MSA_twohop_wd020": {"geometry_layers": 2},
    "K23_G2_MSA_twohop_ls002": {"geometry_layers": 2},
    "K24_G2_MSA_twohop_msa_drop010": {
        "geometry_layers": 2, "prottrans_residual_dropout": .10,
    },
    "K25_G2_MSA_twohop_preTCN_ls002": {"geometry_layers": 2, "msa_residual_position": "pre_tcn"},
    "K26_G2_MSA_twohop_preTCN_pure_CE": {"geometry_layers": 2, "msa_residual_position": "pre_tcn"},
    "K27_G2_MSA_twohop_preTCN_width256": {
        "geometry_layers": 2, "msa_residual_position": "pre_tcn",
        "prottrans_residual_dim": 256, "prottrans_residual_hidden": 128,
    },
    "K28_G2_MSA_twohop_residue_mean": {"geometry_layers": 2},
    "K29_G2_MSA_twohop_preTCN_residue_mean": {"geometry_layers": 2, "msa_residual_position": "pre_tcn"},
}
MSA_KEYS = (
    "use_prottrans_embeddings", "d_prottrans", "prottrans_layer_count",
    "prottrans_fusion_mode", "prottrans_residual_dim", "prottrans_residual_hidden",
    "prottrans_residual_dropout",
)
IDENTITY_KEYS = ("esm_root", "primary_embedding_root", "label_root", "manifest", "split_dir")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def absolute_path(value):
    return Path(value).expanduser().resolve()


def build(b1, g2, msa, experiment, name):
    if name not in VARIANTS:
        raise ValueError(name)
    if any(p["data"][k] != b1["data"][k] for p in (g2, msa) for k in IDENTITY_KEYS):
        raise ValueError("Parent inputs/splits/labels differ")
    for parent in (b1, g2, msa):
        t = parent["training"]
        if (t["seed"] != 49 or t["mass_loss_type"] != "ratio_mse"
                or t["ratio_loss_weight"] != .75 or t["loss_reduction"] != "chain_mean"):
            raise ValueError("Expected seed49 D-loss parents")
    if (b1["model"]["d_seq"] != 91 or g2["model"]["d_seq"] != 91
            or g2["model"]["geometry_mode"] != "edge_attention"
            or b1["data"]["sequence_feature_root"] != g2["data"]["sequence_feature_root"]
            or msa["model"]["d_prottrans"] != 768
            or not msa["data"]["require_prottrans_embeddings"]
            or msa["model"]["prottrans_fusion_mode"] != "zero_init_pre_contact_residual"):
        raise ValueError("Expected corrected91D B1/G2 and frozen MSA Transformer")
    cfg = copy.deepcopy(b1 if name == "K0_B1_MSA" else g2)
    cfg["data"].update(
        prottrans_embedding_root=msa["data"]["prottrans_embedding_root"],
        require_prottrans_embeddings=True,
    )
    cfg["model"].update({k: msa["model"][k] for k in MSA_KEYS})
    cfg["model"].update(msa_residual_position="post_tcn", post_fusion_context_mode="none")
    cfg["model"].update(VARIANTS[name])
    if name == "K4_G2_MSA_dualgraph":
        cfg["data"].update(contact_graph_root=msa["data"]["contact_graph_root"],
                           require_contact_graph=True)
    t = cfg["training"]
    t.update(output_dir=str(experiment / "runs" / f"{name}_seed49"), seed=49, epochs=50,
             init_checkpoint=None, eval_only=False, validation_only=True,
             eval_test_each_epoch=False, refuse_existing_run=True,
             allow_random_init_residual=True, aux_ratio_loss_weight=0.,
             selection_metric="pooled_ap",
             prediction_provenance=str(experiment / "prediction_provenance.json"))
    metrics = t["save_metric_checkpoints"].split(",")
    t["save_metric_checkpoints"] = ",".join(dict.fromkeys([*metrics, "chain_ap_macro", "pooled_ap"]))
    if name in {"K7_G2_MSA_residue_mean", "K28_G2_MSA_twohop_residue_mean",
                "K29_G2_MSA_twohop_preTCN_residue_mean"}:
        t["loss_reduction"] = "residue_mean"
    if name in {"K8_G2_MSA_pure_CE", "K11_G2_MSA_twohop_pure_CE", "K13_G2_MSA_threehop_pure_CE",
                "K26_G2_MSA_twohop_preTCN_pure_CE"}:
        t["ratio_loss_weight"] = 0.
    if name in {"K21_G2_MSA_twohop_wd003", "K22_G2_MSA_twohop_wd020"}:
        t["weight_decay"] = .03 if name == "K21_G2_MSA_twohop_wd003" else .20
    if name in {"K23_G2_MSA_twohop_ls002", "K25_G2_MSA_twohop_preTCN_ls002"}:
        t["label_smoothing"] = .02
    cfg["metadata"].update(
        matrix_variant=name, matrix_seed=49, experiment_role="geometry x MSA pooled-AP screen",
        msa_feature="encoder", msa_source="Frozen unpaired UniRef90 2026_03; no templates/partner",
        msa_fusion="768D MSA Transformer query row via zero-init residual adapter; not ProtT5",
        initialization="from scratch; no resume/checkpoint initialization",
        original_results="No result before execution; old runs unchanged",
        selection_policy_file=str(experiment / "selection_policy.json"),
        selection_policy="Validation pooled AP target; report same-checkpoint macro AP and both metric-best checkpoints",
        contact_prior=("ESM2 contact followed by OF3 geometry" if name == "K4_G2_MSA_dualgraph"
                       else "ESM2 contact" if name == "K0_B1_MSA" else "OF3 monomer geometry"),
        limitation="Single-seed system screen; constructor/dropout RNG differ across architecture/feature arms",
    )
    return cfg


def verify(repo, experiment, task):
    jobs = load(experiment / "jobs.json")
    if not 0 <= task < len(jobs):
        raise ValueError(task)
    job = jobs[task]
    if sha(job["config"]) != job["sha256"]:
        raise ValueError("Config checksum mismatch")
    cfg = load(job["config"])
    for source in load(experiment / "parents.json").values():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Parent config changed")
    geo = load(Path(cfg["data"]["sequence_feature_root"]).parent / "ready.json")
    msa = load(Path(cfg["data"]["prottrans_embedding_root"]) / "validation_summary.json")
    assert geo["sequences"] == msa["sequences"] == 6667
    assert geo["full_length"] and msa["full_length"] and msa["exact_queries"] and msa["finite"]
    assert not geo["labels_read"] and not geo["test_labels_read"] and not geo["smoke"]
    assert geo["fit_split"] == geo["normalization"]["fit_split"] == "train"
    assert geo["normalization"]["fit_sequences"] == 4868
    assert not msa["labels_read"] and msa["mode"] == "encoder" and msa["dimension"] == 768
    assert msa["split_counts"] == {"train": 4868, "val": 1075, "test": 724}
    assert geo["source_code_sha256"] == sha(repo / "scripts/preparations/m18_union/build_geometry_features.py")
    assert msa["source_code_sha256"] == sha(repo / "scripts/preparations/m18_union/build_msa_features.py")
    assert len(load(cfg["training"]["prediction_provenance"])["records"]) == 1075
    t = cfg["training"]
    assert t["validation_only"] and not t["eval_test_each_epoch"] and t["init_checkpoint"] is None
    assert t["seed"] == 49 and t["epochs"] == 50 and t["selection_metric"] == "pooled_ap"
    assert t["max_residues"] == 0 and t["require_labels"] and cfg["data"]["require_primary_embeddings"]
    return job["config"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for name in ("b1", "g2", "msa", "repo", "experiment"):
        prep.add_argument("--" + name, type=absolute_path, required=True)
    prep.add_argument("--variants", nargs="+", choices=tuple(VARIANTS))
    check = sub.add_parser("verify")
    for name in ("repo", "experiment"):
        check.add_argument("--" + name, type=absolute_path, required=True)
    check.add_argument("--task", type=int, required=True)
    a = p.parse_args()
    if a.mode == "verify":
        print(verify(a.repo, a.experiment, a.task), flush=True)
        return
    variants = a.variants or list(VARIANTS)
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate variants")
    parents = {key: {"path": str(getattr(a, key)), "sha256": sha(getattr(a, key))}
               for key in ("b1", "g2", "msa")}
    b1, g2, msa = (load(getattr(a, key)) for key in ("b1", "g2", "msa"))
    (a.experiment / "configs").mkdir(parents=True, exist_ok=False)
    jobs = []
    for name in variants:
        cfg = build(b1, g2, msa, a.experiment, name)
        path = a.experiment / "configs" / f"{name}_seed49.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        jobs.append({"name": name, "config": str(path), "sha256": sha(path),
                     "output": cfg["training"]["output_dir"], "partition": "acd_ue"})
    provenance_path = Path(g2["training"]["prediction_provenance"])
    provenance = load(provenance_path)
    if len(provenance.get("records", {})) != 1075:
        raise ValueError("Incomplete validation provenance")
    provenance["geometry_msa_screen"] = dict(
        parents=parents, parent_provenance_sha256=sha(provenance_path),
        code=str(a.repo), labels="unchanged exact-sequence union v2,5.5A",
        split="unchanged original pairwise SI30 components", test_evaluated=False,
    )
    policy = dict(
        target_pooled_AP=.80, primary="validation pooled AP",
        macro_guardrail=.7467948466002535 - .005,
        macro_guardrail_meaning="No more than0.005 below prior G2 at the same selected checkpoint",
        co_report="macro AP, AUROC, F1/MCC with validation gates, ratio MAE, loss",
        reference_G2={"macro_AP": .7467948466002535, "pooled_AP": .7802500425571836},
        epochs=50, seed=49, from_scratch=True, test_evaluated=False,
        labels="exact-sequence ever-observed union5.5A",
        feature_contract="Corrected91D OF3, ESM-C MLC3456D, frozen768D MSA Transformer",
        architecture_rows="D loss except K7/K28/K29 residue_mean CE aggregation and K8/K11/K13/K26 ratio weight0",
        regularization_rows="K19/K20 main+classifier dropout;K21/K22 weight_decay;K23 label_smoothing;K24 elementwise MSA projection+adapter dropout",
        interaction_rows="K25/K26/K27 complete preTCN x label_smoothing/pure_CE/MSA_width factorial cells",
        aggregation_rows="K28/K29 complete twohop CE aggregation x MSA fusion-position cells",
        selected_variants=variants,
        checkpoints="Retain every epoch and both AP-best checkpoints; never combine scores from different epochs",
        concurrency="No artificial array throttle; scheduler/account quotas apply",
        promotion="Exploratory validation screen only; later seed/holdout confirmation required",
    )
    for name, obj in (("jobs", jobs), ("parents", parents), ("selection_policy", policy),
                      ("prediction_provenance", provenance)):
        (a.experiment / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
    for task in range(len(jobs)):
        verify(a.repo, a.experiment, task)
    print(json.dumps({"configs": len(jobs), "root": str(a.experiment)}), flush=True)


if __name__ == "__main__":
    main()
