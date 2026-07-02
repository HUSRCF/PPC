#!/usr/bin/env python3
"""Export ESM-site test residue scores and compute per-chain AP on CPU."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_esm_site import _apply_config, _build_loader, _load_yaml, _move_batch, _read_ids  # noqa: E402
from ppcbind.models import ESMSiteClassifier  # noqa: E402

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for AP/AUROC export") from exc


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    if not labels:
        return None
    if sum(labels) == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _default_args() -> SimpleNamespace:
    return SimpleNamespace(
        esm_root=Path("features/esm2_t33_650M_UR50D/pt"),
        label_root=Path("features/contact_labels"),
        manifest=Path("features/contact_labels/manifest.csv"),
        split_dir=Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit"),
        sequence_feature_root=None,
        contact_graph_root=None,
        aux_contact_graph_root=None,
        require_sequence_features=False,
        output_dir=None,
        max_residues=0,
        eval_crop_mode="none",
        seed=42,
        strict_ids=True,
        require_labels=True,
        strict_label_metadata=True,
        require_contact_graph=None,
        require_aux_contact_graph=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config_path = args.seed_dir / "config.yaml"
    checkpoint = args.seed_dir / "best.pt"
    config = _load_yaml(config_path)
    cfg_args = _default_args()
    model_config = _apply_config(cfg_args, config)
    device = torch.device("cpu")
    require_contact_graph = (
        bool(model_config.get("use_contact_graph"))
        if cfg_args.require_contact_graph is None
        else bool(cfg_args.require_contact_graph)
    )

    ids = _read_ids(cfg_args.split_dir / f"{args.split}_ids.txt")
    loader = _build_loader(
        cfg_args.esm_root,
        cfg_args.label_root,
        ids,
        cfg_args.sequence_feature_root,
        cfg_args.contact_graph_root,
        cfg_args.aux_contact_graph_root,
        cfg_args.require_sequence_features,
        args.batch_size,
        args.num_workers,
        False,
        None,
        cfg_args.max_residues,
        cfg_args.eval_crop_mode,
        cfg_args.seed + 100,
        False,
        preload=False,
        strict_ids=cfg_args.strict_ids,
        require_labels=cfg_args.require_labels,
        strict_label_metadata=cfg_args.strict_label_metadata,
        require_contact_graph=require_contact_graph,
        require_aux_contact_graph=cfg_args.require_aux_contact_graph,
    )

    model = ESMSiteClassifier(model_config).to(device)
    ckpt = _load_checkpoint(checkpoint, device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    seed_name = args.seed_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chain_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    raw_path = args.out_dir / f"{seed_name}_{args.split}_residue_scores.tsv.gz"

    with gzip.open(raw_path, "wt", newline="") as raw_handle, torch.no_grad():
        raw_fields = ["seed", "pdb_id", "chain_local_id", "position_0based", "score", "label"]
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields, delimiter="\t")
        raw_writer.writeheader()
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            probs = torch.softmax(model(**inputs)["logits"].float(), dim=-1)[..., 1].cpu()
            labels_cpu = labels.cpu().long()
            chains_cpu = batch["chain_ids"].cpu().long()
            valid_cpu = labels_cpu != -100

            for item_idx, pdb_id in enumerate(batch["pdb_id"]):
                valid = valid_cpu[item_idx]
                scores = [float(x) for x in probs[item_idx][valid].tolist()]
                y = [int(x) for x in labels_cpu[item_idx][valid].tolist()]
                chains = [int(x) for x in chains_cpu[item_idx][valid].tolist()]
                sample_rows.append(
                    {
                        "seed": seed_name,
                        "pdb_id": str(pdb_id),
                        "n_residues": len(y),
                        "n_positive": sum(y),
                        "average_precision": _average_precision(y, scores),
                        "auroc": _auroc(y, scores),
                    }
                )

                by_chain: dict[int, dict[str, list[float] | list[int]]] = {}
                for pos, (chain_id, score, label) in enumerate(zip(chains, scores, y)):
                    raw_writer.writerow(
                        {
                            "seed": seed_name,
                            "pdb_id": str(pdb_id),
                            "chain_local_id": chain_id,
                            "position_0based": pos,
                            "score": score,
                            "label": label,
                        }
                    )
                    by_chain.setdefault(chain_id, {"scores": [], "labels": []})
                    by_chain[chain_id]["scores"].append(score)  # type: ignore[index]
                    by_chain[chain_id]["labels"].append(label)  # type: ignore[index]

                for chain_id, values in sorted(by_chain.items()):
                    cy = [int(x) for x in values["labels"]]
                    cs = [float(x) for x in values["scores"]]
                    n_pos = sum(cy)
                    n = len(cy)
                    chain_rows.append(
                        {
                            "seed": seed_name,
                            "pdb_id": str(pdb_id),
                            "chain_local_id": chain_id,
                            "chain_key": f"{pdb_id}__chain{chain_id}",
                            "n_residues": n,
                            "n_positive": n_pos,
                            "n_negative": n - n_pos,
                            "average_precision": _average_precision(cy, cs),
                            "auroc": _auroc(cy, cs),
                            "two_class": int(0 < n_pos < n),
                        }
                    )

    chain_tsv = args.out_dir / f"{seed_name}_{args.split}_chain_ap.tsv"
    with chain_tsv.open("w", newline="") as handle:
        fields = [
            "seed",
            "pdb_id",
            "chain_local_id",
            "chain_key",
            "n_residues",
            "n_positive",
            "n_negative",
            "average_precision",
            "auroc",
            "two_class",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(chain_rows)

    sample_tsv = args.out_dir / f"{seed_name}_{args.split}_sample_ap.tsv"
    with sample_tsv.open("w", newline="") as handle:
        fields = ["seed", "pdb_id", "n_residues", "n_positive", "average_precision", "auroc"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(sample_rows)

    chain_ap = [float(row["average_precision"]) for row in chain_rows if row["average_precision"] is not None]
    chain_auc = [float(row["auroc"]) for row in chain_rows if row["auroc"] is not None]
    sample_ap = [float(row["average_precision"]) for row in sample_rows if row["average_precision"] is not None]
    summary = {
        "seed": seed_name,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "n_samples": len(sample_rows),
        "n_chains": len(chain_rows),
        "n_residues": sum(int(row["n_residues"]) for row in chain_rows),
        "n_positive": sum(int(row["n_positive"]) for row in chain_rows),
        "n_all_positive_chains": sum(
            1 for row in chain_rows if int(row["n_positive"]) == int(row["n_residues"])
        ),
        "n_all_negative_chains": sum(1 for row in chain_rows if int(row["n_positive"]) == 0),
        "chain_ap_macro": sum(chain_ap) / len(chain_ap),
        "chain_auroc_macro_two_class_only": sum(chain_auc) / len(chain_auc) if chain_auc else None,
        "sample_ap_macro": sum(sample_ap) / len(sample_ap),
        "chain_ap_tsv": str(chain_tsv),
        "sample_ap_tsv": str(sample_tsv),
        "residue_scores_tsv_gz": str(raw_path),
    }
    summary_path = args.out_dir / f"{seed_name}_{args.split}_chain_ap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
