#!/usr/bin/env python3
"""Evaluate a trained ESM residue contact-site checkpoint with threshold/top-k metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_esm_site import (  # noqa: E402
    _apply_config,
    _build_loader,
    _class_weight_from_manifest,
    _evaluate,
    _limit_ids,
    _load_yaml,
    _parse_float_grid,
    _read_ids,
    _split_ids_path,
    _lengths_from_manifest,
    ResidueSiteLoss,
)
from ppcbind.models import ESMSiteClassifier  # noqa: E402


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--threshold-grid", default="0.01:0.99:0.01")
    parser.add_argument("--topk-fracs", default="0.05,0.10")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--esm-root", default="features/esm2_t33_650M_UR50D/pt", type=Path)
    parser.add_argument("--label-root", default="features/contact_labels", type=Path)
    parser.add_argument("--manifest", default="features/contact_labels/manifest.csv", type=Path)
    parser.add_argument("--split-dir", default="features/contact_labels/splits_mmseq30_tmk_no_len_limit", type=Path)
    parser.add_argument("--sequence-feature-root", default=None, type=Path)
    parser.add_argument("--primary-embedding-root", default=None, type=Path)
    parser.add_argument("--prottrans-embedding-root", default=None, type=Path)
    parser.add_argument("--chain-filter-manifest", default=None, type=Path)
    parser.add_argument("--require-primary-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-prottrans-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contact-graph-root", default=None, type=Path)
    parser.add_argument("--aux-contact-graph-root", default=None, type=Path)
    parser.add_argument("--require-sequence-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-label-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-sequence-feature-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-contact-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-aux-contact-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-residues", type=int, default=0)
    parser.add_argument("--eval-crop-mode", choices=["none", "first", "random"], default="none")
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    model_config = _apply_config(args, config)
    splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    if not splits or any(split not in {"train", "val", "test"} for split in splits):
        raise ValueError("splits must contain train, val or test")
    if args.validation_only and any(split != "val" for split in splits):
        raise ValueError("This configuration is validation-only; use --splits val")
    thresholds = _parse_float_grid(args.threshold_grid, [i / 100.0 for i in range(1, 100)])
    topk_fracs = _parse_float_grid(args.topk_fracs, [0.05, 0.10])

    if args.output_dir is None:
        raise ValueError("training.output_dir must be set in YAML config or CLI")
    checkpoint = args.checkpoint or (args.output_dir / "best.pt")
    output_json = args.output_json or (args.output_dir / "eval_metrics.json")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    chain_filtered = args.chain_filter_manifest is not None
    train_ids = _read_ids(_split_ids_path(args.split_dir, "train", chain_filtered), lowercase=not chain_filtered)
    pos_weight, train_pos, train_neg = _class_weight_from_manifest(args.manifest, train_ids, args.max_pos_weight)
    if getattr(args, "class_weight_mode", "auto") == "none":
        pos_weight = 1.0
    require_contact_graph = bool(model_config.get("use_contact_graph")) if args.require_contact_graph is None else bool(args.require_contact_graph)

    model = ESMSiteClassifier(model_config).to(device)
    ckpt = _load_checkpoint(checkpoint, device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    criterion = ResidueSiteLoss(
        pos_weight=pos_weight, label_smoothing=args.label_smoothing,
        loss_type=getattr(args, "loss_type", "ce"), dice_weight=getattr(args, "dice_weight", 0.0),
        focal_gamma=getattr(args, "focal_gamma", 2.0), focal_alpha=getattr(args, "focal_alpha", None),
        reduction=getattr(args, "loss_reduction", "residue_mean"),
        ratio_loss_weight=getattr(args, "ratio_loss_weight", 0.0), device=device,
        mass_loss_type=getattr(args, "mass_loss_type", "ratio_mse"),
        aux_ratio_loss_weight=getattr(args, "aux_ratio_loss_weight", 0.0),
        ratio_loss_source=getattr(args, "ratio_loss_source", "auto"),
    )

    results: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "device_resolved": str(device),
        "train_positive_residues": train_pos,
        "train_negative_residues": train_neg,
        "pos_weight": pos_weight,
        "thresholds": thresholds,
        "topk_fracs": topk_fracs,
        "strict_ids": args.strict_ids,
        "require_labels": args.require_labels,
        "strict_label_metadata": args.strict_label_metadata,
        "require_contact_graph_resolved": require_contact_graph,
        "require_aux_contact_graph": args.require_aux_contact_graph,
        "splits": {},
    }
    for split in splits:
        ids = _read_ids(_split_ids_path(args.split_dir, split, chain_filtered), lowercase=not chain_filtered)
        ids = _limit_ids(ids, args.max_samples)
        loader = _build_loader(
            esm_root=args.esm_root, label_root=args.label_root, ids=ids,
            sequence_feature_root=args.sequence_feature_root,
            primary_embedding_root=args.primary_embedding_root,
            contact_graph_root=args.contact_graph_root, aux_contact_graph_root=args.aux_contact_graph_root,
            prottrans_embedding_root=args.prottrans_embedding_root,
            require_sequence_features=args.require_sequence_features,
            require_primary_embeddings=args.require_primary_embeddings,
            require_prottrans_embeddings=args.require_prottrans_embeddings,
            batch_size=getattr(args, "eval_batch_size", 0) or args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor,
            persistent_workers=getattr(args, "persistent_workers", False),
            payload_cache_size=getattr(args, "payload_cache_size", 0),
            max_residues=args.max_residues, crop_mode=args.eval_crop_mode, seed=args.seed + 100,
            shuffle=False, batching=getattr(args, "batching", "random"),
            max_batch_tokens=getattr(args, "max_batch_tokens", 0),
            length_bucket_size=getattr(args, "length_bucket_size", 0),
            lengths_by_id=_lengths_from_manifest(args.manifest), preload=getattr(args, "preload", False),
            strict_ids=args.strict_ids,
            require_labels=args.require_labels,
            strict_label_metadata=args.strict_label_metadata,
            strict_sequence_feature_metadata=args.strict_sequence_feature_metadata,
            require_contact_graph=require_contact_graph,
            require_aux_contact_graph=args.require_aux_contact_graph,
            chain_filter_manifest=args.chain_filter_manifest,
        )
        results["splits"][split] = _evaluate(
            model,
            loader,
            criterion,
            device,
            max_batches=args.eval_max_batches,
            show_progress=args.progress,
            desc=f"eval {split}",
            thresholds=thresholds,
            topk_fracs=topk_fracs,
            strict_gradient_isolation=getattr(args, "strict_eval_gradient_isolation", True),
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({"event": "evaluated", "output_json": str(output_json), "splits": results["splits"]}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
