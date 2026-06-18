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
from torch import nn

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
    parser.add_argument("--require-sequence-features", action=argparse.BooleanOptionalAction, default=False)
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
    thresholds = _parse_float_grid(args.threshold_grid, [i / 100.0 for i in range(1, 100)])
    topk_fracs = _parse_float_grid(args.topk_fracs, [0.05, 0.10])

    if args.output_dir is None:
        raise ValueError("training.output_dir must be set in YAML config or CLI")
    checkpoint = args.checkpoint or (args.output_dir / "best.pt")
    output_json = args.output_json or (args.output_dir / "eval_metrics.json")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    train_ids = _read_ids(args.split_dir / "train_ids.txt")
    pos_weight, train_pos, train_neg = _class_weight_from_manifest(args.manifest, train_ids, args.max_pos_weight)

    model = ESMSiteClassifier(model_config).to(device)
    ckpt = _load_checkpoint(checkpoint, device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device),
        ignore_index=-100,
        label_smoothing=max(0.0, min(1.0, float(args.label_smoothing))),
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
        "splits": {},
    }
    for split in [x.strip() for x in str(args.splits).split(",") if x.strip()]:
        ids = _read_ids(args.split_dir / f"{split}_ids.txt")
        ids = _limit_ids(ids, args.max_samples)
        loader = _build_loader(
            args.esm_root,
            args.label_root,
            ids,
            args.sequence_feature_root,
            args.require_sequence_features,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            args.max_residues,
            args.eval_crop_mode,
            args.seed + 100,
            False,
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
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({"event": "evaluated", "output_json": str(output_json), "splits": results["splits"]}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
