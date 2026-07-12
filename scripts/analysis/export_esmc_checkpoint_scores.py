#!/usr/bin/env python3
"""Export aligned per-chain residue probabilities from an ESM-site checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_training_module(project_root: Path) -> Any:
    sys.path.insert(0, str(project_root / "src"))
    script = project_root / "scripts" / "training" / "train_esm_site.py"
    spec = importlib.util.spec_from_file_location("ppc_train_esm_site_export", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--payload-cache-size", type=int, default=8)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    training = load_training_module(project_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint must contain a model state dictionary")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model_config"), dict):
        raise ValueError("Checkpoint has no resolved config/model_config")
    model_config = config["model_config"]

    esm_root = resolve(project_root, config["esm_root"])
    label_root = resolve(project_root, config["label_root"])
    manifest = resolve(project_root, config["manifest"])
    split_dir = resolve(project_root, config["split_dir"])
    chain_filter_manifest = resolve(project_root, config.get("chain_filter_manifest"))
    sequence_feature_root = resolve(project_root, config.get("sequence_feature_root"))
    primary_embedding_root = resolve(project_root, config.get("primary_embedding_root"))
    prottrans_embedding_root = resolve(project_root, config.get("prottrans_embedding_root"))
    contact_graph_root = resolve(project_root, config.get("contact_graph_root"))
    aux_contact_graph_root = resolve(project_root, config.get("aux_contact_graph_root"))
    if esm_root is None or label_root is None or manifest is None or split_dir is None:
        raise ValueError("Checkpoint config is missing a required data path")

    chain_filtered = chain_filter_manifest is not None
    split_ids = training._read_ids(
        training._split_ids_path(split_dir, args.split, chain_filtered),
        lowercase=not chain_filtered,
    )
    lengths_by_id = training._lengths_from_manifest(manifest)
    loader = training._build_loader(
        esm_root,
        label_root,
        split_ids,
        sequence_feature_root,
        primary_embedding_root,
        contact_graph_root,
        aux_contact_graph_root,
        prottrans_embedding_root,
        bool(config.get("require_sequence_features", False)),
        bool(config.get("require_primary_embeddings", False)),
        bool(config.get("require_prottrans_embeddings", False)),
        args.batch_size,
        args.num_workers,
        True,
        args.prefetch_factor,
        True,
        args.payload_cache_size,
        int(config.get("max_residues", 0)),
        str(config.get("eval_crop_mode", "none")),
        int(config.get("seed", 42)) + (1 if args.split == "val" else 2),
        False,
        batching=str(config.get("batching", "token_budget")),
        max_batch_tokens=args.max_batch_tokens,
        length_bucket_size=int(config.get("length_bucket_size", 1024)),
        lengths_by_id=lengths_by_id,
        preload=False,
        strict_ids=True,
        require_labels=True,
        strict_label_metadata=False,
        strict_sequence_feature_metadata=False,
        require_contact_graph=bool(model_config.get("use_contact_graph")),
        require_aux_contact_graph=bool(config.get("require_aux_contact_graph", False)),
        chain_filter_manifest=chain_filter_manifest,
    )

    device = torch.device(args.device)
    model = training.ESMSiteClassifier(model_config).to(device)
    state = model.load_state_dict(checkpoint["model"], strict=True)
    if state.missing_keys or state.unexpected_keys:
        raise ValueError(f"Strict state load failed: {state}")
    model.eval()

    by_chain: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            inputs, labels = training._move_batch(batch, device)
            logits = model(**inputs)["logits"]
            if not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Non-finite logits in batch {batch_index}")
            scores = torch.softmax(logits.float(), dim=-1)[..., 1].cpu()
            labels = labels.cpu()
            for index, seq_id_value in enumerate(batch["pdb_id"]):
                seq_id = str(seq_id_value)
                if seq_id in by_chain:
                    raise ValueError(f"Duplicate chain emitted: {seq_id}")
                valid = labels[index] != -100
                chain_labels = labels[index][valid].numpy().astype(np.uint8, copy=True)
                chain_scores = scores[index][valid].numpy().astype(np.float32, copy=True)
                by_chain[seq_id] = (chain_labels, chain_scores)
            if batch_index % 20 == 0:
                print(json.dumps({"event": "progress", "batch": batch_index, "batches": len(loader), "chains": len(by_chain)}), flush=True)

    if set(by_chain) != set(split_ids):
        raise ValueError(f"Coverage mismatch: emitted={len(by_chain)} expected={len(split_ids)}")
    ordered_ids = sorted(by_chain)
    lengths = np.asarray([len(by_chain[seq_id][0]) for seq_id in ordered_ids], dtype=np.int32)
    offsets = np.empty(len(ordered_ids) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    labels_flat = np.concatenate([by_chain[seq_id][0] for seq_id in ordered_ids])
    scores_flat = np.concatenate([by_chain[seq_id][1] for seq_id in ordered_ids])
    max_id_len = max(len(seq_id) for seq_id in ordered_ids)
    seq_ids = np.asarray(ordered_ids, dtype=f"U{max_id_len}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        seq_ids=seq_ids,
        lengths=lengths,
        offsets=offsets,
        labels=labels_flat,
        scores=scores_flat,
    )
    summary = {
        "method": args.method,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "output_npz": str(args.output_npz),
        "output_sha256": sha256_file(args.output_npz),
        "n_chains": len(ordered_ids),
        "n_residues": int(labels_flat.size),
        "n_positive": int(labels_flat.sum()),
        "score_min": float(scores_flat.min()),
        "score_max": float(scores_flat.max()),
        "strict_state_load": True,
        "offline_metadata_validation_assumed": True,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
