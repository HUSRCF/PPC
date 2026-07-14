#!/usr/bin/env python3
"""Export aligned split logits from ESMC-primary and ESM2-secondary models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = REPO_ROOT / "scripts" / "training"
SRC_DIR = REPO_ROOT / "src"
for path in (TRAINING_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_esm_site import _build_loader, _load_yaml, _move_batch, _read_ids  # noqa: E402
from ppcbind.models import ESMSiteClassifier  # noqa: E402


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return checkpoint


def _model_config(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        model_config = config.get("model_config")
        if isinstance(model_config, dict):
            return dict(model_config)
        yaml_config = config.get("yaml_config", {})
        if isinstance(yaml_config, dict) and isinstance(yaml_config.get("model"), dict):
            return dict(yaml_config["model"])
    raise KeyError(f"Checkpoint does not contain model_config: {path}")


def _load_model(path: Path, device: torch.device) -> tuple[ESMSiteClassifier, dict[str, Any], dict[str, Any]]:
    checkpoint = _load_checkpoint(path, device)
    config = _model_config(checkpoint, path)
    model = ESMSiteClassifier(config).to(device)
    state = checkpoint.get("model_state", checkpoint.get("model"))
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint does not contain model/model_state: {path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint


def _manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["seq_id"].strip().lower(): row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"Empty chain manifest: {path}")
    return rows


def _path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-config", required=True, type=Path)
    parser.add_argument("--primary-checkpoint", required=True, type=Path)
    parser.add_argument("--secondary-checkpoint", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    args = parser.parse_args()

    config = _load_yaml(args.dual_config)
    data = config.get("data", {})
    training = config.get("training", {})
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("dual-config requires data and training mappings")
    split_dir = Path(data["split_dir"])
    chain_manifest_path = Path(data["chain_filter_manifest"])
    split_ids = _read_ids(split_dir / f"{args.split}_ids.txt", lowercase=False)
    normalized_split_ids = [seq_id.strip().lower() for seq_id in split_ids]
    if len(normalized_split_ids) != len(set(normalized_split_ids)):
        raise ValueError(f"Duplicate chain IDs in {args.split} split")
    manifest = _manifest_rows(chain_manifest_path)
    missing_manifest = sorted(set(normalized_split_ids) - set(manifest))
    if missing_manifest:
        raise KeyError(f"{args.split} IDs missing from chain manifest: {missing_manifest[:20]}")

    device = torch.device(args.device)
    primary, primary_config, primary_checkpoint = _load_model(args.primary_checkpoint, device)
    secondary, secondary_config, secondary_checkpoint = _load_model(args.secondary_checkpoint, device)
    if bool(primary_config.get("use_prottrans_embeddings")) or bool(secondary_config.get("use_prottrans_embeddings")):
        raise ValueError("Dual-logit export expects two single-encoder base checkpoints")

    loader = _build_loader(
        Path(data["esm_root"]),
        Path(data["label_root"]),
        split_ids,
        _path(data.get("sequence_feature_root")),
        _path(data.get("primary_embedding_root")),
        _path(data.get("contact_graph_root")),
        _path(data.get("aux_contact_graph_root")),
        _path(data.get("prottrans_embedding_root")),
        bool(data.get("require_sequence_features", False)),
        bool(data.get("require_primary_embeddings", False)),
        True,
        args.batch_size,
        args.num_workers,
        True,
        4,
        args.num_workers > 0,
        8,
        int(training.get("max_residues", 0) or 0),
        "none",
        int(training.get("seed", 42)) + 100,
        False,
        batching="token_budget",
        max_batch_tokens=args.max_batch_tokens,
        length_bucket_size=1024,
        preload=False,
        strict_ids=True,
        require_labels=True,
        strict_label_metadata=False,
        strict_sequence_feature_metadata=False,
        require_contact_graph=bool(data.get("require_contact_graph", False)),
        require_aux_contact_graph=bool(data.get("require_aux_contact_graph", False)),
        chain_filter_manifest=chain_manifest_path,
    )

    labels_chunks: list[np.ndarray] = []
    primary_chunks: list[np.ndarray] = []
    secondary_chunks: list[np.ndarray] = []
    chain_index_chunks: list[np.ndarray] = []
    residue_position_chunks: list[np.ndarray] = []
    chain_ids: list[str] = []
    source_pdb_ids: list[str] = []
    component_ids: list[str] = []

    expected_primary_dim = int(primary_config["d_esm"])
    expected_secondary_dim = int(secondary_config["d_esm"])
    seen: set[str] = set()
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            secondary_embeddings = inputs.pop("prottrans_embeddings", None)
            if secondary_embeddings is None:
                raise KeyError("Dual-input loader did not return prottrans_embeddings")
            if inputs["esm_embeddings"].shape[-1] != expected_primary_dim:
                raise ValueError(
                    f"Primary embedding dim {inputs['esm_embeddings'].shape[-1]} != checkpoint d_esm={expected_primary_dim}"
                )
            if secondary_embeddings.shape[-1] != expected_secondary_dim:
                raise ValueError(
                    f"Secondary embedding dim {secondary_embeddings.shape[-1]} != checkpoint d_esm={expected_secondary_dim}"
                )
            primary_logits = primary(**inputs)["logits"]
            secondary_inputs = dict(inputs)
            secondary_inputs["esm_embeddings"] = secondary_embeddings
            secondary_logits = secondary(**secondary_inputs)["logits"]
            primary_log_odds = (primary_logits[..., 1] - primary_logits[..., 0]).float().cpu()
            secondary_log_odds = (secondary_logits[..., 1] - secondary_logits[..., 0]).float().cpu()
            labels_cpu = labels.cpu()
            mask_cpu = batch["protein_mask"].bool().cpu() & (labels_cpu >= 0)

            for item_index, raw_id in enumerate(batch["pdb_id"]):
                seq_id = str(raw_id).strip().lower()
                if seq_id in seen:
                    raise ValueError(f"Duplicate {args.split} chain from loader: {seq_id}")
                row = manifest.get(seq_id)
                if row is None:
                    raise KeyError(f"Loader chain missing from manifest: {seq_id}")
                if row.get("split", "").strip().lower() != args.split:
                    raise ValueError(f"Wrong-split chain in {args.split} export: {seq_id} split={row.get('split')}")
                valid = mask_cpu[item_index]
                n_residues = int(valid.sum().item())
                if n_residues <= 0:
                    raise ValueError(f"No labeled residues for {args.split} chain: {seq_id}")
                chain_index = len(chain_ids)
                labels_chunks.append(labels_cpu[item_index, valid].numpy().astype(np.int8, copy=False))
                primary_chunks.append(primary_log_odds[item_index, valid].numpy().astype(np.float32, copy=False))
                secondary_chunks.append(secondary_log_odds[item_index, valid].numpy().astype(np.float32, copy=False))
                chain_index_chunks.append(np.full(n_residues, chain_index, dtype=np.int32))
                residue_position_chunks.append(
                    (batch["chain_pos"][item_index, valid].cpu().numpy() + 1).astype(np.int32, copy=False)
                )
                chain_ids.append(seq_id)
                source_pdb_ids.append(row["pdb_id"].strip().lower())
                component_ids.append(row["component_id"].strip())
                seen.add(seq_id)

    if seen != set(normalized_split_ids):
        raise ValueError(f"{args.split} coverage mismatch: expected={len(split_ids)} observed={len(seen)}")
    payload = {
        "labels": np.concatenate(labels_chunks),
        "primary_logit": np.concatenate(primary_chunks),
        "secondary_logit": np.concatenate(secondary_chunks),
        "chain_index": np.concatenate(chain_index_chunks),
        "residue_position": np.concatenate(residue_position_chunks),
        "chain_ids": np.asarray(chain_ids),
        "source_pdb_ids": np.asarray(source_pdb_ids),
        "component_ids": np.asarray(component_ids),
    }
    if not all(np.isfinite(payload[key]).all() for key in ("primary_logit", "secondary_logit")):
        raise ValueError("Non-finite logits in exported payload")
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **payload)
    summary = {
        "scope": args.split,
        "dual_config": str(args.dual_config),
        "primary_checkpoint": str(args.primary_checkpoint),
        "primary_epoch": primary_checkpoint.get("epoch"),
        "primary_d_esm": expected_primary_dim,
        "secondary_checkpoint": str(args.secondary_checkpoint),
        "secondary_epoch": secondary_checkpoint.get("epoch"),
        "secondary_d_esm": expected_secondary_dim,
        "out_npz": str(args.out_npz.resolve()),
        "out_npz_sha256": _sha256(args.out_npz),
        "chains": len(chain_ids),
        "source_pdbs": len(set(source_pdb_ids)),
        "components": len(set(component_ids)),
        "residues": int(payload["labels"].size),
        "positive_residues": int(payload["labels"].sum()),
        "coverage_exact": True,
        "test_accessed": args.split == "test",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
