#!/usr/bin/env python3
"""Smoke test ESM + sequence_v1 dataset/model wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from ppcbind.data import ESMProteinSiteDataset, collate_esm_site_features
from ppcbind.models import ESMSiteClassifier


def read_ids(path: Path, n: int) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        value = line.strip().lower()
        if value and not value.startswith("#"):
            ids.append(value)
        if len(ids) >= n:
            break
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_contact_site_esm_seqv1_mmseq30.yaml"))
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    split_dir = Path(data_cfg["split_dir"])
    ids = read_ids(split_dir / "train_ids.txt", args.n)
    dataset = ESMProteinSiteDataset(
        esm_root=data_cfg["esm_root"],
        label_root=data_cfg["label_root"],
        ids=ids,
        sequence_feature_root=data_cfg["sequence_feature_root"],
        require_sequence_features=bool(data_cfg.get("require_sequence_features", False)),
        max_residues=None,
        crop_mode="none",
        seed=0,
    )
    items = [dataset[i] for i in range(len(dataset))]
    batch = collate_esm_site_features(items)
    model = ESMSiteClassifier(model_cfg).eval()
    with torch.no_grad():
        out = model(
            esm_embeddings=batch["esm_embeddings"],
            seq_features=batch["seq_features"],
            protein_mask=batch["protein_mask"],
            chain_ids=batch["chain_ids"],
            chain_rel_pos=batch["chain_rel_pos"],
            protein_rel_pos=batch["protein_rel_pos"],
        )
    summary = {
        "ids": batch["pdb_id"],
        "esm_embeddings": tuple(batch["esm_embeddings"].shape),
        "seq_features": tuple(batch["seq_features"].shape),
        "labels": tuple(batch["labels"].shape),
        "logits": tuple(out["logits"].shape),
        "seq_feature_dim": int(batch["seq_features"].shape[-1]),
        "d_seq_config": int(model_cfg["d_seq"]),
        "finite_logits": bool(torch.isfinite(out["logits"]).all()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if summary["seq_feature_dim"] != summary["d_seq_config"]:
        raise ValueError("seq feature dim does not match model d_seq")
    if not summary["finite_logits"]:
        raise ValueError("nonfinite logits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
