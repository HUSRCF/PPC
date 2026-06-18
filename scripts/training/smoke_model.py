#!/usr/bin/env python3
"""Smoke-test PPC protein-site model forward pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ppcbind.data import ProteinFeatureDataset, collate_protein_features
from ppcbind.models.protein_site_gvp import ProteinSiteGVP


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", default="features/protein_v4/pt", type=Path)
    parser.add_argument("--esm-root", default=None, type=Path)
    parser.add_argument("--label-root", default=None, type=Path)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-esm", action="store_true")
    parser.add_argument("--d-esm", type=int, default=1280)
    args = parser.parse_args()

    dataset = ProteinFeatureDataset(
        features_root=args.features_root,
        esm_root=args.esm_root,
        label_root=args.label_root,
    )
    n = min(args.max_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(n))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_protein_features,
    )
    batch = next(iter(loader))

    use_esm = args.use_esm and "esm_embeddings" in batch
    config = {"use_esm": use_esm, "d_esm": args.d_esm}
    model = ProteinSiteGVP(config).to(args.device).eval()
    tensor_keys = [
        "protein_physchem",
        "protein_spatial_scalar",
        "protein_spatial_vector",
        "protein_backbone_vector",
        "protein_coords",
        "protein_mask",
        "chain_ids",
    ]
    inputs = {key: batch[key].to(args.device) for key in tensor_keys}
    if use_esm:
        inputs["esm_embeddings"] = batch["esm_embeddings"].to(args.device)

    with torch.no_grad():
        out = model(**inputs)
    print(
        {
            "pdb_id": batch["pdb_id"],
            "use_esm": use_esm,
            "logits_shape": tuple(out["logits"].shape),
            "h_protein_shape": tuple(out["h_protein"].shape),
            "valid_residues": int(batch["protein_mask"].sum().item()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
