#!/usr/bin/env python3
"""Smoke-test compact chain loading and ESM-C fusion modes."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import torch

from ppcbind.data import ESMProteinSiteDataset
from ppcbind.models import ESMSiteClassifier


def _write_manifest(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seq_id", "pdb_id", "chain_id", "n_residues"))
        writer.writeheader()
        writer.writerow({"seq_id": "p1__B", "pdb_id": "p1", "chain_id": "B", "n_residues": 2})


def _test_loader(root: Path) -> None:
    esm_root = root / "legacy"
    compact_root = root / "compact"
    embedding_root = root / "chains"
    label_root = root / "labels"
    for path in (esm_root / "p1", compact_root / "p1", embedding_root, label_root / "p1"):
        path.mkdir(parents=True, exist_ok=True)
    embeddings = torch.arange(30, dtype=torch.float32).view(5, 6)
    source = {
        "embeddings": embeddings,
        "chain_ids": ["A", "A", "A", "B", "B"],
        "residue_names_1": list("ACDEF"),
        "feature_n_residues": 5,
        "contact_edge_index": torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]]),
        "contact_edge_scores": torch.tensor([0.2, 0.2, 0.9, 0.9]),
    }
    torch.save(source, esm_root / "p1" / "p1_esm2.pt")
    torch.save(
        {key: value for key, value in source.items() if key not in {"embeddings", "contact_edge_index", "contact_edge_scores"}},
        compact_root / "p1" / "p1_esm2.pt",
    )
    torch.save(
        {
            "embeddings": embeddings[3:].clone(),
            "sequence": "EF",
            "n_residues": 2,
            "contact_edge_index": torch.tensor([[0, 1], [1, 0]]),
            "contact_edge_scores": torch.tensor([0.9, 0.9]),
        },
        embedding_root / "p1__B.pt",
    )
    torch.save({"labels": torch.tensor([0, 1, 0, 1, 1])}, label_root / "p1" / "p1_labels.pt")
    manifest = root / "chains.csv"
    _write_manifest(manifest)

    common = dict(
        label_root=label_root,
        ids=["p1__B"],
        chain_filter_manifest=manifest,
        require_labels=True,
        require_contact_graph=True,
        strict_ids=True,
    )
    legacy = ESMProteinSiteDataset(esm_root=esm_root, **common)[0]
    compact = ESMProteinSiteDataset(
        esm_root=compact_root,
        primary_embedding_root=embedding_root,
        contact_graph_root=embedding_root,
        require_primary_embeddings=True,
        payload_cache_size=2,
        **common,
    )[0]
    for key in ("esm_embeddings", "labels", "contact_edge_index", "contact_edge_scores", "chain_ids"):
        if not torch.equal(legacy[key], compact[key]):
            raise AssertionError(f"compact loader changed {key}")


def _run_model(config: dict, esm_dim: int, secondary_dim: int | None = None) -> ESMSiteClassifier:
    model = ESMSiteClassifier(config).eval()
    kwargs = {
        "esm_embeddings": torch.randn(2, 7, esm_dim),
        "protein_mask": torch.ones(2, 7, dtype=torch.bool),
    }
    if secondary_dim is not None:
        kwargs["prottrans_embeddings"] = torch.randn(2, 7, secondary_dim)
    with torch.no_grad():
        output = model(**kwargs)["logits"]
    if output.shape != (2, 7, 2):
        raise AssertionError(f"unexpected output shape {tuple(output.shape)}")
    return model


def _test_models() -> None:
    base = {
        "d_esm": 12,
        "d_model": 8,
        "d_hidden": 16,
        "use_seq_features": False,
        "use_chain_embedding": False,
        "use_position_features": False,
        "use_global_context": False,
        "esm_layer_count": 3,
    }
    _run_model({**base, "esm_layer_fusion": "last"}, 12)
    _run_model({**base, "esm_layer_fusion": "scalar_mix"}, 12)
    model = _run_model(
        {
            **base,
            "esm_layer_fusion": "concat",
            "use_prottrans_embeddings": True,
            "d_prottrans": 9,
            "prottrans_fusion_mode": "projected_concat",
        },
        12,
        9,
    )
    if model.esm_proj[0].out_features != 4 or model.prottrans_proj[0].out_features != 4:
        raise AssertionError("projected_concat did not split d_model evenly")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ppc_esmc_smoke_") as tmp:
        _test_loader(Path(tmp))
    _test_models()
    print("ESM-C matrix smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
