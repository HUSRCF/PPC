#!/usr/bin/env python3
"""Smoke-test compact chain loading and ESM-C fusion modes."""

from __future__ import annotations

import csv
import inspect
import tempfile
from pathlib import Path

import torch

from ppcbind.data import ESMProteinSiteDataset
from ppcbind.models import ESMSiteClassifier


def _assert_repo_model_import() -> None:
    relative_model = Path("src/ppcbind/models/esm_site.py")
    expected = next(
        (
            (parent / relative_model).resolve()
            for parent in Path(__file__).resolve().parents
            if (parent / relative_model).is_file()
        ),
        None,
    )
    if expected is None:
        raise AssertionError(f"cannot locate repository model {relative_model}")
    actual = Path(inspect.getfile(ESMSiteClassifier)).resolve()
    if actual != expected:
        raise AssertionError(f"ESMSiteClassifier imported from {actual}, expected {expected}")


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


def _test_dual_fusion_backward(base: dict, mode: str) -> ESMSiteClassifier:
    config = {
        **base,
        "esm_layer_fusion": "concat",
        "use_prottrans_embeddings": True,
        "d_prottrans": 9,
        "prottrans_fusion_mode": mode,
        "prottrans_interaction_rank": 4,
        "dropout": 0.0,
    }
    model = ESMSiteClassifier(config).train()
    esm = torch.randn(2, 7, 12, requires_grad=True)
    secondary = torch.randn(2, 7, 9, requires_grad=True)
    logits = model(
        esm_embeddings=esm,
        prottrans_embeddings=secondary,
        protein_mask=torch.ones(2, 7, dtype=torch.bool),
    )["logits"]
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"non-finite {mode} gradient in {name}")
    if esm.grad is None or secondary.grad is None:
        raise AssertionError(f"{mode} did not backpropagate into both embedding streams")
    return model


def _assert_strict_round_trip(
    model: ESMSiteClassifier,
    *,
    required_prefixes: tuple[str, ...],
    forbidden_prefixes: tuple[str, ...],
) -> None:
    state = model.state_dict()
    keys = tuple(state)
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in keys):
            raise AssertionError(f"missing expected state_dict prefix {prefix!r}")
    for prefix in forbidden_prefixes:
        if any(key.startswith(prefix) for key in keys):
            raise AssertionError(f"unexpected state_dict prefix {prefix!r}")
    clone = ESMSiteClassifier(model.config)
    incompatible = clone.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"strict state_dict round trip failed: {incompatible}")


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
    legacy = _test_dual_fusion_backward(base, "gated_residual")
    legacy_bias = legacy.prottrans_gate[1].bias
    if not torch.equal(legacy_bias, torch.full_like(legacy_bias, -2.0)):
        raise AssertionError("legacy gated_residual default gate bias changed")
    _assert_strict_round_trip(
        legacy,
        required_prefixes=("prottrans_gate.",),
        forbidden_prefixes=("prottrans_fusion_norm.", "prottrans_interaction."),
    )
    convex = _test_dual_fusion_backward(base, "convex_gate")
    gate_linear = convex.prottrans_gate[1]
    if torch.count_nonzero(gate_linear.weight).item() != 0:
        raise AssertionError("convex_gate must initialize as an exact equal-weight mixture")
    if torch.count_nonzero(gate_linear.bias).item() != 0:
        raise AssertionError("convex_gate default bias must initialize to zero")
    gate_input_dim = gate_linear.in_features
    gate = convex.prottrans_gate(torch.randn(2, 7, gate_input_dim))
    if not torch.equal(gate, torch.full_like(gate, 0.5)):
        raise AssertionError("convex_gate default must produce an exact 0.5 mixture")
    if convex.prottrans_fusion_norm is None:
        raise AssertionError("convex_gate is missing post-fusion normalization")
    _assert_strict_round_trip(
        convex,
        required_prefixes=("prottrans_gate.", "prottrans_fusion_norm."),
        forbidden_prefixes=("prottrans_interaction.",),
    )
    bilinear = _test_dual_fusion_backward(base, "symmetric_bilinear")
    if bilinear.prottrans_interaction is None or bilinear.prottrans_fusion_norm is None:
        raise AssertionError("symmetric_bilinear modules were not constructed")
    if bilinear.prottrans_interaction_scale is None:
        raise AssertionError("symmetric_bilinear interaction scale was not constructed")
    _assert_strict_round_trip(
        bilinear,
        required_prefixes=(
            "prottrans_fusion_norm.",
            "prottrans_interaction.",
            "prottrans_interaction_scale",
        ),
        forbidden_prefixes=("prottrans_gate.",),
    )


def main() -> int:
    _assert_repo_model_import()
    with tempfile.TemporaryDirectory(prefix="ppc_esmc_smoke_") as tmp:
        _test_loader(Path(tmp))
    _test_models()
    print("ESM-C matrix smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
