#!/usr/bin/env python3
"""Export ESM-site checkpoint predictions for chains listed in a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = REPO_ROOT / "scripts" / "training"
SRC_DIR = REPO_ROOT / "src"
for path in (TRAINING_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_esm_site import _apply_config, _build_loader, _load_yaml, _move_batch  # noqa: E402
from ppcbind.models import ESMSiteClassifier  # noqa: E402


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_seed_config(seed_dir: Path, explicit_config: Path | None = None) -> dict:
    if explicit_config is not None:
        if explicit_config.suffix.lower() == ".json":
            with explicit_config.open() as handle:
                data = json.load(handle)
            return data.get("yaml_config", data)
        return _load_yaml(explicit_config)

    yaml_path = seed_dir / "config.yaml"
    if yaml_path.exists():
        return _load_yaml(yaml_path)

    json_path = seed_dir / "config.json"
    if json_path.exists():
        with json_path.open() as handle:
            data = json.load(handle)
        return data.get("yaml_config", data)

    raise FileNotFoundError(f"missing config.yaml or config.json under {seed_dir}")


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


def _load_chain_manifest(path: Path) -> tuple[dict[str, dict[str, dict[str, object]]], list[str]]:
    by_pdb: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    seq_ids: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = row["pdb_id"].strip().lower()
            chain_id = row["chain_id"].strip()
            seq_id = row["seq_id"].strip().lower()
            by_pdb[pdb_id][chain_id] = {
                "seq_id": seq_id,
                "n_residues": int(row["n_residues"]),
                "residue_indices_json": row.get("residue_indices_json", ""),
            }
            seq_ids.append(seq_id)
    return by_pdb, seq_ids


def _load_label_chain_order(label_root: Path, pdb_id: str) -> tuple[dict[int, str], dict[str, int]]:
    path = label_root / pdb_id / f"{pdb_id}_labels.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    mapping: dict[str, int] = {}
    reverse: dict[int, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for raw in data["chain_ids"]:
        chain = str(raw)
        if chain not in mapping:
            idx = len(mapping)
            mapping[chain] = idx
            reverse[idx] = chain
        counts[chain] += 1
    return reverse, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--esm-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sequence-feature-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = _load_seed_config(args.seed_dir, args.config)
    cfg_args = _default_args()
    model_config = _apply_config(cfg_args, config)
    if args.esm_root is not None:
        cfg_args.esm_root = args.esm_root
    if args.label_root is not None:
        cfg_args.label_root = args.label_root
    if args.manifest is not None:
        cfg_args.manifest = args.manifest
    if args.sequence_feature_root is not None:
        cfg_args.sequence_feature_root = args.sequence_feature_root

    chain_manifest, expected_seq_ids = _load_chain_manifest(args.chain_manifest)
    ids = sorted(chain_manifest)
    device = torch.device(args.device)
    require_contact_graph = (
        bool(model_config.get("use_contact_graph"))
        if cfg_args.require_contact_graph is None
        else bool(cfg_args.require_contact_graph)
    )
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
        strict_ids=True,
        require_labels=True,
        strict_label_metadata=True,
        require_contact_graph=require_contact_graph,
        require_aux_contact_graph=cfg_args.require_aux_contact_graph,
    )

    model = ESMSiteClassifier(model_config).to(device)
    ckpt = _load_checkpoint(args.seed_dir / "best.pt", device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    covered: set[str] = set()
    skipped_local_chains: list[str] = []
    length_mismatches: dict[str, dict[str, int]] = {}
    with args.out_tsv.open("w", newline="") as handle, torch.no_grad():
        writer = csv.DictWriter(handle, fieldnames=["seq_id", "position", "residue", "score"], delimiter="\t")
        writer.writeheader()
        for batch in loader:
            inputs, _labels = _move_batch(batch, device)
            probs = torch.softmax(model(**inputs)["logits"].float(), dim=-1)[..., 1].cpu()
            valid_mask = batch["protein_mask"].cpu().bool()
            chain_ids = batch["chain_ids"].cpu().long()
            chain_pos = batch["chain_pos"].cpu().long()
            for item_idx, pdb_id_value in enumerate(batch["pdb_id"]):
                pdb_id = str(pdb_id_value).lower()
                local_to_chain, chain_lengths = _load_label_chain_order(cfg_args.label_root, pdb_id)
                allowed = chain_manifest[pdb_id]
                observed_counts: dict[str, int] = defaultdict(int)
                valid_indices = torch.nonzero(valid_mask[item_idx], as_tuple=False).flatten().tolist()
                for pos_idx in valid_indices:
                    local_id = int(chain_ids[item_idx, pos_idx].item())
                    chain = local_to_chain.get(local_id)
                    if chain is None or chain not in allowed:
                        skipped_local_chains.append(f"{pdb_id}:{local_id}:{chain}")
                        continue
                    seq_id = str(allowed[chain]["seq_id"])
                    position = int(chain_pos[item_idx, pos_idx].item()) + 1
                    writer.writerow(
                        {
                            "seq_id": seq_id,
                            "position": position,
                            "residue": "",
                            "score": float(probs[item_idx, pos_idx].item()),
                        }
                    )
                    rows_written += 1
                    covered.add(seq_id)
                    observed_counts[seq_id] += 1
                for chain, meta in allowed.items():
                    seq_id = str(meta["seq_id"])
                    expected = int(meta["n_residues"])
                    observed = observed_counts.get(seq_id, 0)
                    if observed != expected:
                        length_mismatches[seq_id] = {"expected": expected, "observed": observed}

    expected = set(expected_seq_ids)
    summary = {
        "seed_dir": str(args.seed_dir),
        "checkpoint": str(args.seed_dir / "best.pt"),
        "chain_manifest": str(args.chain_manifest),
        "out_tsv": str(args.out_tsv),
        "pdb_ids": len(ids),
        "expected_chains": len(expected),
        "covered_chains": len(covered),
        "missing_chains": len(expected - covered),
        "missing_examples": sorted(expected - covered)[:50],
        "rows_written": rows_written,
        "length_mismatch_count": len(length_mismatches),
        "length_mismatch_examples": dict(list(sorted(length_mismatches.items()))[:20]),
        "skipped_local_chain_examples": skipped_local_chains[:50],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
