#!/usr/bin/env python3
"""Materialize exact-sequence chain embeddings and compact PDB metadata.

The chain-filtered loader samples individual chains while legacy ESM payloads
store every chain of a PDB in one large tensor.  This utility reads each PDB
once, writes one embedding/contact payload per unique sequence, and creates
lightweight per-chain aliases.  The compact metadata root retains only the
residue indexing needed to slice labels, so training no longer reloads a full
PDB embedding for every chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


METADATA_KEYS = (
    "chain_id",
    "chain_ids",
    "residue_index",
    "residue_indices",
    "insertion_code",
    "insertion_codes",
    "residue_name_1",
    "residue_names_1",
    "residue_name_3",
    "residue_names_3",
    "sequence",
    "chain_sequences",
    "chain_boundaries",
    "chain_window_info",
    "feature_n_residues",
    "pdb_id",
    "model_name",
    "repr_layer",
    "repr_layers",
    "embedding_combine",
    "embedding_dim",
    "contact_source",
    "contact_top_k",
    "contact_min_score",
    "contact_min_seq_sep",
    "contact_bidirectional",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict payload: {path}")
    return value


def _is_selected(row: dict[str, str]) -> bool:
    return str(row.get("is_selected", "")).strip().lower() in {"1", "true", "yes", "y"}


def _source_path(root: Path, pdb_id: str) -> Path:
    candidates = (
        root / pdb_id / f"{pdb_id}_esm2.pt",
        root / pdb_id / f"{pdb_id}_protein.pt",
        root / f"{pdb_id}_esm2.pt",
        root / f"{pdb_id}_protein.pt",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"{pdb_id}: source ESM payload not found under {root}")
    return path


def _local_contacts(data: dict[str, Any], start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.as_tensor(data.get("contact_edge_index", torch.empty((2, 0))), dtype=torch.long)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Invalid contact_edge_index shape: {tuple(edge_index.shape)}")
    scores = torch.as_tensor(
        data.get("contact_edge_scores", torch.ones(edge_index.shape[1])), dtype=torch.float32
    ).flatten()
    if scores.shape[0] != edge_index.shape[1]:
        raise ValueError(f"Contact score count {scores.shape[0]} != edge count {edge_index.shape[1]}")
    keep = (
        (edge_index[0] >= start)
        & (edge_index[0] < stop)
        & (edge_index[1] >= start)
        & (edge_index[1] < stop)
    )
    return (edge_index[:, keep] - start).contiguous(), scores[keep].contiguous()


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _ensure_alias(target: Path, alias: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(target, alias.parent))
    if alias.is_symlink() and Path(os.readlink(alias)) == relative:
        return
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(f"Alias exists with a different target: {alias}")
    alias.symlink_to(relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--source-esm-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    rows_by_pdb: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.chain_manifest.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if _is_selected(row):
                rows_by_pdb[row["pdb_id"].strip().lower()].append(row)
    if not rows_by_pdb:
        raise ValueError(f"No selected chains in {args.chain_manifest}")

    compact_root = args.output_root / "compact_pdb"
    unique_root = args.output_root / "unique"
    alias_root = args.output_root / "by_chain"
    n_unique_written = 0
    n_aliases = 0
    n_chains = 0
    total_pdb = len(rows_by_pdb)

    for pdb_idx, pdb_id in enumerate(sorted(rows_by_pdb), 1):
        source_path = _source_path(args.source_esm_root, pdb_id)
        data = _load(source_path)
        embeddings = torch.as_tensor(data["embeddings"])
        residue_names = list(data.get("residue_names_1", data.get("residue_name_1", ())))
        if len(residue_names) != embeddings.shape[0]:
            raise ValueError(f"{pdb_id}: residue metadata length does not match embeddings")

        compact_path = compact_root / pdb_id / f"{pdb_id}_esm2.pt"
        if not compact_path.exists():
            compact = {key: data[key] for key in METADATA_KEYS if key in data}
            compact["feature_n_residues"] = int(embeddings.shape[0])
            compact["source_feature_path"] = str(source_path)
            compact["cache_format"] = "compact_pdb_metadata_v1"
            _atomic_save(compact, compact_path)

        for row in rows_by_pdb[pdb_id]:
            start = int(row["first_row"])
            stop = int(row["last_row"]) + 1
            sequence = row["sequence"].strip()
            seq_id = row["selected_seq_id"].strip()
            unique_id = row["unique_id"].strip()
            if not seq_id or not unique_id:
                raise ValueError(f"{pdb_id}: selected row lacks selected_seq_id/unique_id")
            observed = "".join(str(value) for value in residue_names[start:stop])
            if observed != sequence:
                raise ValueError(f"{seq_id}: manifest sequence does not match source residue rows")
            digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            expected_digest = row.get("seq_sha256", "").strip()
            if expected_digest and digest != expected_digest:
                raise ValueError(f"{seq_id}: sequence SHA256 mismatch")

            unique_path = unique_root / digest[:2] / f"{unique_id}.pt"
            if not unique_path.exists():
                contact_edge_index, contact_edge_scores = _local_contacts(data, start, stop)
                payload = {
                    "embeddings": embeddings[start:stop].clone().contiguous(),
                    "sequence": sequence,
                    "sequence_sha256": digest,
                    "unique_id": unique_id,
                    "n_residues": len(sequence),
                    "contact_edge_index": contact_edge_index,
                    "contact_edge_scores": contact_edge_scores,
                    "source_pdb_id": pdb_id,
                    "source_chain_id": row["chain_id"],
                    "source_feature_path": str(source_path),
                    "model_name": data.get("model_name"),
                    "repr_layers": data.get("repr_layers"),
                    "embedding_combine": data.get("embedding_combine"),
                    "contact_source": data.get("contact_source"),
                    "contact_top_k": data.get("contact_top_k"),
                    "contact_min_score": data.get("contact_min_score"),
                    "contact_min_seq_sep": data.get("contact_min_seq_sep"),
                    "contact_bidirectional": data.get("contact_bidirectional"),
                    "cache_format": "exact_sequence_chain_embedding_v1",
                }
                _atomic_save(payload, unique_path)
                n_unique_written += 1
            _ensure_alias(unique_path, alias_root / f"{seq_id}.pt")
            n_aliases += 1
            n_chains += 1

        if pdb_idx % max(1, args.progress_every) == 0 or pdb_idx == total_pdb:
            print(
                json.dumps(
                    {
                        "pdb_done": pdb_idx,
                        "pdb_total": total_pdb,
                        "chains_done": n_chains,
                        "unique_written": n_unique_written,
                    }
                ),
                flush=True,
            )

    summary = {
        "chain_manifest": str(args.chain_manifest),
        "source_esm_root": str(args.source_esm_root),
        "compact_pdb_root": str(compact_root),
        "unique_root": str(unique_root),
        "by_chain_root": str(alias_root),
        "n_pdb": total_pdb,
        "n_chains": n_chains,
        "n_aliases": n_aliases,
        "n_unique_written": n_unique_written,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
