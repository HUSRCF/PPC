#!/usr/bin/env python3
"""Build complex-level predicted-structure contact graph payloads.

This converts per-chain `features/pred_struct_v1` predicted C-alpha contact
graphs into one graph-only `.pt` file per PDB complex, aligned to the ESM
residue order. It does not read experimental complex coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


FIELDNAMES = [
    "pdb_id",
    "status",
    "n_residues",
    "n_edges",
    "n_chains",
    "n_chains_ok",
    "n_chains_missing",
    "n_chains_mismatch",
    "mean_score",
    "mean_dist",
    "output_path",
    "error",
]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _esm_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_esm2"):
        return stem[: -len("_esm2")]
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _discover_esm_paths(esm_root: Path, ids: list[str] | None) -> list[Path]:
    if ids is None:
        return sorted(esm_root.glob("*/*_esm2.pt"))
    out: list[Path] = []
    for pdb_id in ids:
        pdb_id = pdb_id.lower()
        candidates = (
            esm_root / pdb_id / f"{pdb_id}_esm2.pt",
            esm_root / pdb_id / f"{pdb_id}_protein.pt",
            esm_root / f"{pdb_id}_esm2.pt",
            esm_root / f"{pdb_id}_protein.pt",
        )
        for path in candidates:
            if path.exists():
                out.append(path)
                break
    return out


def _read_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [
        line.strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _chain_segments(chain_ids: list[Any]) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    if not chain_ids:
        return segments
    start = 0
    current = str(chain_ids[0])
    for idx, value in enumerate(chain_ids[1:], 1):
        key = str(value)
        if key != current:
            segments.append((current, start, idx))
            current = key
            start = idx
    segments.append((current, start, len(chain_ids)))
    return segments


def _score_from_dist(dist: torch.Tensor, cutoff: float, mode: str) -> torch.Tensor:
    dist = torch.nan_to_num(dist.float(), nan=cutoff, posinf=cutoff, neginf=cutoff).clamp_min(1.0e-6)
    if mode == "exp":
        return torch.exp(-dist / max(cutoff, 1.0e-6)).float()
    if mode == "linear":
        return torch.clamp((float(cutoff) - dist) / max(float(cutoff), 1.0e-6), min=0.0, max=1.0).float()
    if mode == "inverse":
        return (1.0 / (1.0 + dist)).float()
    raise ValueError(f"Unsupported score mode: {mode}")


def _pred_path(pred_root: Path, pdb_id: str, chain_id: str) -> Path:
    return pred_root / pdb_id / f"{pdb_id}__{chain_id}_predstruct.pt"


def _process_one(
    esm_path: Path,
    pred_root: Path,
    output_root: Path,
    cutoff: float,
    score_mode: str,
    overwrite: bool,
    min_coverage: float,
) -> dict[str, Any]:
    pdb_id = _esm_id(esm_path).lower()
    row: dict[str, Any] = {key: "" for key in FIELDNAMES}
    row.update({"pdb_id": pdb_id, "status": "ERROR"})
    try:
        out_path = output_root / pdb_id / f"{pdb_id}_contact_graph.pt"
        if out_path.exists() and not overwrite:
            payload = _torch_load(out_path)
            row.update(
                {
                    "status": "SKIP",
                    "n_residues": int(payload.get("n_residues", 0)),
                    "n_edges": int(payload.get("contact_edge_index", torch.empty(2, 0)).shape[1]),
                    "output_path": str(out_path),
                }
            )
            return row

        esm = _torch_load(esm_path)
        residue_names = list(esm.get("residue_names_1", esm.get("residue_name_1", [])))
        chain_ids = list(esm.get("chain_ids", esm.get("chain_id", [""] * len(residue_names))))
        if len(residue_names) != len(chain_ids):
            raise ValueError(f"metadata length mismatch: residues={len(residue_names)} chains={len(chain_ids)}")
        n_res = len(residue_names)
        edge_indices: list[torch.Tensor] = []
        edge_scores: list[torch.Tensor] = []
        edge_dists: list[torch.Tensor] = []
        records: list[dict[str, Any]] = []
        counts = Counter()

        for chain_id, start, stop in _chain_segments(chain_ids):
            counts["chains"] += 1
            chain_len = stop - start
            path = _pred_path(pred_root, pdb_id, chain_id)
            if not path.exists():
                counts["missing"] += 1
                records.append({"chain_id": chain_id, "start": start, "stop": stop, "status": "MISSING", "path": str(path)})
                continue
            pred = _torch_load(path)
            pred_res = list(pred.get("residue_names_1", []))
            esm_res = residue_names[start:stop]
            if len(pred_res) != chain_len or pred_res != esm_res:
                counts["mismatch"] += 1
                records.append(
                    {
                        "chain_id": chain_id,
                        "start": start,
                        "stop": stop,
                        "status": "MISMATCH",
                        "path": str(path),
                        "pred_len": len(pred_res),
                        "esm_len": chain_len,
                        "first_mismatch": next((i for i, (a, b) in enumerate(zip(pred_res, esm_res)) if a != b), None),
                    }
                )
                continue
            edge_index = torch.as_tensor(pred.get("pred_contact_edge_index"), dtype=torch.long)
            edge_dist = torch.as_tensor(pred.get("pred_contact_edge_dist"), dtype=torch.float32).flatten()
            if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_dist.shape[0] != edge_index.shape[1]:
                counts["mismatch"] += 1
                records.append({"chain_id": chain_id, "start": start, "stop": stop, "status": "BAD_GRAPH", "path": str(path)})
                continue
            valid = (
                (edge_index[0] >= 0)
                & (edge_index[1] >= 0)
                & (edge_index[0] < chain_len)
                & (edge_index[1] < chain_len)
                & (edge_index[0] != edge_index[1])
                & torch.isfinite(edge_dist)
            )
            edge_index = edge_index[:, valid] + int(start)
            edge_dist = edge_dist[valid]
            if edge_index.numel() > 0:
                edge_indices.append(edge_index.contiguous())
                edge_dists.append(edge_dist.contiguous())
                edge_scores.append(_score_from_dist(edge_dist, cutoff=cutoff, mode=score_mode))
            counts["ok"] += 1
            records.append(
                {
                    "chain_id": chain_id,
                    "start": start,
                    "stop": stop,
                    "status": "OK",
                    "path": str(path),
                    "n_edges": int(edge_dist.numel()),
                    "source": pred.get("source", ""),
                    "uniprot_acc": pred.get("uniprot_acc", ""),
                    "uniprot_range": pred.get("uniprot_range", ""),
                }
            )

        if edge_indices:
            contact_edge_index = torch.cat(edge_indices, dim=1)
            pred_contact_edge_dist = torch.cat(edge_dists, dim=0)
            contact_edge_scores = torch.cat(edge_scores, dim=0)
        else:
            contact_edge_index = torch.empty((2, 0), dtype=torch.long)
            pred_contact_edge_dist = torch.empty((0,), dtype=torch.float32)
            contact_edge_scores = torch.empty((0,), dtype=torch.float32)

        coverage = float(counts["ok"]) / max(1, int(counts["chains"]))
        status = "OK" if counts["ok"] == counts["chains"] else ("PARTIAL" if coverage >= min_coverage and counts["ok"] > 0 else "FAIL")
        payload = {
            "pdb_id": pdb_id,
            "n_residues": n_res,
            "residue_names_1": residue_names,
            "chain_ids": chain_ids,
            "contact_edge_index": contact_edge_index,
            "contact_edge_scores": contact_edge_scores,
            "pred_contact_edge_dist": pred_contact_edge_dist,
            "pred_contact_cutoff": float(cutoff),
            "pred_contact_score_mode": score_mode,
            "contact_source": "predicted_structure_ca_distance",
            "pred_struct_root": str(pred_root),
            "chain_records": records,
            "coverage_chains": coverage,
            "status": status,
            "feature_semantics": "predicted-structure contact graph only; no experimental complex coordinates",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, out_path)
        row.update(
            {
                "status": status,
                "n_residues": n_res,
                "n_edges": int(contact_edge_index.shape[1]),
                "n_chains": int(counts["chains"]),
                "n_chains_ok": int(counts["ok"]),
                "n_chains_missing": int(counts["missing"]),
                "n_chains_mismatch": int(counts["mismatch"]),
                "mean_score": float(contact_edge_scores.mean().item()) if contact_edge_scores.numel() else math.nan,
                "mean_dist": float(pred_contact_edge_dist.mean().item()) if pred_contact_edge_dist.numel() else math.nan,
                "output_path": str(out_path),
            }
        )
        return row
    except Exception as exc:
        row["error"] = repr(exc)
        return row


def _write_manifest(rows: list[dict[str, Any]], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    os.replace(tmp, manifest)
    summary = {
        "n_total": len(rows),
        "statuses": dict(Counter(str(row.get("status", "")) for row in rows)),
        "n_edges_total": int(sum(int(row.get("n_edges") or 0) for row in rows)),
        "manifest": str(manifest),
    }
    summary_path = manifest.with_suffix(".summary.json")
    tmp_json = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_json, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-root", type=Path, default=Path("features/esm2_t33_650M_UR50D_mlc/pt"))
    parser.add_argument("--pred-root", type=Path, default=Path("features/pred_struct_v1/pt"))
    parser.add_argument("--output-root", type=Path, default=Path("features/pred_struct_contact_graph_v1/pt"))
    parser.add_argument("--manifest", type=Path, default=Path("features/pred_struct_contact_graph_v1/manifest.csv"))
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--score-mode", choices=["exp", "linear", "inverse"], default="exp")
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ids = _read_ids(args.ids)
    esm_paths = _discover_esm_paths(args.esm_root, ids)
    if not esm_paths:
        raise SystemExit(f"No ESM files found under {args.esm_root}")
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for path in esm_paths:
            rows.append(_process_one(path, args.pred_root, args.output_root, args.contact_cutoff, args.score_mode, args.overwrite, args.min_coverage))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_process_one, path, args.pred_root, args.output_root, args.contact_cutoff, args.score_mode, args.overwrite, args.min_coverage)
                for path in esm_paths
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["pdb_id"])
    _write_manifest(rows, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
