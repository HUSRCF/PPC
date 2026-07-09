#!/usr/bin/env python3
"""Build chain-level ESMFold structural sequence features.

This is the chain-filtered counterpart of the complex-level stitcher.  Each
sample id is a single chain (`pdb__chain`) from the split manifest, and its
predicted-structure features come from an exact-sequence ESMFold unique id.
No UniProt/TmAlphaFold bridge is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def feature_tensor(data: dict[str, Any], feature_set: str) -> tuple[torch.Tensor, list[str]]:
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    if feature_set in {"scalar", "scalar_physchem"}:
        value = torch.as_tensor(data["pred_spatial_scalar_features"], dtype=torch.float32)
        tensors.append(value)
        raw_names = list(data.get("spatial_scalar_feature_names", []))
        if len(raw_names) == value.shape[1]:
            names.extend([f"pred_scalar:{name}" for name in raw_names])
        else:
            names.extend([f"pred_scalar:{idx}" for idx in range(value.shape[1])])
    if feature_set == "scalar_physchem":
        value = torch.as_tensor(data["pred_physchem_features"], dtype=torch.float32)
        tensors.append(value)
        names.extend([f"pred_physchem:{idx}" for idx in range(value.shape[1])])
    if not tensors:
        raise ValueError(f"Unsupported feature_set={feature_set!r}")
    return torch.nan_to_num(torch.cat(tensors, dim=1), nan=0.0, posinf=0.0, neginf=0.0), names


def build_one(
    row: dict[str, str],
    pred_root: Path,
    output_root: Path,
    feature_set: str,
    add_availability: bool,
    overwrite: bool,
) -> dict[str, Any]:
    seq_id = norm_text(row["seq_id"])
    unique_id = norm_text(row["unique_id"])
    out_path = output_root / seq_id / f"{seq_id}_seq.pt"
    out = {
        "seq_id": seq_id,
        "pdb_id": norm_text(row.get("pdb_id") or row.get("prot")).lower(),
        "chain_id": norm_text(row.get("chain_id") or row.get("chain")),
        "unique_id": unique_id,
        "status": "ERROR",
        "message": "",
        "n_residues": "",
        "n_features": "",
        "output_path": str(out_path),
    }
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        out["status"] = "SKIP"
        return out
    pred_path = pred_root / "pt" / unique_id / f"{unique_id}_predcomplete.pt"
    if not pred_path.exists():
        out["message"] = f"missing unique feature: {pred_path}"
        return out
    try:
        pred = torch_load(pred_path)
        expected_seq = "".join(str(row.get("sequence") or "").split()).upper()
        pred_residues = [norm_text(x).upper() for x in list(pred["residue_names_1"])]
        if expected_seq and pred_residues != list(expected_seq):
            mismatch = next((i for i, (a, b) in enumerate(zip(pred_residues, expected_seq), start=1) if a != b), 0)
            raise ValueError(
                f"residue mismatch at {mismatch}: pred={''.join(pred_residues[:30])} "
                f"expected={expected_seq[:30]}"
            )
        features, names = feature_tensor(pred, feature_set)
        if features.shape[0] != len(pred_residues):
            raise ValueError(f"feature length {features.shape[0]} != residue length {len(pred_residues)}")
        if add_availability:
            availability = torch.ones((features.shape[0], 1), dtype=torch.float32)
            features = torch.cat([features, availability], dim=1)
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite chain sequence features")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.pt")
        torch.save(
            {
                "pdb_id": seq_id,
                "source_pdb_id": out["pdb_id"],
                "chain_id": out["chain_id"],
                "seq_id": seq_id,
                "unique_id": unique_id,
                "seq_features": features,
                "feature_names": names,
                "feature_names_with_availability": names + (["pred_struct_available"] if add_availability else []),
                "sequence": "".join(pred_residues),
                "residue_names_1": pred_residues,
                "chain_ids": [out["chain_id"]] * len(pred_residues),
                "pred_residue_indices": list(pred.get("residue_indices", [])),
                "pred_insertion_codes": list(pred.get("insertion_codes", [])),
                "uniprot_positions": list(pred.get("uniprot_positions", [])),
                "pred_struct_root": str(pred_root),
                "source_pdb_path": pred.get("source_pdb_path", ""),
                "feature_set": feature_set,
                "add_availability": add_availability,
                "mapping_method": "esmfold_exact_sequence_chain_filtered",
                "mapping_policy": (
                    "chain-filtered ESMFold exact-sequence features; no UniProt, "
                    "TmAlphaFold, AFDB, or PDBTM bridge"
                ),
            },
            tmp_path,
        )
        tmp_path.replace(out_path)
        out.update({"status": "OK", "n_residues": features.shape[0], "n_features": features.shape[1]})
    except Exception as exc:  # noqa: BLE001
        out["message"] = str(exc)[:1000]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("scalar", "scalar_physchem"), default="scalar")
    parser.add_argument("--add-availability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    with args.chain_map.open(newline="") as handle:
        rows_in = list(csv.DictReader(handle))
    if args.max > 0:
        rows_in = rows_in[: args.max]
    rows = [
        build_one(
            row,
            args.pred_root,
            args.output_root,
            args.feature_set,
            args.add_availability,
            args.overwrite,
        )
        for row in rows_in
    ]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seq_id", "pdb_id", "chain_id", "unique_id", "status", "message", "n_residues", "n_features", "output_path"]
    tmp_path = args.manifest.with_suffix(f".tmp.{os.getpid()}.csv")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(args.manifest)
    summary = {
        "chain_map": str(args.chain_map),
        "pred_root": str(args.pred_root),
        "output_root": str(args.output_root),
        "manifest": str(args.manifest),
        "n_total": len(rows),
        "status_counts": Counter(str(row["status"]) for row in rows),
        "n_features": sorted({int(row["n_features"]) for row in rows if row["status"] == "OK" and row["n_features"]}),
    }
    summary_path = args.manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=dict) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
