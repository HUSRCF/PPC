#!/usr/bin/env python3
"""Build complex-level sequence-feature files from predicted-structure features.

The ESM training dataset consumes one complex-level ``*_seq.pt`` file per PDB
ID. Predicted complete features are chain-level, so this adapter stitches chain
features back into the ESM residue order with strict chain/sequence checks.
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



def get_spatial_scalar_feature_names_v4() -> list[str]:
    names: list[str] = []
    names += [f"v2_sasa_{i}" for i in range(6)]
    names += [f"v2_surface_{i}" for i in range(14)]
    names += [f"v2_structural_geometry_{i}" for i in range(26)]
    names += [f"v2_enhanced_{i}" for i in range(12)]
    names += [f"local_density_raw_{i}" for i in range(4)]
    names += ["distance_to_center_raw"]
    names += ["surface_normal_magnitude"]
    names += [f"charge_density_raw_{i}" for i in range(4)]
    names += [f"hydrophobicity_{i}" for i in range(4)]
    names += ["phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin", "omega_cos"]
    names += ["nearest_sidechain_dist", "packing_score", "void_ratio"]
    names += ["env_aromatic_ratio", "env_hbond_donor_ratio", "env_hbond_acceptor_ratio", "env_gly_pro_ratio"]
    names += ["self_metal_binder", "neighbor_metal_binder_6A", "neighbor_metal_binder_10A"]
    names += ["local_anisotropy"]
    return names

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


def read_ids(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def contiguous_segments(chains: list[str]) -> list[tuple[str, int, int]]:
    if not chains:
        return []
    segments: list[tuple[str, int, int]] = []
    start = 0
    current = chains[0]
    for idx, chain_id in enumerate(chains[1:], start=1):
        if chain_id != current:
            segments.append((current, start, idx))
            start = idx
            current = chain_id
    segments.append((current, start, len(chains)))
    return segments


def feature_tensor(data: dict[str, Any], feature_set: str) -> tuple[torch.Tensor, list[str]]:
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    if feature_set in {"scalar", "scalar_physchem"}:
        value = torch.as_tensor(data["pred_spatial_scalar_features"], dtype=torch.float32)
        tensors.append(value)
        canonical_names = get_spatial_scalar_feature_names_v4()
        raw_names = list(data.get("spatial_scalar_feature_names", []))
        if value.shape[1] == len(canonical_names):
            names.extend([f"pred_scalar:{name}" for name in canonical_names])
        elif len(raw_names) == value.shape[1]:
            names.extend([f"pred_scalar:{name}" for name in raw_names])
        else:
            names.extend([f"pred_scalar:{idx}" for idx in range(value.shape[1])])
    if feature_set == "scalar_physchem":
        value = torch.as_tensor(data["pred_physchem_features"], dtype=torch.float32)
        tensors.append(value)
        names.extend([f"pred_physchem:{idx}" for idx in range(value.shape[1])])
    if not tensors:
        raise ValueError(f"Unsupported feature_set={feature_set!r}")
    features = torch.cat(tensors, dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), names


def template_feature_names(pred_root: Path, feature_set: str) -> list[str]:
    for path in sorted((pred_root / "pt").glob("*/*_predcomplete.pt")):
        data = torch_load(path)
        _features, names = feature_tensor(data, feature_set)
        return names
    raise FileNotFoundError(f"No *_predcomplete.pt files under {pred_root / 'pt'}")


def build_one(
    pdb_id: str,
    esm_root: Path,
    pred_root: Path,
    output_root: Path,
    feature_set: str,
    template_names: list[str],
    missing_chain_policy: str,
    add_availability: bool,
    overwrite: bool,
) -> dict[str, Any]:
    out_path = output_root / pdb_id / f"{pdb_id}_seq.pt"
    row = {
        "pdb_id": pdb_id,
        "status": "ERROR",
        "message": "",
        "n_residues": "",
        "n_features": "",
        "n_chains": "",
        "n_available_chains": "",
        "n_available_residues": "",
        "coverage_residue": "",
        "output_path": str(out_path),
        "mapping_methods": "",
    }
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        row["status"] = "SKIP"
        return row
    try:
        esm_path = esm_root / pdb_id / f"{pdb_id}_esm2.pt"
        if not esm_path.exists():
            raise FileNotFoundError(f"missing ESM file: {esm_path}")
        esm = torch_load(esm_path)
        n_res = int(torch.as_tensor(esm["embeddings"]).shape[0])
        residues = [norm_text(x).upper() for x in list(esm.get("residue_names_1", []))]
        chains = [norm_text(x) for x in list(esm.get("chain_ids", esm.get("chain_id", [])))]
        if len(residues) != n_res or len(chains) != n_res:
            raise ValueError(f"bad ESM metadata lengths residues={len(residues)} chains={len(chains)} n={n_res}")

        chunks: list[torch.Tensor] = []
        feature_names: list[str] | None = None
        mapping_methods: list[str] = []
        residue_indices: list[Any] = []
        insertion_codes: list[Any] = []
        uniprot_positions: list[Any] = []
        available_residues = 0
        available_chains = 0
        for chain_id, start, end in contiguous_segments(chains):
            pred_path = pred_root / "pt" / pdb_id / f"{pdb_id}__{chain_id}_predcomplete.pt"
            if not pred_path.exists():
                if missing_chain_policy != "zero":
                    raise FileNotFoundError(f"missing predicted chain feature: {pred_path}")
                names = template_names
                features = torch.zeros((end - start, len(names)), dtype=torch.float32)
                method = "missing_zero"
                residue_indices.extend([""] * (end - start))
                insertion_codes.extend([""] * (end - start))
                uniprot_positions.extend([""] * (end - start))
            else:
                pred = torch_load(pred_path)
                pred_residues = [norm_text(x).upper() for x in list(pred["residue_names_1"])]
                expected = residues[start:end]
                if pred_residues != expected:
                    mismatch = next((i for i, (a, b) in enumerate(zip(pred_residues, expected), start=1) if a != b), 0)
                    raise ValueError(
                        f"{chain_id}: residue mismatch at chain position {mismatch}; "
                        f"pred={''.join(pred_residues[:30])} esm={''.join(expected[:30])}"
                    )
                features, names = feature_tensor(pred, feature_set)
                if features.shape[0] != end - start:
                    raise ValueError(f"{chain_id}: feature length {features.shape[0]} != segment length {end-start}")
                method = str(pred.get("pred_mapping_method", ""))
                residue_indices.extend(list(pred.get("residue_indices", [])))
                insertion_codes.extend(list(pred.get("insertion_codes", [])))
                uniprot_positions.extend(list(pred.get("uniprot_positions", [])))
                available_residues += end - start
                available_chains += 1
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError(f"{chain_id}: feature names mismatch")
            if add_availability:
                availability = torch.ones((features.shape[0], 1), dtype=torch.float32)
                if method == "missing_zero":
                    availability.zero_()
                features = torch.cat([features, availability], dim=1)
            chunks.append(features)
            mapping_methods.append(method)

        seq_features = torch.cat(chunks, dim=0)
        if seq_features.shape[0] != n_res:
            raise ValueError(f"stitched length {seq_features.shape[0]} != ESM length {n_res}")
        if not torch.isfinite(seq_features).all():
            raise ValueError("nonfinite structural sequence features")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.pt")
        torch.save(
            {
                "pdb_id": pdb_id,
                "seq_features": seq_features,
                "feature_names": feature_names or [],
                "feature_names_with_availability": (feature_names or []) + (["pred_struct_available"] if add_availability else []),
                "sequence": "".join(residues),
                "residue_names_1": residues,
                "chain_ids": chains,
                "pred_residue_indices": residue_indices,
                "pred_insertion_codes": insertion_codes,
                "uniprot_positions": uniprot_positions,
                "pred_struct_root": str(pred_root),
                "esm_path": str(esm_path),
                "feature_set": feature_set,
                "missing_chain_policy": missing_chain_policy,
                "add_availability": add_availability,
                "n_available_chains": available_chains,
                "n_available_residues": available_residues,
                "coverage_residue": float(available_residues / max(1, n_res)),
                "mapping_methods": mapping_methods,
                "mapping_policy": (
                    "strict complex stitching: ESM chain/residue order must equal "
                    "predicted complete chain residue order; predicted structures are "
                    "TmAlphaFold/AlphaFold, not experimental complex coordinates"
                ),
            },
            tmp_path,
        )
        tmp_path.replace(out_path)
        row.update(
            {
                "status": "OK",
                "n_residues": int(seq_features.shape[0]),
                "n_features": int(seq_features.shape[1]),
                "n_chains": len(mapping_methods),
                "n_available_chains": available_chains,
                "n_available_residues": available_residues,
                "coverage_residue": f"{available_residues / max(1, n_res):.8f}",
                "mapping_methods": ";".join(mapping_methods),
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001
        row["message"] = str(exc)[:1000]
        return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "status",
        "message",
        "n_residues",
        "n_features",
        "n_chains",
        "n_available_chains",
        "n_available_residues",
        "coverage_residue",
        "output_path",
        "mapping_methods",
    ]
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.csv")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_split_intersection(source_split: Path, output_split: Path, ok_ids: set[str]) -> dict[str, int]:
    output_split.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        src = source_split / f"{split}_ids.txt"
        if not src.exists():
            continue
        ids = [pdb_id for pdb_id in read_ids(src) if pdb_id in ok_ids]
        (output_split / f"{split}_ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
        counts[split] = len(ids)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-root", type=Path, default=Path("features/esm2_t33_650M_UR50D_mlc/pt"))
    parser.add_argument("--pred-root", type=Path, default=Path("features/pred_struct_complete_v1_relaxed"))
    parser.add_argument("--source-split-dir", type=Path, default=Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit"))
    parser.add_argument("--output-root", type=Path, default=Path("features/pred_struct_sequence_scalar_relaxed/pt"))
    parser.add_argument("--output-split-dir", type=Path, default=Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit_predstruct_relaxed"))
    parser.add_argument("--manifest", type=Path, default=Path("features/pred_struct_sequence_scalar_relaxed/manifest.csv"))
    parser.add_argument("--feature-set", choices=("scalar", "scalar_physchem"), default="scalar")
    parser.add_argument("--missing-chain-policy", choices=("error", "zero"), default="error")
    parser.add_argument("--add-availability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    template_names = template_feature_names(args.pred_root, args.feature_set)
    ids = sorted({path.parent.name.lower() for path in args.esm_root.glob("*/*_esm2.pt")})
    if args.max and args.max > 0:
        ids = ids[: args.max]
    rows = [
        build_one(
            pdb_id,
            args.esm_root,
            args.pred_root,
            args.output_root,
            args.feature_set,
            template_names,
            args.missing_chain_policy,
            args.add_availability,
            args.overwrite,
        )
        for pdb_id in ids
    ]
    write_csv(args.manifest, rows)
    ok_ids = {row["pdb_id"] for row in rows if row["status"] == "OK"}
    split_counts = write_split_intersection(args.source_split_dir, args.output_split_dir, ok_ids)
    summary = {
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "output_split_dir": str(args.output_split_dir),
        "feature_set": args.feature_set,
        "n_total": len(rows),
        "status_counts": Counter(row["status"] for row in rows),
        "n_ok": len(ok_ids),
        "split_counts": split_counts,
        "n_features": sorted({int(row["n_features"]) for row in rows if row["status"] == "OK" and row["n_features"]}),
    }
    summary_path = args.manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
