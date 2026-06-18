#!/usr/bin/env python3
"""Augment strict sequence_v1 features with stronger sequence-only context features.

The parent sequence_v1 files already enforce strict ESM/label/fixed-PDB mapping.
This script only uses sequence strings and chain IDs saved in sequence_v1.  It
does not read PDB coordinates, XML topology, DSSP, SASA, or other structure-
derived tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ORDER)
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
AA_GROUPS = {
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQCY"),
    "hydrophobic": set("AILMFWV"),
    "aromatic": set("FWYH"),
    "small": set("AGSTCV"),
}
AA_HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
BLOSUM62 = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
    "C": [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    "D": [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
    "E": [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
    "F": [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
    "G": [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -4, 0, -2, -2, -2, 0, -2, -3, -2, -3],
    "H": [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
    "I": [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
    "K": [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
    "L": [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
    "M": [-1, -1, -3, -2, 0, -4, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
    "N": [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
    "P": [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
    "Q": [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
    "R": [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
    "S": [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
    "T": [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
    "V": [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
    "W": [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
    "Y": [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
}
ATCHLEY = {
    "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
    "C": [-1.343, 0.465, -0.862, -1.020, -0.255],
    "D": [1.050, 0.302, -3.656, -0.259, -3.242],
    "E": [1.357, -1.453, 1.477, 0.113, -0.837],
    "F": [-1.006, -0.590, 1.891, -0.397, 0.412],
    "G": [-0.384, 1.652, 1.330, 1.045, 2.064],
    "H": [0.336, -0.417, -1.673, -1.474, -0.078],
    "I": [-1.239, -0.547, 2.131, 0.393, 0.816],
    "K": [1.831, -0.561, 0.533, -0.277, 1.648],
    "L": [-1.019, -0.987, -1.505, 1.266, -0.912],
    "M": [-0.663, -1.524, 2.219, -1.005, 1.212],
    "N": [0.945, 0.828, 1.299, -0.169, 0.933],
    "P": [0.189, 2.081, -1.628, 0.421, -1.392],
    "Q": [0.931, -0.179, -3.005, -0.503, -1.853],
    "R": [1.538, -0.055, 1.502, 0.440, 2.897],
    "S": [-0.228, 1.399, -4.760, 0.670, -2.647],
    "T": [-0.032, 0.326, 2.213, 0.908, 1.313],
    "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
    "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
    "Y": [0.260, 0.830, 3.097, -0.838, 1.512],
}


def torch_load(path: Path) -> Any:
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def read_ids(split_dir: Path | None, ids_file: Path | None) -> list[str]:
    if ids_file is not None:
        files = [ids_file]
    elif split_dir is not None:
        files = [split_dir / "train_ids.txt", split_dir / "val_ids.txt", split_dir / "test_ids.txt"]
    else:
        raise ValueError("Provide --split-dir or --ids-file.")
    ids: list[str] = []
    for path in files:
        for line in path.read_text().splitlines():
            value = line.strip().lower()
            if value and not value.startswith("#"):
                ids.append(value)
    return sorted(set(ids))


def onehot(aa: str) -> list[float]:
    row = [0.0] * len(AA_ORDER)
    if aa in AA_TO_INDEX:
        row[AA_TO_INDEX[aa]] = 1.0
    return row


def entropy(chars: list[str]) -> float:
    valid = [aa for aa in chars if aa in AA_SET]
    if not valid:
        return 0.0
    counts = [valid.count(aa) for aa in set(valid)]
    total = float(len(valid))
    value = -sum((count / total) * math.log(count / total + 1.0e-12) for count in counts)
    return value / math.log(20.0)


def hydrophobic_moment(chars: list[str], degrees: float) -> float:
    if not chars:
        return 0.0
    angle = math.radians(degrees)
    x_total = 0.0
    y_total = 0.0
    n = 0
    for idx, aa in enumerate(chars):
        if aa not in AA_SET:
            continue
        h = AA_HYDROPATHY.get(aa, 0.0)
        x_total += h * math.cos(idx * angle)
        y_total += h * math.sin(idx * angle)
        n += 1
    if n <= 0:
        return 0.0
    return min(math.sqrt(x_total * x_total + y_total * y_total) / (n * 4.5), 1.0)


def chain_local_positions(chains: list[str]) -> tuple[list[int], list[int], list[int]]:
    chain_to_indices: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, chain in enumerate(chains):
        if chain not in chain_to_indices:
            chain_to_indices[chain] = []
            order.append(chain)
        chain_to_indices[chain].append(idx)
    local_pos = [0] * len(chains)
    chain_len = [0] * len(chains)
    chain_rank = {chain: idx for idx, chain in enumerate(order)}
    chain_index = [0] * len(chains)
    for chain, indices in chain_to_indices.items():
        for pos, global_idx in enumerate(indices):
            local_pos[global_idx] = pos
            chain_len[global_idx] = len(indices)
            chain_index[global_idx] = chain_rank[chain]
    return local_pos, chain_len, chain_index


def build_extra_features(sequence: list[str], chains: list[str]) -> tuple[torch.Tensor, list[str]]:
    n_res = len(sequence)
    local_pos, chain_len, _chain_index = chain_local_positions(chains)
    windows = [5, 9, 17, 33]
    moment_windows = [9, 17, 33]

    names: list[str] = []
    names += [f"prev_aa_{aa}" for aa in AA_ORDER]
    names += [f"next_aa_{aa}" for aa in AA_ORDER]
    names += [f"blosum62_{aa}" for aa in AA_ORDER]
    names += [f"atchley_{idx + 1}" for idx in range(5)]
    for scope in ("protein", "chain"):
        for freq in (1, 2, 4, 8):
            names += [f"{scope}_pos_sin{freq}", f"{scope}_pos_cos{freq}"]
    names += ["dist_to_n_norm", "dist_to_c_norm", "dist_to_chain_end_norm"]
    for window in windows:
        for group_name in AA_GROUPS:
            names.append(f"win{window}_frac_{group_name}")
        names += [f"win{window}_aa_entropy_v2", f"win{window}_max_aa_frac"]
    for window in moment_windows:
        for degrees in (100, 160):
            names.append(f"win{window}_hydrophobic_moment_{degrees}")

    rows: list[list[float]] = []
    for idx, aa in enumerate(sequence):
        row: list[float] = []
        prev_aa = sequence[idx - 1] if idx > 0 else "X"
        next_aa = sequence[idx + 1] if idx + 1 < n_res else "X"
        row.extend(onehot(prev_aa))
        row.extend(onehot(next_aa))
        row.extend([score / 11.0 for score in BLOSUM62.get(aa, [0] * 20)])
        row.extend([value / 5.0 for value in ATCHLEY.get(aa, [0.0] * 5)])

        protein_rel = idx / max(1, n_res - 1)
        chain_rel = local_pos[idx] / max(1, chain_len[idx] - 1)
        for rel in (protein_rel, chain_rel):
            for freq in (1, 2, 4, 8):
                angle = 2.0 * math.pi * freq * rel
                row.extend([math.sin(angle), math.cos(angle)])

        dist_n = local_pos[idx] / max(1, chain_len[idx])
        dist_c = (chain_len[idx] - 1 - local_pos[idx]) / max(1, chain_len[idx])
        row.extend([dist_n, dist_c, min(dist_n, dist_c)])

        for window in windows:
            radius = window // 2
            start = max(0, idx - radius)
            end = min(n_res, idx + radius + 1)
            subseq = sequence[start:end]
            valid = [x for x in subseq if x in AA_SET]
            denom = max(1, len(valid))
            for group in AA_GROUPS.values():
                row.append(sum(1 for x in valid if x in group) / denom)
            row.append(entropy(valid))
            counts = [valid.count(target) for target in set(valid)] if valid else [0]
            row.append(max(counts) / denom)

        for window in moment_windows:
            radius = window // 2
            start = max(0, idx - radius)
            end = min(n_res, idx + radius + 1)
            subseq = sequence[start:end]
            row.append(hydrophobic_moment(subseq, 100.0))
            row.append(hydrophobic_moment(subseq, 160.0))
        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32), names


def find_parent(parent_root: Path, pdb_id: str) -> Path:
    candidates = (
        parent_root / pdb_id / f"{pdb_id}_seq.pt",
        parent_root / pdb_id / f"{pdb_id}_sequence.pt",
        parent_root / f"{pdb_id}_seq.pt",
        parent_root / f"{pdb_id}.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{pdb_id}: parent sequence_v1 file not found under {parent_root}")


def process_one(task: tuple[str, str, str, bool]) -> dict[str, Any]:
    pdb_id, parent_root_s, output_root_s, overwrite = task
    parent_root = Path(parent_root_s)
    output_root = Path(output_root_s)
    out_dir = output_root / pdb_id
    out_path = out_dir / f"{pdb_id}_seq.pt"
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "status": "error",
        "message": "",
        "n_residues": "",
        "n_features": "",
        "n_parent_features": "",
        "n_extra_features": "",
        "output_path": str(out_path),
    }
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        row["status"] = "skip"
        row["message"] = "output exists"
        return row
    try:
        parent_path = find_parent(parent_root, pdb_id)
        parent = torch_load(parent_path)
        parent_features = torch.as_tensor(parent["seq_features"], dtype=torch.float32)
        sequence = [norm_text(x).upper() for x in list(parent.get("residue_names_1") or list(parent["sequence"]))]
        chains = [norm_text(x) for x in list(parent["chain_ids"])]
        if parent_features.ndim != 2:
            raise ValueError(f"parent seq_features must be 2D, got {tuple(parent_features.shape)}")
        if len(sequence) != int(parent_features.shape[0]):
            raise ValueError(f"sequence length {len(sequence)} != parent feature length {parent_features.shape[0]}")
        if len(chains) != int(parent_features.shape[0]):
            raise ValueError(f"chain length {len(chains)} != parent feature length {parent_features.shape[0]}")
        extra, extra_names = build_extra_features(sequence, chains)
        features = torch.cat([parent_features, extra], dim=1)
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite sequence_v2 features")
        feature_names = list(parent["feature_names"]) + extra_names
        if int(features.shape[1]) != len(feature_names):
            raise ValueError(f"feature dim {features.shape[1]} != feature_names {len(feature_names)}")

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.pt")
        payload = dict(parent)
        payload.update(
            {
                "pdb_id": pdb_id,
                "seq_features": features,
                "feature_names": feature_names,
                "sequence_v1_path": str(parent_path),
                "sequence_v1_dim": int(parent_features.shape[1]),
                "sequence_v2_extra_dim": int(extra.shape[1]),
                "sequence_v2_extra_feature_names": extra_names,
                "mapping_policy": parent.get(
                    "mapping_policy",
                    "inherited strict mapping from sequence_v1 parent features",
                ),
                "input_policy": "sequence_v2 uses sequence and chain IDs only; no coordinates/XML/topology/structure-derived features",
                "forbidden_inputs": [
                    "coordinates",
                    "PDB-derived SASA",
                    "PDB-derived DSSP",
                    "PDB-derived hbond",
                    "PDB-derived local density",
                    "PDBTM XML annotations as model inputs",
                    "MSA or homolog database features",
                ],
            }
        )
        torch.save(payload, tmp_path)
        tmp_path.replace(out_path)
        row.update(
            {
                "status": "ok",
                "n_residues": int(features.shape[0]),
                "n_features": int(features.shape[1]),
                "n_parent_features": int(parent_features.shape[1]),
                "n_extra_features": int(extra.shape[1]),
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001
        row["message"] = str(exc)[:1000]
        return row


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "status",
        "message",
        "n_residues",
        "n_features",
        "n_parent_features",
        "n_extra_features",
        "output_path",
    ]
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.csv")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=Path("features/sequence_v1/pt"))
    parser.add_argument("--output-root", type=Path, default=Path("features/sequence_v2/pt"))
    parser.add_argument("--split-dir", type=Path, default=Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit"))
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("features/sequence_v2/extract_report.csv"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    ids = read_ids(args.split_dir, args.ids_file)
    if args.max is not None:
        ids = ids[: args.max]
    tasks = [(pdb_id, str(args.parent_root), str(args.output_root), bool(args.overwrite)) for pdb_id in ids]
    workers = max(1, min(int(args.workers), len(tasks) if tasks else 1))

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            rows.append(process_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda item: item["pdb_id"])
    write_report(args.report, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "n_total": len(rows),
        "statuses": counts,
        "output_root": str(args.output_root),
        "report": str(args.report),
        "workers": workers,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
