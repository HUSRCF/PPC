#!/usr/bin/env python3
"""Extract strict sequence-level residue features for PPC contact-site models.

This script intentionally does not use structural coordinates as model input.
It uses fixed PDB files only to verify residue/chain order against ESM metadata
and offline labels. Samples with any mismatch are reported and skipped.
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
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

AA_GROUPS = {
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQCY"),
    "hydrophobic": set("AILMFWV"),
    "aromatic": set("FWYH"),
    "small": set("AGSTCV"),
}

AA_MASS = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
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
AA_VOLUME = {
    "A": 88.6,
    "C": 108.5,
    "D": 111.1,
    "E": 138.4,
    "F": 189.9,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "K": 168.6,
    "L": 166.7,
    "M": 162.9,
    "N": 114.1,
    "P": 112.7,
    "Q": 143.8,
    "R": 173.4,
    "S": 89.0,
    "T": 116.1,
    "V": 140.0,
    "W": 227.8,
    "Y": 193.6,
}
AA_CHARGE_PH7 = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}
AA_PI = {
    "A": 6.00,
    "C": 5.07,
    "D": 2.77,
    "E": 3.22,
    "F": 5.48,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "K": 9.74,
    "L": 5.98,
    "M": 5.74,
    "N": 5.41,
    "P": 6.30,
    "Q": 5.65,
    "R": 10.76,
    "S": 5.68,
    "T": 5.60,
    "V": 5.96,
    "W": 5.89,
    "Y": 5.66,
}
AA_PKA = {
    "D": 3.65,
    "E": 4.25,
    "C": 8.18,
    "Y": 10.07,
    "H": 6.00,
    "K": 10.53,
    "R": 12.48,
}


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def torch_load(path: Path) -> Any:
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def read_ids(split_dir: Path | None, ids_file: Path | None) -> list[str]:
    ids: list[str] = []
    if ids_file is not None:
        files = [ids_file]
    elif split_dir is not None:
        files = [split_dir / "train_ids.txt", split_dir / "val_ids.txt", split_dir / "test_ids.txt"]
    else:
        raise ValueError("Provide --split-dir or --ids-file.")
    for path in files:
        with path.open("rt") as handle:
            for line in handle:
                value = line.strip().lower()
                if value and not value.startswith("#"):
                    ids.append(value)
    return sorted(set(ids))


def load_label(label_root: Path, pdb_id: str) -> torch.Tensor:
    candidates = (
        label_root / pdb_id / f"{pdb_id}_labels.pt",
        label_root / pdb_id / f"{pdb_id}_contact.pt",
        label_root / f"{pdb_id}_labels.pt",
        label_root / f"{pdb_id}.pt",
    )
    for path in candidates:
        if not path.exists():
            continue
        data = torch_load(path)
        if isinstance(data, dict):
            value = data.get("labels", data.get("is_contact", data.get("contact_labels")))
        else:
            value = data
        if value is None:
            raise ValueError(f"{pdb_id}: label file has no labels/is_contact/contact_labels")
        return torch.as_tensor(value, dtype=torch.long)
    raise FileNotFoundError(f"{pdb_id}: label file not found under {label_root}")


def load_esm_metadata(esm_root: Path, pdb_id: str) -> dict[str, Any]:
    candidates = (
        esm_root / pdb_id / f"{pdb_id}_esm2.pt",
        esm_root / pdb_id / f"{pdb_id}_protein.pt",
        esm_root / f"{pdb_id}_esm2.pt",
        esm_root / f"{pdb_id}_protein.pt",
    )
    for path in candidates:
        if not path.exists():
            continue
        data = torch_load(path)
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(f"{pdb_id}: ESM file has no embeddings")
        n_res = int(embeddings.shape[0])
        residues = data.get("residue_names_1", data.get("residue_name_1"))
        chains = data.get("chain_ids", data.get("chain_id"))
        if residues is None or chains is None:
            raise ValueError(f"{pdb_id}: ESM file missing residue_names_1/chain_ids")
        residues = [norm_text(x).upper() for x in list(residues)]
        chains = [norm_text(x) for x in list(chains)]
        if len(residues) != n_res:
            raise ValueError(f"{pdb_id}: ESM residues length {len(residues)} != embeddings length {n_res}")
        if len(chains) != n_res:
            raise ValueError(f"{pdb_id}: ESM chain length {len(chains)} != embeddings length {n_res}")
        return {"path": path, "n_res": n_res, "residues": residues, "chains": chains}
    raise FileNotFoundError(f"{pdb_id}: ESM file not found under {esm_root}")


def parse_fixed_pdb_ca(pdb_path: Path) -> dict[str, list[Any]]:
    residues: list[str] = []
    chains: list[str] = []
    resseqs: list[int] = []
    icodes: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    with pdb_path.open("rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip()
            resseq_text = line[22:26].strip()
            icode = line[26].strip()
            if not resseq_text:
                continue
            resseq = int(resseq_text)
            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            residues.append(AA3_TO_1.get(resname, "X"))
            chains.append(chain_id)
            resseqs.append(resseq)
            icodes.append(icode)
    return {"residues": residues, "chains": chains, "resseqs": resseqs, "icodes": icodes}


def normalize(values: list[float], min_value: float, max_value: float) -> list[float]:
    scale = max(max_value - min_value, 1.0e-8)
    return [(v - min_value) / scale for v in values]


def shannon_entropy(chars: list[str]) -> float:
    valid = [x for x in chars if x in AA_SET]
    if not valid:
        return 0.0
    counts = {aa: valid.count(aa) for aa in set(valid)}
    total = float(len(valid))
    entropy = -sum((c / total) * math.log(c / total + 1.0e-12) for c in counts.values())
    return entropy / math.log(20.0)


def rolling_stats(values: list[float], sequence: list[str], windows: list[int]) -> tuple[list[list[float]], list[str]]:
    out = [[0.0 for _ in range(0)] for _ in values]
    names: list[str] = []
    for window in windows:
        radius = window // 2
        for name in ("hydropathy", "charge", "mass", "volume"):
            names.extend([f"win{window}_{name}_mean", f"win{window}_{name}_std"])
        names.append(f"win{window}_entropy")
        for i in range(len(values)):
            start = max(0, i - radius)
            end = min(len(values), i + radius + 1)
            subseq = sequence[start:end]
            props = {
                "hydropathy": [AA_HYDROPATHY.get(aa, 0.0) for aa in subseq],
                "charge": [AA_CHARGE_PH7.get(aa, 0.0) for aa in subseq],
                "mass": normalize([AA_MASS.get(aa, 0.0) for aa in subseq], 75.0, 205.0),
                "volume": normalize([AA_VOLUME.get(aa, 0.0) for aa in subseq], 60.0, 228.0),
            }
            row_values: list[float] = []
            for key in ("hydropathy", "charge", "mass", "volume"):
                xs = props[key]
                mean = sum(xs) / max(1, len(xs))
                var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs))
                if key == "hydropathy":
                    mean = (mean + 4.5) / 9.0
                    std = math.sqrt(var) / 9.0
                elif key == "charge":
                    mean = (mean + 1.0) / 2.0
                    std = math.sqrt(var) / 2.0
                else:
                    std = math.sqrt(var)
                row_values.extend([mean, std])
            row_values.append(shannon_entropy(subseq))
            out[i].extend(row_values)
    return out, names


def composition(sequence: list[str]) -> list[float]:
    denom = max(1, sum(1 for aa in sequence if aa in AA_SET))
    return [sum(1 for aa in sequence if aa == target) / denom for target in AA_ORDER]


def build_features(residues: list[str], chains: list[str]) -> tuple[torch.Tensor, list[str]]:
    n = len(residues)
    chain_to_indices: dict[str, list[int]] = {}
    chain_order: list[str] = []
    for idx, chain in enumerate(chains):
        if chain not in chain_to_indices:
            chain_to_indices[chain] = []
            chain_order.append(chain)
        chain_to_indices[chain].append(idx)

    protein_aac = composition(residues)
    chain_aac = {chain: composition([residues[i] for i in idxs]) for chain, idxs in chain_to_indices.items()}
    protein_len_log = math.log1p(n) / math.log1p(4096.0)
    chain_count_norm = min(len(chain_order), 128) / 128.0

    base_names = [f"aa_{aa}" for aa in AA_ORDER]
    base_names += [f"group_{name}" for name in AA_GROUPS]
    base_names += ["mass", "hydropathy", "volume", "charge", "pI", "pKa", "has_pKa"]
    pos_names = [
        "protein_rel_pos",
        "chain_rel_pos",
        "is_n_term10",
        "is_c_term10",
        "is_n_term20",
        "is_c_term20",
        "protein_len_log",
        "chain_len_log",
        "chain_count_norm",
        "chain_index_norm",
    ]
    global_names = [f"protein_aac_{aa}" for aa in AA_ORDER]
    global_names += [f"chain_aac_{aa}" for aa in AA_ORDER]
    global_names += [f"chain_minus_protein_aac_{aa}" for aa in AA_ORDER]
    window_rows, window_names = rolling_stats([0.0] * n, residues, windows=[7, 15, 31])
    feature_names = base_names + pos_names + global_names + window_names

    rows: list[list[float]] = []
    chain_rank = {chain: idx for idx, chain in enumerate(chain_order)}
    chain_pos: dict[int, int] = {}
    for chain, idxs in chain_to_indices.items():
        for local_idx, global_idx in enumerate(idxs):
            chain_pos[global_idx] = local_idx

    for i, aa in enumerate(residues):
        row: list[float] = []
        onehot = [0.0] * len(AA_ORDER)
        if aa in AA_TO_INDEX:
            onehot[AA_TO_INDEX[aa]] = 1.0
        row.extend(onehot)
        for group in AA_GROUPS.values():
            row.append(1.0 if aa in group else 0.0)
        row.append((AA_MASS.get(aa, 75.0) - 75.0) / (205.0 - 75.0))
        row.append((AA_HYDROPATHY.get(aa, 0.0) + 4.5) / 9.0)
        row.append((AA_VOLUME.get(aa, 60.0) - 60.0) / (228.0 - 60.0))
        row.append((AA_CHARGE_PH7.get(aa, 0.0) + 1.0) / 2.0)
        row.append(AA_PI.get(aa, 6.0) / 14.0)
        row.append(AA_PKA.get(aa, 0.0) / 14.0)
        row.append(1.0 if aa in AA_PKA else 0.0)

        chain = chains[i]
        idxs = chain_to_indices[chain]
        chain_len = len(idxs)
        local_pos = chain_pos[i]
        protein_rel_pos = i / max(1, n - 1)
        chain_rel_pos = local_pos / max(1, chain_len - 1)
        row.extend(
            [
                protein_rel_pos,
                chain_rel_pos,
                1.0 if local_pos < 10 else 0.0,
                1.0 if local_pos >= chain_len - 10 else 0.0,
                1.0 if local_pos < 20 else 0.0,
                1.0 if local_pos >= chain_len - 20 else 0.0,
                protein_len_log,
                math.log1p(chain_len) / math.log1p(4096.0),
                chain_count_norm,
                chain_rank[chain] / max(1, len(chain_order) - 1),
            ]
        )
        caac = chain_aac[chain]
        row.extend(protein_aac)
        row.extend(caac)
        row.extend([c - p for c, p in zip(caac, protein_aac)])
        row.extend(window_rows[i])
        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32), feature_names


def compare_lists(name: str, expected: list[str], observed: list[str]) -> str | None:
    if len(expected) != len(observed):
        return f"{name} length mismatch: expected {len(expected)} observed {len(observed)}"
    for idx, (a, b) in enumerate(zip(expected, observed)):
        if norm_text(a) != norm_text(b):
            return f"{name} mismatch at index {idx}: expected {a!r} observed {b!r}"
    return None


def process_one(task: tuple[str, str, str, str, str, bool]) -> dict[str, Any]:
    pdb_id, esm_root_s, label_root_s, pdb_root_s, output_root_s, overwrite = task
    esm_root = Path(esm_root_s)
    label_root = Path(label_root_s)
    pdb_root = Path(pdb_root_s)
    output_root = Path(output_root_s)
    out_dir = output_root / pdb_id
    out_path = out_dir / f"{pdb_id}_seq.pt"
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "status": "error",
        "message": "",
        "n_residues": "",
        "n_features": "",
        "output_path": str(out_path),
    }
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        row["status"] = "skip"
        row["message"] = "output exists"
        return row
    try:
        esm = load_esm_metadata(esm_root, pdb_id)
        labels = load_label(label_root, pdb_id)
        if int(labels.shape[0]) != int(esm["n_res"]):
            raise ValueError(f"label length {labels.shape[0]} != ESM length {esm['n_res']}")

        pdb_path = pdb_root / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"fixed PDB not found: {pdb_path}")
        pdb = parse_fixed_pdb_ca(pdb_path)
        msg = compare_lists("residue", esm["residues"], pdb["residues"])
        if msg:
            raise ValueError(msg)
        msg = compare_lists("chain", esm["chains"], pdb["chains"])
        if msg:
            raise ValueError(msg)

        features, feature_names = build_features(esm["residues"], esm["chains"])
        if int(features.shape[0]) != int(esm["n_res"]):
            raise ValueError(f"feature length {features.shape[0]} != ESM length {esm['n_res']}")
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite sequence features")

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.pt")
        torch.save(
            {
                "pdb_id": pdb_id,
                "seq_features": features,
                "feature_names": feature_names,
                "sequence": "".join(esm["residues"]),
                "residue_names_1": esm["residues"],
                "chain_ids": esm["chains"],
                "pdb_resseq": pdb["resseqs"],
                "pdb_icode": pdb["icodes"],
                "esm_path": str(esm["path"]),
                "pdb_path": str(pdb_path),
                "mapping_policy": "strict: ESM length/residue/chain == label length == fixed-PDB CA order",
                "forbidden_inputs": [
                    "coordinates",
                    "PDB-derived SASA",
                    "PDB-derived DSSP",
                    "PDB-derived hbond",
                    "PDB-derived local density",
                    "PDBTM XML annotations as model inputs",
                ],
            },
            tmp_path,
        )
        tmp_path.replace(out_path)
        row["status"] = "ok"
        row["message"] = ""
        row["n_residues"] = int(features.shape[0])
        row["n_features"] = int(features.shape[1])
        return row
    except Exception as exc:  # noqa: BLE001
        row["message"] = str(exc)[:1000]
        return row


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pdb_id", "status", "message", "n_residues", "n_features", "output_path"]
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.csv")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-root", required=True, type=Path)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--pdb-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    ids = read_ids(args.split_dir, args.ids_file)
    if args.max is not None:
        ids = ids[: args.max]
    tasks = [
        (
            pdb_id,
            str(args.esm_root),
            str(args.label_root),
            str(args.pdb_root),
            str(args.output_root),
            bool(args.overwrite),
        )
        for pdb_id in ids
    ]

    rows: list[dict[str, Any]] = []
    workers = max(1, min(int(args.workers), len(tasks) if tasks else 1))
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
