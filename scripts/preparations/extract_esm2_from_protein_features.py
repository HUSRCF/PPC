#!/usr/bin/env python3
"""Extract ESM2 embeddings aligned to PPC complete protein features.

The alignment source is ``features/protein_v4/pt/*/*_protein.pt``.  For each
feature file, this script builds per-chain amino-acid sequences from
``residue_names``, runs ESM2 on each chain, and writes one embedding row per
original feature residue.  Insertion codes are not re-parsed from PDB; they are
copied from the feature file so downstream joins use the same residue table.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

try:
    import esm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fair-esm is not installed. Run this in the AIAA env.") from exc


ESM2_MAX_LEN = 1022
ESM2_WINDOW_OVERLAP = 768

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
    "ASX": "B",
    "GLX": "Z",
    "SEC": "C",
    "PYL": "K",
    "MSE": "M",
    "TPO": "T",
    "SEP": "S",
    "PTR": "Y",
    "HIC": "H",
    "KCX": "K",
    "FME": "M",
    "CSO": "C",
    "CSD": "C",
    "CME": "C",
    "OCS": "C",
    "NEP": "H",
    "MHS": "H",
    "LLP": "K",
    "MLZ": "K",
    "M3L": "K",
    "AGM": "R",
    "2MR": "R",
    "HYP": "P",
    "P1L": "C",
    "DYA": "D",
    "MGN": "Q",
    "SMC": "C",
    "GL3": "G",
    "TRX": "C",
    "ELY": "K",
    "0TD": "D",
    "33W": "W",
    "PCA": "Q",
}


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return list(value)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _norm_resseq(value: Any) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return int(value)


def _feature_pdb_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _discover_features(features_root: Path, id_list: Path | None) -> list[Path]:
    if id_list is None:
        return sorted(features_root.glob("*/*_protein.pt"))
    ids = [
        line.strip().lower()
        for line in id_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    paths: list[Path] = []
    for pdb_id in ids:
        path = features_root / pdb_id / f"{pdb_id}_protein.pt"
        if path.exists():
            paths.append(path)
    return sorted(paths)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        device_str = "cuda:0"
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device_str} requested but CUDA is not available")
        idx = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
        if idx >= torch.cuda.device_count():
            raise ValueError(f"Invalid CUDA device {device_str}; count={torch.cuda.device_count()}")
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def _parse_devices(devices: str) -> list[str]:
    out: list[str] = []
    for part in devices.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            out.append(f"cuda:{part}")
        elif part == "cuda":
            out.append("cuda:0")
        else:
            out.append(part)
    return out


def _load_model(model_name: str, device: torch.device):
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown ESM2 model loader: esm.pretrained.{model_name}")
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.to(device).eval()
    return model, alphabet


def _window_starts(n_res: int, window_len: int, overlap: int) -> list[int]:
    if n_res <= window_len:
        return [0]
    if overlap < 0 or overlap >= window_len:
        raise ValueError(f"Invalid overlap {overlap}; expected 0 <= overlap < {window_len}")
    stride = window_len - overlap
    starts = [0]
    while starts[-1] + window_len < n_res:
        nxt = starts[-1] + stride
        if nxt + window_len >= n_res:
            nxt = n_res - window_len
        if nxt <= starts[-1]:
            break
        starts.append(nxt)
    return starts


def _parse_int_list(value: str | None, default: list[int] | None = None) -> list[int]:
    if value is None or not str(value).strip():
        return list(default or [])
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _combine_representations(representations: torch.Tensor, layers: list[int], mode: str) -> torch.Tensor:
    if representations.ndim != 3:
        raise ValueError(f"Expected layer representations with shape (R, L, D), got {tuple(representations.shape)}")
    if mode == "concat":
        return representations.permute(1, 0, 2).reshape(representations.shape[1], representations.shape[0] * representations.shape[2])
    if mode == "mean":
        return representations.mean(dim=0)
    if mode == "last":
        return representations[layers.index(max(layers))]
    raise ValueError(f"Unsupported embedding_combine={mode!r}; use concat, mean, or last")


def _run_esm2_once(
    model,
    alphabet,
    sequence: str,
    repr_layers: list[int] | None = None,
    embedding_combine: str = "last",
    return_contacts: bool = False,
) -> dict[str, torch.Tensor | None]:
    if len(sequence) > ESM2_MAX_LEN:
        raise ValueError(f"Sequence length {len(sequence)} exceeds ESM2 limit {ESM2_MAX_LEN}")
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("protein", sequence)])
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    layers = sorted(set(int(x) for x in (repr_layers or [model.num_layers])))
    with torch.no_grad():
        out = model(tokens, repr_layers=layers, return_contacts=return_contacts)
    reps = torch.stack([out["representations"][layer][0, 1:-1].detach().cpu() for layer in layers], dim=0)
    emb = _combine_representations(reps, layers, embedding_combine)
    if emb.shape[0] != len(sequence):
        raise RuntimeError(f"Embedding length mismatch: {emb.shape[0]} vs {len(sequence)}")
    contacts = None
    if return_contacts and out.get("contacts") is not None:
        contacts = out["contacts"][0].detach().cpu().float()
        if contacts.shape != (len(sequence), len(sequence)):
            raise RuntimeError(f"Contact map shape mismatch: {tuple(contacts.shape)} vs {len(sequence)}")
    return {"embeddings": emb, "layer_embeddings": reps, "contacts": contacts}


def _contact_edges_from_map(
    contacts: torch.Tensor | None,
    positions: list[int],
    top_k: int,
    min_score: float,
    min_seq_sep: int,
    bidirectional: bool,
) -> dict[tuple[int, int], float]:
    if contacts is None or top_k <= 0:
        return {}
    n_res = int(contacts.shape[0])
    if n_res <= 1:
        return {}
    scores = torch.nan_to_num(contacts.float(), nan=-1.0, posinf=1.0, neginf=-1.0).clone()
    idx = torch.arange(n_res)
    seq_sep = torch.abs(idx[:, None] - idx[None, :])
    scores.masked_fill_(seq_sep < int(min_seq_sep), float("-inf"))
    if min_score > 0:
        scores.masked_fill_(scores < float(min_score), float("-inf"))
    k = min(int(top_k), max(1, n_res - 1))
    top_scores, top_idx = torch.topk(scores, k=k, dim=1, largest=True)
    edge_scores: dict[tuple[int, int], float] = {}
    for src_local in range(n_res):
        src_global = int(positions[src_local])
        for score_value, dst_local_tensor in zip(top_scores[src_local].tolist(), top_idx[src_local].tolist()):
            if not isinstance(score_value, float) or not torch.isfinite(torch.tensor(score_value)):
                continue
            if score_value < float(min_score):
                continue
            dst_global = int(positions[int(dst_local_tensor)])
            if src_global == dst_global:
                continue
            key = (src_global, dst_global)
            edge_scores[key] = max(edge_scores.get(key, -1.0), float(score_value))
            if bidirectional:
                rkey = (dst_global, src_global)
                edge_scores[rkey] = max(edge_scores.get(rkey, -1.0), float(score_value))
    return edge_scores


def _run_esm2_windowed(
    model,
    alphabet,
    sequence: str,
    max_len: int,
    overlap: int,
    repr_layers: list[int] | None = None,
    embedding_combine: str = "last",
    return_contacts: bool = False,
    save_layer_embeddings: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]], dict[str, Any]]:
    n_res = len(sequence)
    layers = sorted(set(int(x) for x in (repr_layers or [getattr(model, "num_layers", -1)])))
    if n_res <= max_len:
        result = _run_esm2_once(
            model,
            alphabet,
            sequence,
            repr_layers=layers,
            embedding_combine=embedding_combine,
            return_contacts=return_contacts,
        )
        contacts = []
        if return_contacts and result["contacts"] is not None:
            contacts.append({"start": 0, "contacts": result["contacts"]})
        layer_embeddings = result["layer_embeddings"] if save_layer_embeddings else None
        return result["embeddings"], layer_embeddings, contacts, {
            "windowed": False,
            "window_len": max_len,
            "window_overlap": 0,
            "window_stride": max_len,
            "window_starts": [0],
            "n_windows": 1,
        }

    starts = _window_starts(n_res, max_len, overlap)
    acc: torch.Tensor | None = None
    layer_acc: torch.Tensor | None = None
    counts = torch.zeros(n_res, dtype=torch.float32)
    contact_windows: list[dict[str, Any]] = []
    for start in starts:
        end = min(start + max_len, n_res)
        result = _run_esm2_once(
            model,
            alphabet,
            sequence[start:end],
            repr_layers=layers,
            embedding_combine=embedding_combine,
            return_contacts=return_contacts,
        )
        emb = result["embeddings"]
        if acc is None:
            acc = torch.zeros(n_res, emb.shape[1], dtype=emb.dtype)
        acc[start:end] += emb
        if save_layer_embeddings:
            reps = result["layer_embeddings"]
            assert reps is not None
            if layer_acc is None:
                layer_acc = torch.zeros(len(layers), n_res, reps.shape[2], dtype=reps.dtype)
            layer_acc[:, start:end] += reps
        counts[start:end] += 1.0
        if return_contacts and result["contacts"] is not None:
            contact_windows.append({"start": start, "contacts": result["contacts"]})
    if acc is None or bool((counts == 0).any()):
        raise RuntimeError(f"Windowed ESM2 did not cover all residues: n_res={n_res}")
    emb_out = acc / counts[:, None]
    layer_out = layer_acc / counts[None, :, None] if layer_acc is not None else None
    return emb_out, layer_out, contact_windows, {
        "windowed": True,
        "window_len": max_len,
        "window_overlap": overlap,
        "window_stride": max_len - overlap,
        "window_starts": starts,
        "n_windows": len(starts),
    }


def _load_residue_table(feature_path: Path) -> dict[str, Any]:
    data = _torch_load(feature_path)
    residue_names = [_norm_text(x).upper() for x in _as_list(data.get("residue_names"))]
    chain_ids = [_norm_text(x) for x in _as_list(data.get("chain_ids"))]
    residue_indices = [_norm_resseq(x) for x in _as_list(data.get("residue_indices"))]
    insertion_codes = [_norm_text(x) for x in _as_list(data.get("insertion_codes"))]
    lengths = {
        "residue_names": len(residue_names),
        "chain_ids": len(chain_ids),
        "residue_indices": len(residue_indices),
        "insertion_codes": len(insertion_codes),
    }
    for key in ("physchem_features", "spatial_scalar_features", "spatial_vector_features", "ca_coords"):
        value = data.get(key)
        if value is not None:
            lengths[key] = int(value.shape[0])
    if data.get("n_residues") is not None:
        lengths["n_residues"] = int(data["n_residues"])
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        raise ValueError(f"Feature residue length mismatch: {lengths}")
    n_res = unique_lengths[0]
    if n_res <= 0:
        raise ValueError("Feature has zero residues")
    residue_name_1 = [AA3_TO_1.get(name, "X") for name in residue_names]
    return {
        "n_residues": n_res,
        "residue_names_3": residue_names,
        "residue_names_1": residue_name_1,
        "chain_ids": chain_ids,
        "residue_indices": residue_indices,
        "insertion_codes": insertion_codes,
    }


def _extract_one(
    feature_path: Path,
    output_root: Path,
    model,
    alphabet,
    model_name: str,
    overwrite: bool,
    max_len: int,
    overlap: int,
    repr_layers: list[int] | None,
    embedding_combine: str,
    save_layer_embeddings: bool,
    save_contacts: bool,
    contact_top_k: int,
    contact_min_score: float,
    contact_min_seq_sep: int,
    contact_bidirectional: bool,
) -> tuple[bool, str]:
    pdb_id = _feature_pdb_id(feature_path)
    out_path = output_root / pdb_id / f"{pdb_id}_esm2.pt"
    if out_path.exists() and not overwrite:
        return True, f"{pdb_id}: SKIP"

    table = _load_residue_table(feature_path)
    chain_positions: OrderedDict[str, list[int]] = OrderedDict()
    for idx, chain_id in enumerate(table["chain_ids"]):
        chain_positions.setdefault(chain_id, []).append(idx)

    embeddings: torch.Tensor | None = None
    layer_embeddings: torch.Tensor | None = None
    contact_edge_scores: dict[tuple[int, int], float] = {}
    chain_sequences: dict[str, str] = {}
    chain_window_info: dict[str, dict[str, Any]] = {}
    chain_boundaries: list[dict[str, Any]] = []
    layers = sorted(set(int(x) for x in (repr_layers or [getattr(model, "num_layers", -1)])))
    for chain_id, positions in chain_positions.items():
        sequence = "".join(table["residue_names_1"][idx] for idx in positions)
        chain_sequences[chain_id] = sequence
        emb, layer_emb, contact_windows, window_info = _run_esm2_windowed(
            model,
            alphabet,
            sequence,
            max_len=max_len,
            overlap=overlap,
            repr_layers=layers,
            embedding_combine=embedding_combine,
            return_contacts=save_contacts,
            save_layer_embeddings=save_layer_embeddings,
        )
        if emb.shape[0] != len(positions):
            raise RuntimeError(
                f"{pdb_id} chain {chain_id}: ESM rows {emb.shape[0]} != positions {len(positions)}"
            )
        if embeddings is None:
            embeddings = torch.empty(table["n_residues"], emb.shape[1], dtype=emb.dtype)
        embeddings[torch.tensor(positions, dtype=torch.long)] = emb
        if save_layer_embeddings and layer_emb is not None:
            if layer_embeddings is None:
                layer_embeddings = torch.empty(len(layers), table["n_residues"], layer_emb.shape[2], dtype=layer_emb.dtype)
            layer_embeddings[:, torch.tensor(positions, dtype=torch.long)] = layer_emb
        if save_contacts:
            for item in contact_windows:
                start = int(item["start"])
                contacts = item["contacts"]
                window_positions = positions[start : start + int(contacts.shape[0])]
                contact_edge_scores.update(
                    _contact_edges_from_map(
                        contacts,
                        window_positions,
                        top_k=contact_top_k,
                        min_score=contact_min_score,
                        min_seq_sep=contact_min_seq_sep,
                        bidirectional=contact_bidirectional,
                    )
                )
        chain_window_info[chain_id] = window_info
        chain_boundaries.append(
            {
                "chain_id": chain_id,
                "length": len(positions),
                "first_feature_row": positions[0],
                "last_feature_row": positions[-1],
                "n_windows": window_info["n_windows"],
                "windowed": window_info["windowed"],
            }
        )

    if embeddings is None:
        raise RuntimeError(f"{pdb_id}: no chains extracted")
    if embeddings.shape[0] != table["n_residues"]:
        raise RuntimeError(f"{pdb_id}: output length mismatch {embeddings.shape[0]} vs {table['n_residues']}")

    if contact_edge_scores:
        edge_items = sorted(contact_edge_scores.items())
        contact_edge_index = torch.tensor([[src, dst] for (src, dst), _ in edge_items], dtype=torch.long).t().contiguous()
        contact_scores = torch.tensor([score for _, score in edge_items], dtype=torch.float32)
    else:
        contact_edge_index = torch.empty((2, 0), dtype=torch.long)
        contact_scores = torch.empty((0,), dtype=torch.float32)

    result = {
        "embeddings": embeddings.contiguous(),
        "chain_id": table["chain_ids"],
        "chain_ids": table["chain_ids"],
        "residue_index": table["residue_indices"],
        "residue_indices": table["residue_indices"],
        "insertion_code": table["insertion_codes"],
        "insertion_codes": table["insertion_codes"],
        "residue_name_1": table["residue_names_1"],
        "residue_names_1": table["residue_names_1"],
        "residue_name_3": table["residue_names_3"],
        "residue_names_3": table["residue_names_3"],
        "sequence": "|".join(chain_sequences[c] for c in chain_positions.keys()),
        "chain_sequences": chain_sequences,
        "chain_boundaries": chain_boundaries,
        "chain_window_info": chain_window_info,
        "model_name": model_name,
        "repr_layer": int(max(layers)),
        "repr_layers": layers,
        "embedding_combine": embedding_combine,
        "embedding_dim": int(embeddings.shape[1]),
        "source_feature_path": str(feature_path),
        "pdb_id": pdb_id,
        "feature_n_residues": table["n_residues"],
        "alignment_source": "features/protein_v4/pt",
        "contact_edge_index": contact_edge_index,
        "contact_edge_scores": contact_scores,
        "contact_top_k": int(contact_top_k),
        "contact_min_score": float(contact_min_score),
        "contact_min_seq_sep": int(contact_min_seq_sep),
        "contact_bidirectional": bool(contact_bidirectional),
        "contact_source": "esm_return_contacts" if save_contacts else None,
    }
    if save_layer_embeddings and layer_embeddings is not None:
        result["layer_embeddings"] = layer_embeddings.contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / f".{pdb_id}_{uuid.uuid4().hex}.tmp"
    try:
        torch.save(result, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    windowed = sum(1 for info in chain_window_info.values() if info["windowed"])
    return True, (
        f"{pdb_id}: OK n={table['n_residues']} emb={tuple(embeddings.shape)} "
        f"layers={layers} contacts={int(contact_edge_index.shape[1])} chains={len(chain_positions)} windowed={windowed}"
    )


def _worker_entry(
    feature_paths: list[str],
    output_root: str,
    model_name: str,
    device_str: str,
    overwrite: bool,
    max_len: int,
    overlap: int,
    repr_layers: list[int] | None,
    embedding_combine: str,
    save_layer_embeddings: bool,
    save_contacts: bool,
    contact_top_k: int,
    contact_min_score: float,
    contact_min_seq_sep: int,
    contact_bidirectional: bool,
    cpu_threads: int,
    failure_tsv: str,
    tag: str,
) -> None:
    device = _resolve_device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif cpu_threads > 0:
        torch.set_num_threads(cpu_threads)

    model, alphabet = _load_model(model_name, device)
    failures: list[tuple[str, str]] = []
    total = len(feature_paths)
    for idx, feature_path_str in enumerate(feature_paths, 1):
        feature_path = Path(feature_path_str)
        try:
            ok, msg = _extract_one(
                feature_path,
                Path(output_root),
                model,
                alphabet,
                model_name,
                overwrite,
                max_len,
                overlap,
                repr_layers,
                embedding_combine,
                save_layer_embeddings,
                save_contacts,
                contact_top_k,
                contact_min_score,
                contact_min_seq_sep,
                contact_bidirectional,
            )
        except Exception as exc:
            ok = False
            msg = f"{_feature_pdb_id(feature_path)}: ERROR {exc!r}"
        print(f"[{tag} {idx}/{total}] {msg}", flush=True)
        if not ok:
            failures.append((_feature_pdb_id(feature_path), msg))

    if failures:
        path = Path(failure_tsv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            handle.write("pdb_id\terror\n")
            for pdb_id, err in failures:
                handle.write(f"{pdb_id}\t{err.replace(chr(9), ' ')}\n")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default=None, help="Comma-separated devices, e.g. 0,1 or cuda:0,cuda:1,cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--id-list", default=None, type=Path)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-len", type=int, default=ESM2_MAX_LEN)
    parser.add_argument("--overlap", type=int, default=ESM2_WINDOW_OVERLAP)
    parser.add_argument("--repr-layer", type=int, default=None, help="Backward-compatible single ESM layer selector")
    parser.add_argument("--repr-layers", default=None, help="Comma-separated ESM layers, e.g. 12,24,33")
    parser.add_argument("--embedding-combine", choices=["last", "mean", "concat"], default="last")
    parser.add_argument("--save-layer-embeddings", action="store_true")
    parser.add_argument("--save-contacts", action="store_true")
    parser.add_argument("--contact-top-k", type=int, default=16)
    parser.add_argument("--contact-min-score", type=float, default=0.05)
    parser.add_argument("--contact-min-seq-sep", type=int, default=6)
    parser.add_argument("--no-contact-bidirectional", action="store_true")
    parser.add_argument("--failure-dir", default=None, type=Path)
    args = parser.parse_args()

    repr_layers = _parse_int_list(args.repr_layers)
    if not repr_layers and args.repr_layer is not None:
        repr_layers = [int(args.repr_layer)]
    if not repr_layers:
        repr_layers = []

    feature_paths = _discover_features(args.features_root, args.id_list)
    if args.max is not None:
        feature_paths = feature_paths[: args.max]
    args.output_root.mkdir(parents=True, exist_ok=True)
    failure_dir = args.failure_dir or (args.output_root / "_failures")
    print(
        json.dumps(
            {
                "n_features": len(feature_paths),
                "model": args.model,
                "output_root": str(args.output_root),
                "devices": args.devices or args.device,
                "max_len": args.max_len,
                "overlap": args.overlap,
                "repr_layers": repr_layers or "final",
                "embedding_combine": args.embedding_combine,
                "save_layer_embeddings": args.save_layer_embeddings,
                "save_contacts": args.save_contacts,
                "contact_top_k": args.contact_top_k,
                "contact_min_score": args.contact_min_score,
                "contact_min_seq_sep": args.contact_min_seq_sep,
                "contact_bidirectional": not args.no_contact_bidirectional,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.devices:
        devices = _parse_devices(args.devices)
        if not devices:
            raise ValueError("--devices was provided but no valid devices were parsed")
        shards = [feature_paths[i:: len(devices)] for i in range(len(devices))]
        ctx = mp.get_context("spawn")
        procs: list[mp.Process] = []
        for rank, (device, shard) in enumerate(zip(devices, shards)):
            if not shard:
                continue
            proc = ctx.Process(
                target=_worker_entry,
                args=(
                    [str(path) for path in shard],
                    str(args.output_root),
                    args.model,
                    device,
                    args.overwrite,
                    args.max_len,
                    args.overlap,
                    repr_layers or None,
                    args.embedding_combine,
                    args.save_layer_embeddings,
                    args.save_contacts,
                    args.contact_top_k,
                    args.contact_min_score,
                    args.contact_min_seq_sep,
                    not args.no_contact_bidirectional,
                    args.cpu_threads,
                    str(failure_dir / f"failures_rank{rank}.tsv"),
                    f"rank{rank}:{device}",
                ),
            )
            proc.start()
            procs.append(proc)
        exit_codes = []
        for proc in procs:
            proc.join()
            exit_codes.append(proc.exitcode)
        return 0 if all(code == 0 for code in exit_codes) else 1

    _worker_entry(
        [str(path) for path in feature_paths],
        str(args.output_root),
        args.model,
        args.device,
        args.overwrite,
        args.max_len,
        args.overlap,
        repr_layers or None,
        args.embedding_combine,
        args.save_layer_embeddings,
        args.save_contacts,
        args.contact_top_k,
        args.contact_min_score,
        args.contact_min_seq_sep,
        not args.no_contact_bidirectional,
        args.cpu_threads,
        str(failure_dir / "failures.tsv"),
        args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
