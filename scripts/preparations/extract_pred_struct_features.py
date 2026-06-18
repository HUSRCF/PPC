#!/usr/bin/env python3
"""Extract residue features from predicted TmAlphaFold/AlphaFold structures."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import warnings
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


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
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}

TOPO_CODE = {"": 0, "G": 1, "M": 2, "F": 3, "m": 4, "H": 5, "1": 6, "2": 7, "I": 8, "O": 9, "L": 10, "S": 11}

FIELDNAMES = [
    "pdb_id",
    "chain_id",
    "seq_id",
    "uniprot_acc",
    "status",
    "source",
    "feature_path",
    "n_residues",
    "n_edges",
    "mean_plddt",
    "min_plddt",
    "dssp_status",
    "error",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict_bridge_rows(path: Path, allow_nonstrict: bool, max_rows: int | None) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(path):
        if not allow_nonstrict and str(row.get("strict_pass", "")).strip() not in {"1", "true", "True"}:
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _download_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        out[row["uniprot_acc"]] = row
    return out


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _open_text_maybe_gzip(path: Path):
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def _find_pred_paths(accession: str, raw_root: Path, manifest: dict[str, dict[str, str]]) -> dict[str, Path | None]:
    row = manifest.get(accession, {})
    acc_dir = raw_root / accession
    paths: dict[str, Path | None] = {
        "pdb": None,
        "tmdet": None,
        "cctop": None,
        "eval": None,
        "confidence": None,
    }
    for key, manifest_key in (
        ("pdb", "tmalphafold_pdb_path"),
        ("tmdet", "tmdet_xml_path"),
        ("cctop", "cctop_xml_path"),
        ("eval", "eval_xml_path"),
        ("confidence", "alphafold_confidence_path"),
    ):
        value = row.get(manifest_key)
        if value and Path(value).exists():
            paths[key] = Path(value)
    if paths["pdb"] is None:
        for candidate in (
            acc_dir / f"{accession}_tmalphafold.pdb",
            acc_dir / f"{accession}_alphafold.pdb",
        ):
            if candidate.exists():
                paths["pdb"] = candidate
                break
    for key, suffix in (("tmdet", "tmdet.xml"), ("cctop", "cctop.xml"), ("eval", "eval.xml"), ("confidence", "alphafold_confidence.json")):
        if paths[key] is None:
            candidate = acc_dir / f"{accession}_{suffix}"
            if candidate.exists():
                paths[key] = candidate
    return paths


def _parse_pdb_residues(path: Path) -> list[dict[str, Any]]:
    residues: OrderedDict[tuple[str, int, str], dict[str, Any]] = OrderedDict()
    with _open_text_maybe_gzip(path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip().upper()
            chain_id = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
            except ValueError:
                continue
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            residue = residues.setdefault(
                key,
                {
                    "chain_id": chain_id,
                    "resseq": resseq,
                    "icode": icode,
                    "residue_name_3": res_name,
                    "residue_name_1": AA3_TO_1.get(res_name, "X"),
                    "atoms": {},
                    "bfactors": [],
                },
            )
            residue["atoms"][atom_name] = (x, y, z)
            residue["bfactors"].append(bfactor)
    out: list[dict[str, Any]] = []
    for residue in residues.values():
        atoms = residue["atoms"]
        if "CA" in atoms:
            ca = atoms["CA"]
        else:
            ca = tuple(np.asarray(list(atoms.values()), dtype=np.float32).mean(axis=0).tolist())
        residue["ca_coord"] = ca
        residue["plddt"] = float(np.mean(residue["bfactors"])) if residue["bfactors"] else math.nan
        out.append(residue)
    return out


def _parse_range(value: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", value or "")
    if not match:
        return 1, 0
    return int(match.group(1)), int(match.group(2))


def _slice_residues_for_range(residues: list[dict[str, Any]], start: int, end: int, expected_len: int) -> list[dict[str, Any]]:
    if start > 0 and end >= start and end <= len(residues):
        sliced = residues[start - 1 : end]
        if len(sliced) == expected_len:
            return sliced
    if expected_len and len(residues) == expected_len:
        return residues
    if start > 0 and end >= start:
        return residues[max(0, start - 1) : min(len(residues), end)]
    return residues[:expected_len] if expected_len else residues


def _contact_graph(ca_coords: np.ndarray, cutoff: float, max_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_res = int(ca_coords.shape[0])
    if n_res == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    tree = cKDTree(ca_coords)
    pairs: list[tuple[int, int, float]] = []
    for i, coord in enumerate(ca_coords):
        neighbors = tree.query_ball_point(coord, r=cutoff)
        scored = []
        for j in neighbors:
            if i == j:
                continue
            dist = float(np.linalg.norm(ca_coords[i] - ca_coords[j]))
            scored.append((dist, j))
        scored.sort(key=lambda x: x[0])
        for dist, j in scored[:max_neighbors]:
            pairs.append((i, j, dist))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    edge_index = torch.tensor([[i for i, _, _ in pairs], [j for _, j, _ in pairs]], dtype=torch.long)
    edge_dist = torch.tensor([dist for _, _, dist in pairs], dtype=torch.float32)
    return edge_index, edge_dist


def _assign_regions(xml_path: Path | None, n_res: int) -> tuple[torch.Tensor, list[str]]:
    labels = [""] * n_res
    if xml_path is None or not xml_path.exists():
        return torch.zeros(n_res, dtype=torch.long), labels
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return torch.zeros(n_res, dtype=torch.long), labels
    for elem in root.iter():
        attrib = {key.lower(): value for key, value in elem.attrib.items()}
        beg = attrib.get("seq_beg") or attrib.get("seqbeg") or attrib.get("from") or attrib.get("begin") or attrib.get("start")
        end = attrib.get("seq_end") or attrib.get("seqend") or attrib.get("to") or attrib.get("end") or attrib.get("stop")
        typ = attrib.get("type") or attrib.get("loc") or attrib.get("location") or attrib.get("topology") or elem.tag.rsplit("}", 1)[-1]
        try:
            i1 = int(float(beg))
            i2 = int(float(end))
        except Exception:
            continue
        code = str(typ).strip()
        for idx in range(max(1, i1), min(n_res, i2) + 1):
            labels[idx - 1] = code
    return torch.tensor([TOPO_CODE.get(label, 0) for label in labels], dtype=torch.long), labels


def _membrane_half_thickness(tmdet_xml: Path | None, default: float = 15.0) -> float:
    if tmdet_xml is None or not tmdet_xml.exists():
        return default
    try:
        root = ET.parse(tmdet_xml).getroot()
    except Exception:
        return default
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1].upper()
        if tag != "NORMAL":
            continue
        for key in ("Z", "z"):
            if key in elem.attrib:
                try:
                    value = abs(float(elem.attrib[key]))
                    return value if value > 0 else default
                except ValueError:
                    pass
    return default


def _eval_flags(eval_xml: Path | None) -> dict[str, str]:
    if eval_xml is None or not eval_xml.exists():
        return {}
    try:
        root = ET.parse(eval_xml).getroot()
    except Exception:
        return {}
    flags: dict[str, str] = {}
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        text = (elem.text or "").strip()
        for key, value in elem.attrib.items():
            flags[f"{tag}.{key}"] = str(value)
        if text and len(text) < 80:
            flags[tag] = text
    return flags


def _run_dssp_optional(pdb_path: Path, residues: list[dict[str, Any]]) -> tuple[str, torch.Tensor, torch.Tensor, list[str]]:
    n_res = len(residues)
    rsa = torch.full((n_res,), float("nan"), dtype=torch.float32)
    sasa = torch.full((n_res,), float("nan"), dtype=torch.float32)
    ss = [""] * n_res
    try:
        from Bio.PDB import DSSP, PDBParser
    except Exception:
        return "NO_BIOPYTHON", rsa, sasa, ss
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pred", str(pdb_path))
        model = structure[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dssp = DSSP(model, str(pdb_path))
        by_key: dict[tuple[str, int, str], Any] = {}
        for key, value in dssp.property_dict.items():
            chain, residue_id = key
            hetflag, resseq, icode = residue_id
            if str(hetflag).strip() not in {"", " "}:
                continue
            by_key[(str(chain).strip() or "A", int(resseq), str(icode).strip())] = value
        for idx, residue in enumerate(residues):
            value = by_key.get((residue["chain_id"], residue["resseq"], residue["icode"]))
            if value is None:
                continue
            ss[idx] = str(value[2])
            rsa[idx] = float(value[3])
        return "OK", rsa, sasa, ss
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc!r}", rsa, sasa, ss


def _extract_one(
    row: dict[str, str],
    raw_root: Path,
    manifest: dict[str, dict[str, str]],
    output_root: Path,
    cutoff: float,
    max_neighbors: int,
    overwrite: bool,
) -> dict[str, Any]:
    out = {key: "" for key in FIELDNAMES}
    pdb_id = (row.get("pdb_id") or "").lower()
    chain_id = row.get("chain_id") or ""
    seq_id = row.get("seq_id") or f"{pdb_id}__{chain_id}"
    accession = row.get("uniprot_acc") or ""
    out.update({"pdb_id": pdb_id, "chain_id": chain_id, "seq_id": seq_id, "uniprot_acc": accession, "status": "ERROR"})
    try:
        paths = _find_pred_paths(accession, raw_root, manifest)
        pdb_path = paths["pdb"]
        if pdb_path is None or not pdb_path.exists():
            out["status"] = "NO_PDB"
            return out
        source = "tmalphafold" if str(pdb_path).endswith("_tmalphafold.pdb") else "alphafold"
        out["source"] = source
        residues_all = _parse_pdb_residues(pdb_path)
        start, end = _parse_range(row.get("uniprot_range", ""))
        expected_len = int(row.get("len_seq") or len(row.get("pdbtm_seq") or ""))
        residues = _slice_residues_for_range(residues_all, start, end, expected_len)
        if expected_len and len(residues) != expected_len:
            out["status"] = "LENGTH_MISMATCH"
            out["error"] = f"predicted slice length {len(residues)} != expected {expected_len}"
            return out
        ca_coords = np.asarray([res["ca_coord"] for res in residues], dtype=np.float32)
        plddt = torch.tensor([res["plddt"] for res in residues], dtype=torch.float32)
        edge_index, edge_dist = _contact_graph(ca_coords, cutoff=cutoff, max_neighbors=max_neighbors)
        topo_tmdet, topo_tmdet_raw = _assign_regions(paths["tmdet"], len(residues))
        topo_cctop, topo_cctop_raw = _assign_regions(paths["cctop"], len(residues))
        half_thickness = _membrane_half_thickness(paths["tmdet"])
        membrane_z = torch.tensor(ca_coords[:, 2], dtype=torch.float32)
        membrane_abs_z = membrane_z.abs()
        membrane_inside = (membrane_abs_z <= half_thickness).long()
        dssp_status, rsa, sasa, ss = _run_dssp_optional(pdb_path, residues)
        feature = {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "seq_id": seq_id,
            "uniprot_acc": accession,
            "uniprot_range": row.get("uniprot_range", ""),
            "source": source,
            "source_pdb_path": str(pdb_path),
            "pdbtm_seq": row.get("pdbtm_seq", ""),
            "residue_names_1": [res["residue_name_1"] for res in residues],
            "residue_names_3": [res["residue_name_3"] for res in residues],
            "pred_ca_coords": torch.tensor(ca_coords, dtype=torch.float32),
            "pred_plddt": plddt,
            "pred_contact_edge_index": edge_index,
            "pred_contact_edge_dist": edge_dist,
            "pred_contact_cutoff": float(cutoff),
            "pred_contact_max_neighbors": int(max_neighbors),
            "pred_membrane_z": membrane_z,
            "pred_membrane_abs_z": membrane_abs_z,
            "pred_membrane_inside": membrane_inside,
            "pred_membrane_half_thickness": float(half_thickness),
            "pred_tmdet_topology": topo_tmdet,
            "pred_tmdet_topology_raw": topo_tmdet_raw,
            "pred_cctop_topology": topo_cctop,
            "pred_cctop_topology_raw": topo_cctop_raw,
            "pred_dssp_ss": ss,
            "pred_dssp_rsa": rsa,
            "pred_dssp_sasa": sasa,
            "pred_dssp_status": dssp_status,
            "pred_eval_flags": _eval_flags(paths["eval"]),
            "feature_semantics": "sequence-plus-predicted-structure; no experimental/fixed PDB coordinates used as inputs",
        }
        out_path = output_root / "pt" / pdb_id / f"{seq_id}_predstruct.pt"
        if out_path.exists() and not overwrite:
            out.update({"status": "SKIP", "feature_path": str(out_path)})
            return out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
        torch.save(feature, tmp_path)
        os.replace(tmp_path, out_path)
        finite_plddt = plddt[torch.isfinite(plddt)]
        out.update(
            {
                "status": "OK",
                "feature_path": str(out_path),
                "n_residues": len(residues),
                "n_edges": int(edge_dist.numel()),
                "mean_plddt": float(finite_plddt.mean().item()) if finite_plddt.numel() else "",
                "min_plddt": float(finite_plddt.min().item()) if finite_plddt.numel() else "",
                "dssp_status": dssp_status,
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
        return out


def _write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, manifest_path)
    summary = {
        "n_total": len(rows),
        "statuses": Counter(row["status"] for row in rows),
        "sources": Counter(row["source"] for row in rows),
        "manifest": str(manifest_path),
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    tmp_json = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp_json, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-csv", type=Path, default=Path("features/pred_struct_bridge/bridge.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("features/tmalphafold_raw"))
    parser.add_argument("--download-manifest", type=Path, default=Path("features/tmalphafold_raw/manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("features/pred_struct_v1"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--allow-nonstrict", action="store_true")
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bridge_rows = _strict_bridge_rows(args.bridge_csv, allow_nonstrict=args.allow_nonstrict, max_rows=args.max)
    if not bridge_rows:
        raise SystemExit(f"No bridge rows selected from {args.bridge_csv}")
    manifest = _download_manifest(args.download_manifest)

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for row in bridge_rows:
            rows.append(_extract_one(row, args.raw_root, manifest, args.output_root, args.contact_cutoff, args.max_neighbors, args.overwrite))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_extract_one, row, args.raw_root, manifest, args.output_root, args.contact_cutoff, args.max_neighbors, args.overwrite)
                for row in bridge_rows
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (row["pdb_id"], row["chain_id"], row["seq_id"]))
    _write_manifest(rows, args.manifest or (args.output_root / "manifest.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
