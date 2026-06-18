#!/usr/bin/env python3
"""Build tutorial-style TMKit QC tables for PDBTM protein chains.

The PyPropel tutorial uses ``tmk.qc.obtain_single`` / ``tmk.qc.integrate`` to
create per-chain QC metric files named ``wb_{metric}_c.txt`` and an integrated
table.  This script recreates that table from the PPC artifacts we already use:
chain metadata/FASTA exported from feature files, raw PDBTM complex PDBs, and
PDBTM XML topology files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


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
    "ASX": "B",
    "GLX": "Z",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _file_chain(chain_id: str) -> str:
    return f"{chain_id}l" if chain_id.islower() else chain_id


def _feature_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = OrderedDict()
    current: str | None = None
    with path.open("rt", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records[current] = []
            elif current is not None:
                records[current].append(line)
    return {key: "".join(parts) for key, parts in records.items()}


def _read_chain_rows_from_metadata(metadata_csv: Path, fasta_path: Path) -> list[dict[str, Any]]:
    sequences = _read_fasta(fasta_path)
    rows: list[dict[str, Any]] = []
    with metadata_csv.open("rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = row["seq_id"]
            sequence = sequences.get(seq_id, "")
            chain_id = _as_text(row.get("chain_id"))
            pdb_id = _as_text(row.get("pdb_id")).lower()
            rows.append(
                {
                    "seq_id": seq_id,
                    "prot": pdb_id,
                    "chain": chain_id,
                    "file_chain": _file_chain(chain_id),
                    "chain_index": int(row.get("chain_index") or 0),
                    "prot_mark": f"{pdb_id}{chain_id}",
                    "sequence": sequence,
                    "len_seq": len(sequence) if sequence else int(row.get("n_residues") or 0),
                    "n_unknown": int(row.get("n_unknown") or sequence.count("X")),
                    "unknown_fraction": _safe_float(row.get("unknown_fraction")),
                    "feature_path": row.get("feature_path", ""),
                }
            )
    rows.sort(key=lambda item: (item["prot"], item["chain_index"], item["chain"]))
    return rows


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _feature_rows(features_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_path in sorted(features_root.glob("*/*_protein.pt")):
        pdb_id = _feature_id(feature_path).lower()
        data = _torch_load(feature_path)
        chain_ids = [_as_text(x) for x in data["chain_ids"]]
        residue_names = [_as_text(x).upper() for x in data["residue_names"]]
        chains: OrderedDict[str, list[str]] = OrderedDict()
        for chain_id, residue_name in zip(chain_ids, residue_names):
            chains.setdefault(chain_id, []).append(AA3_TO_1.get(residue_name, "X"))
        for chain_index, (chain_id, letters) in enumerate(chains.items()):
            sequence = "".join(letters)
            rows.append(
                {
                    "seq_id": f"{pdb_id}__{chain_id}",
                    "prot": pdb_id,
                    "chain": chain_id,
                    "file_chain": _file_chain(chain_id),
                    "chain_index": chain_index,
                    "prot_mark": f"{pdb_id}{chain_id}",
                    "sequence": sequence,
                    "len_seq": len(sequence),
                    "n_unknown": sequence.count("X"),
                    "unknown_fraction": sequence.count("X") / len(sequence) if sequence else math.nan,
                    "feature_path": str(feature_path),
                }
            )
    return rows


def _parse_compnd_molecules(compnd_text: str) -> dict[str, str]:
    chain_to_molecule: dict[str, str] = {}
    sections = re.split(r"\bMOL_ID\s*:", compnd_text, flags=re.IGNORECASE)
    for section in sections[1:]:
        fields = [field.strip() for field in section.split(";") if field.strip()]
        molecule = ""
        chains: list[str] = []
        for field in fields:
            if ":" not in field:
                continue
            key, value = field.split(":", 1)
            key = key.strip().upper()
            value = " ".join(value.split()).strip()
            if key == "MOLECULE":
                molecule = value.lower()
            elif key == "CHAIN":
                chains = [part.strip() for part in value.split(",") if part.strip()]
        if molecule:
            for chain in chains:
                chain_to_molecule[chain] = molecule
    return chain_to_molecule


def _parse_pdb_header(path: Path | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "pdb_exists": bool(path and path.exists()),
        "head": "",
        "desc": "",
        "met": "",
        "rez": math.nan,
        "bio_name_by_chain": {},
    }
    if path is None or not path.exists():
        return info

    title_lines: list[str] = []
    compnd_lines: list[str] = []
    expdta_lines: list[str] = []
    with path.open("rt", errors="replace") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "ATOM":
                break
            if rec == "HEADER":
                info["head"] = " ".join(line[10:50].split()).lower()
            elif rec == "TITLE":
                title_lines.append(line[10:80].strip())
            elif rec == "COMPND":
                compnd_lines.append(line[10:80].strip())
            elif rec == "EXPDTA":
                expdta_lines.append(line[10:80].strip())
            elif line.startswith("REMARK   2 RESOLUTION."):
                match = re.search(r"RESOLUTION\.\s*([0-9.]+)\s+ANGSTROMS", line, flags=re.IGNORECASE)
                if match:
                    info["rez"] = float(match.group(1))

    info["desc"] = " ".join(" ".join(title_lines).split()).lower()
    info["met"] = " ".join(" ".join(expdta_lines).split()).lower()
    info["bio_name_by_chain"] = _parse_compnd_molecules(" ".join(compnd_lines))
    try:
        from Bio.PDB.parse_pdb_header import parse_pdb_header

        bio_header = parse_pdb_header(str(path))
        if not info["met"]:
            info["met"] = _as_text(bio_header.get("structure_method")).lower()
        if math.isnan(info["rez"]) and bio_header.get("resolution") is not None:
            info["rez"] = float(bio_header["resolution"])
        if not info["head"]:
            info["head"] = _as_text(bio_header.get("head")).lower()
        if not info["desc"]:
            info["desc"] = _as_text(bio_header.get("name")).lower()
        for compound in bio_header.get("compound", {}).values():
            molecule = _as_text(compound.get("molecule")).lower()
            chain_text = _as_text(compound.get("chain"))
            if not molecule or not chain_text:
                continue
            for chain in [part.strip() for part in chain_text.split(",") if part.strip()]:
                info["bio_name_by_chain"].setdefault(chain, molecule)
                info["bio_name_by_chain"].setdefault(chain.upper(), molecule)
                info["bio_name_by_chain"].setdefault(chain.lower(), molecule)
    except Exception:
        pass
    if info["met"] in {"", "unknown"}:
        method_hint = f"{info['head']} {info['desc']}".lower()
        if "nmr" in method_hint:
            info["met"] = "solution nmr"
        elif "cryo-em" in method_hint or "electron microscopy" in method_hint or "electron cryo" in method_hint:
            info["met"] = "electron microscopy"
    return info


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _parse_xml(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"xml_exists": bool(path and path.exists()), "chains": {}}
    if path is None or not path.exists():
        return result
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        result["xml_error"] = repr(exc)
        return result
    chains: dict[str, dict[str, Any]] = {}
    for node in root.iter():
        if _strip_namespace(node.tag) != "CHAIN":
            continue
        chain_id = _as_text(node.attrib.get("CHAINID"))
        seq = ""
        for child in node:
            if _strip_namespace(child.tag) == "SEQ" and child.text:
                seq = "".join(child.text.split())
                break
        chains[chain_id] = {
            "mthm": int(node.attrib["NUM_TM"]) if str(node.attrib.get("NUM_TM", "")).isdigit() else math.nan,
            "tm_type": _as_text(node.attrib.get("TYPE")),
            "xml_seq": seq,
            "xml_len_seq": len(seq),
        }
    result["chains"] = chains
    return result


def _locate(root: Path, pdb_id: str, suffix: str) -> Path | None:
    candidates = (
        root / f"{pdb_id}{suffix}",
        root / f"{pdb_id.lower()}{suffix}",
        root / f"{pdb_id.upper()}{suffix}",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _build_integrated_rows(chain_rows: list[dict[str, Any]], pdb_root: Path, xml_root: Path) -> list[dict[str, Any]]:
    by_prot = Counter(row["prot"] for row in chain_rows)
    pdb_cache: dict[str, dict[str, Any]] = {}
    xml_cache: dict[str, dict[str, Any]] = {}
    integrated: list[dict[str, Any]] = []

    for row in chain_rows:
        prot = row["prot"]
        chain = row["chain"]
        if prot not in pdb_cache:
            pdb_cache[prot] = _parse_pdb_header(_locate(pdb_root, prot, ".pdb"))
        if prot not in xml_cache:
            xml_cache[prot] = _parse_xml(_locate(xml_root, prot, ".xml"))

        pdb_info = pdb_cache[prot]
        xml_info = xml_cache[prot]
        xml_chain = xml_info.get("chains", {}).get(chain, {})
        xml_seq = xml_chain.get("xml_seq", "")
        sequence = row["sequence"]
        integrated.append(
            {
                "prot": prot,
                "chain": chain,
                "file_chain": row["file_chain"],
                "prot_mark": row["prot_mark"],
                "seq_id": row["seq_id"],
                "seq": sequence,
                "len_seq": row["len_seq"],
                "nchain": by_prot[prot],
                "rez": pdb_info["rez"],
                "met": pdb_info["met"],
                "met1": pdb_info["met"],
                "mthm": xml_chain.get("mthm", math.nan),
                "tm_type": xml_chain.get("tm_type", ""),
                "bio_name": pdb_info["bio_name_by_chain"].get(chain, ""),
                "head": pdb_info["head"],
                "desc": pdb_info["desc"],
                "xml_seq": xml_seq,
                "xml_len_seq": xml_chain.get("xml_len_seq", math.nan),
                "seq_equal_xml": "" if not xml_seq or not sequence else int(sequence == xml_seq),
                "pdb_exists": int(bool(pdb_info["pdb_exists"])),
                "xml_exists": int(bool(xml_info["xml_exists"])),
                "feature_path": row.get("feature_path", ""),
                "n_unknown": row.get("n_unknown", 0),
                "unknown_fraction": row.get("unknown_fraction", math.nan),
            }
        )
    return integrated


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _write_metric_files(df: pd.DataFrame, output_dir: Path) -> None:
    metric_map = {
        "seq": ["prot", "chain", "prot_mark", "seq", "len_seq"],
        "nchain": ["prot", "chain", "prot_mark", "nchain"],
        "rez": ["prot", "chain", "prot_mark", "rez"],
        "met": ["prot", "chain", "prot_mark", "met"],
        "mthm": ["prot", "chain", "prot_mark", "mthm"],
        "bio_name": ["prot", "chain", "prot_mark", "bio_name"],
        "head": ["prot", "chain", "prot_mark", "head"],
        "desc": ["prot", "chain", "prot_mark", "desc"],
    }
    for metric, columns in metric_map.items():
        metric_df = df[columns]
        text = metric_df.to_csv(sep="\t", index=False, header=False, na_rep="")
        _atomic_write_text(output_dir / f"wb_{metric}_c.txt", text)


def _tutorial_filter(df: pd.DataFrame, max_len_seq: int | None) -> pd.DataFrame:
    mask = (
        (df["rez"] < 3.5)
        & (df["mthm"] >= 2)
        & (df["met1"] == "x-ray diffraction")
        & (df["nchain"] >= 2)
    )
    if max_len_seq is not None:
        mask &= df["len_seq"] < max_len_seq
    return df.loc[mask].copy()


def _write_excel(df: pd.DataFrame, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.xlsx"
        df.to_excel(tmp_path, index=False)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        _atomic_write_text(path.with_suffix(path.suffix + ".error.txt"), repr(exc) + "\n")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-metadata", type=Path, default=None)
    parser.add_argument("--chain-fasta", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--pdb-root", required=True, type=Path)
    parser.add_argument("--xml-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument(
        "--max-len-seq",
        type=int,
        default=1000,
        help="Maximum chain sequence length for dataset filtering. Use 0 to disable.",
    )
    parser.add_argument("--write-excel", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.chain_metadata and args.chain_fasta:
        chain_rows = _read_chain_rows_from_metadata(args.chain_metadata, args.chain_fasta)
        chain_source = "chain_metadata+fasta"
    elif args.features_root:
        chain_rows = _feature_rows(args.features_root)
        chain_source = "features_root"
    else:
        raise SystemExit("Provide either --chain-metadata plus --chain-fasta, or --features-root.")

    if args.max is not None:
        keep_prots = {row["prot"] for row in chain_rows[: args.max]}
        chain_rows = [row for row in chain_rows if row["prot"] in keep_prots]

    integrated_rows = _build_integrated_rows(chain_rows, args.pdb_root, args.xml_root)
    df = pd.DataFrame(integrated_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_metric_files(df, args.output_dir)
    _atomic_write_csv(df, args.output_dir / "integrate.csv")
    wrote_integrate_xlsx = _write_excel(df, args.output_dir / "integrate.xlsx") if args.write_excel else False

    max_len_seq = None if args.max_len_seq <= 0 else args.max_len_seq
    dataset = _tutorial_filter(df, max_len_seq=max_len_seq)
    _atomic_write_csv(dataset, args.output_dir / "dataset.csv")
    wrote_dataset_xlsx = _write_excel(dataset, args.output_dir / "dataset.xlsx") if args.write_excel else False

    seq_match_counts = df["seq_equal_xml"].replace("", "missing").value_counts(dropna=False).to_dict()
    summary = {
        "chain_source": chain_source,
        "chain_metadata": str(args.chain_metadata) if args.chain_metadata else "",
        "chain_fasta": str(args.chain_fasta) if args.chain_fasta else "",
        "features_root": str(args.features_root) if args.features_root else "",
        "pdb_root": str(args.pdb_root),
        "xml_root": str(args.xml_root),
        "output_dir": str(args.output_dir),
        "n_rows": int(len(df)),
        "n_proteins": int(df["prot"].nunique()),
        "n_missing_pdb": int((df["pdb_exists"] == 0).sum()),
        "n_missing_xml": int((df["xml_exists"] == 0).sum()),
        "n_missing_mthm": int(df["mthm"].isna().sum()),
        "n_missing_rez": int(df["rez"].isna().sum()),
        "seq_equal_xml_counts": {str(k): int(v) for k, v in seq_match_counts.items()},
        "max_len_seq": max_len_seq,
        "tutorial_filter": (
            "rez < 3.5 and mthm >= 2 and met1 == 'x-ray diffraction' and nchain >= 2"
            + (f" and len_seq < {max_len_seq}" if max_len_seq is not None else "")
        ),
        "dataset_rows": int(len(dataset)),
        "dataset_proteins": int(dataset["prot"].nunique()),
        "wrote_integrate_xlsx": bool(wrote_integrate_xlsx),
        "wrote_dataset_xlsx": bool(wrote_dataset_xlsx),
    }
    _atomic_write_text(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
