#!/usr/bin/env python3
"""Extract PLC-style complete features from predicted structures for PPC.

This adapter keeps the copied PLC feature modules intact and only handles PPC
bookkeeping: bridge rows, predicted-structure lookup, UniProt-range slicing,
and output field naming.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_features_v4 import (
    extract_complete_features_v4,
    get_spatial_scalar_feature_names_v4,
)


FIELDNAMES = [
    "pdb_id",
    "chain_id",
    "seq_id",
    "uniprot_acc",
    "status",
    "source",
    "feature_path",
    "n_residues",
    "mapping_method",
    "mapping_identity",
    "mapping_n_mismatch",
    "mapping_error",
    "error",
]

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


class LengthMismatchError(ValueError):
    pass


class AlignmentMismatchError(ValueError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict_bridge_rows(path: Path, allow_nonstrict: bool, max_rows: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
        acc = row.get("uniprot_acc") or row.get("accession") or ""
        if acc:
            out[acc] = row
    return out


def _find_pred_paths(accession: str, raw_root: Path, manifest: dict[str, dict[str, str]]) -> dict[str, Path | None]:
    row = manifest.get(accession, {})
    acc_dir = raw_root / accession
    paths: dict[str, Path | None] = {"pdb": None}
    for key in ("tmalphafold_pdb_path", "alphafold_pdb_path", "pdb_path"):
        value = row.get(key)
        if value and Path(value).exists():
            paths["pdb"] = Path(value)
            break
    if paths["pdb"] is None:
        for candidate in (
            acc_dir / f"{accession}_tmalphafold.pdb",
            acc_dir / f"{accession}_alphafold.pdb",
            acc_dir / f"{accession}_tmalphafold.pdb.gz",
            acc_dir / f"{accession}_alphafold.pdb.gz",
        ):
            if candidate.exists():
                paths["pdb"] = candidate
                break
    return paths


def _parse_range(text: str) -> tuple[int, int]:
    parts = str(text).replace(":", "-").split("-")
    if len(parts) < 2:
        raise ValueError(f"Bad UniProt range: {text!r}")
    start = int(parts[0])
    end = int(parts[1])
    if start <= 0 or end < start:
        raise ValueError(f"Bad UniProt range: {text!r}")
    return start, end


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def _write_sanitized_pdb(src: Path) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix="ppc_pred_complete_", suffix=".pdb", text=True)
    with os.fdopen(fd, "wt") as out, _open_text(src) as handle:
        out.write("HEADER    PREDICTED STRUCTURE                     01-JAN-00   PPC\n")
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM", "TER", "END")):
                out.write(line)
        out.write("END\n")
    return tmp_path


def _tensor_slice(value: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    index = torch.as_tensor(idx, dtype=torch.long)
    return value[index]


def _slice_feature_dict(full: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in full.items():
        if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == len(full["residue_names"]):
            out[key] = _tensor_slice(value, idx)
        elif isinstance(value, list) and len(value) == len(full["residue_names"]):
            out[key] = [value[int(i)] for i in idx]
        else:
            out[key] = value
    out["n_residues"] = int(len(idx))
    return out


def _parse_cigar(cigar: str) -> list[tuple[str, int]]:
    text = str(cigar or "").strip()
    if not text or text.lower() == "nan":
        raise ValueError(f"Bad alignment_cigar: {cigar!r}")
    pos = 0
    ops: list[tuple[str, int]] = []
    for match in re.finditer(r"(\d+)([=MXID])", text):
        if match.start() != pos:
            raise ValueError(f"Bad alignment_cigar: {cigar!r}")
        count = int(match.group(1))
        op = match.group(2)
        if count <= 0:
            raise ValueError(f"Bad alignment_cigar count: {cigar!r}")
        ops.append((op, count))
        pos = match.end()
    if pos != len(text) or not ops:
        raise ValueError(f"Bad alignment_cigar: {cigar!r}")
    return ops


def _mapped_uniprot_positions(start: int, expected_len: int, cigar: str) -> list[int | None]:
    """Map PDBTM/query residues to UniProt residue numbers using bridge CIGAR.

    The bridge builder aligns query=PDBTM sequence to target=UniProt window:
    M/X/= consume one query and one UniProt residue, D consumes query only,
    and I consumes UniProt only.
    """

    target_pos = start
    query_count = 0
    positions: list[int | None] = []
    for op, count in _parse_cigar(cigar):
        if op in {"M", "=", "X"}:
            positions.extend(range(target_pos, target_pos + count))
            query_count += count
            target_pos += count
        elif op == "D":
            positions.extend([None] * count)
            query_count += count
        elif op == "I":
            target_pos += count
        else:  # pragma: no cover - parser rejects this.
            raise ValueError(f"Unsupported CIGAR op: {op}")
    if query_count != expected_len:
        raise LengthMismatchError(f"CIGAR query length {query_count} != expected {expected_len}")
    return positions


def _blank_insertion_code(value: Any) -> bool:
    return str(value or "").strip() in {"", "."}


def _predicted_position_index(full: dict[str, Any]) -> dict[int, int]:
    residue_indices = full.get("residue_indices", [])
    insertion_codes = full.get("insertion_codes", [""] * len(residue_indices))
    pos_to_idx: dict[int, int] = {}
    for idx, (residue_index, insertion_code) in enumerate(zip(residue_indices, insertion_codes)):
        if not _blank_insertion_code(insertion_code):
            continue
        try:
            pos = int(residue_index)
        except (TypeError, ValueError):
            continue
        if pos in pos_to_idx:
            raise AlignmentMismatchError(f"Duplicate blank-insertion predicted residue number: {pos}")
        pos_to_idx[pos] = idx
    return pos_to_idx


def _select_alignment(
    full: dict[str, Any],
    start: int,
    end: int,
    expected_len: int,
    expected_seq: str,
    cigar: str,
) -> tuple[dict[str, Any], np.ndarray, list[int | None], dict[str, Any]]:
    uniprot_positions = _mapped_uniprot_positions(start, expected_len, cigar)
    if any(pos is None for pos in uniprot_positions):
        raise AlignmentMismatchError("CIGAR contains query-only deletion; cannot map every PDBTM residue to predicted structure")
    mapped_positions = [int(pos) for pos in uniprot_positions if pos is not None]
    if mapped_positions and (min(mapped_positions) < start or max(mapped_positions) > end):
        raise AlignmentMismatchError(
            f"CIGAR mapped positions outside UniProt range {start}-{end}: {min(mapped_positions)}-{max(mapped_positions)}"
        )

    pos_to_idx = _predicted_position_index(full)
    missing = [pos for pos in mapped_positions if pos not in pos_to_idx]
    if missing:
        preview = ",".join(str(pos) for pos in missing[:10])
        raise LengthMismatchError(f"Missing predicted residue numbers for UniProt positions: {preview}")
    idx = np.asarray([pos_to_idx[pos] for pos in mapped_positions], dtype=int)
    if len(idx) != expected_len:
        raise LengthMismatchError(f"predicted slice length {len(idx)} != expected {expected_len} for range {start}-{end}")
    sliced = _slice_feature_dict(full, idx)
    observed_seq = "".join(AA3_TO_1.get(str(name).upper(), "X") for name in sliced["residue_names"])
    if expected_seq and observed_seq != expected_seq:
        mismatch = next((i for i, (a, b) in enumerate(zip(observed_seq, expected_seq), start=1) if a != b), 0)
        raise AlignmentMismatchError(
            f"predicted slice sequence mismatch at position {mismatch}: observed={observed_seq[:30]}... expected={expected_seq[:30]}..."
        )
    return sliced, idx, uniprot_positions, {
        "method": "strict_uniprot_resnum",
        "identity": 1.0,
        "n_mismatch": 0,
        "mismatches": [],
        "strict_error": "",
    }


def _residue_sequence(full: dict[str, Any]) -> str:
    return "".join(AA3_TO_1.get(str(name).upper(), "X") for name in full["residue_names"])


def _best_window(seq: str, query: str) -> tuple[int, int]:
    n = len(query)
    if n <= 0 or len(seq) < n:
        return -1, -1
    best_matches = -1
    best_start = -1
    for start in range(0, len(seq) - n + 1):
        matches = sum(a == b for a, b in zip(seq[start : start + n], query))
        if matches > best_matches:
            best_matches = matches
            best_start = start
    return best_start, best_matches


def _sequence_mismatches(observed: str, expected: str, offset: int = 0) -> list[dict[str, Any]]:
    return [
        {"position": idx + 1, "pred_index": offset + idx, "expected": exp, "observed": obs}
        for idx, (obs, exp) in enumerate(zip(observed, expected))
        if obs != exp
    ]


def _select_predicted_sequence_window(
    full: dict[str, Any],
    start: int,
    expected_len: int,
    expected_seq: str,
    min_identity: float,
    strict_error: str,
) -> tuple[dict[str, Any], np.ndarray, list[int | None], dict[str, Any]]:
    pred_seq = _residue_sequence(full)
    if not expected_seq:
        raise AlignmentMismatchError("predicted-sequence fallback requires expected PDBTM sequence")
    if len(pred_seq) < expected_len:
        raise LengthMismatchError(f"predicted sequence length {len(pred_seq)} < expected {expected_len}")

    window_start = pred_seq.find(expected_seq)
    if window_start >= 0:
        matches = expected_len
        method = "pred_seq_exact_window"
    else:
        window_start, matches = _best_window(pred_seq, expected_seq)
        if window_start < 0:
            raise AlignmentMismatchError("no predicted sequence window available")
        method = "pred_seq_near_identity"

    identity = float(matches / max(1, expected_len))
    if identity < min_identity:
        raise AlignmentMismatchError(
            f"best predicted sequence window identity {identity:.6f} < threshold {min_identity:.6f}"
        )

    idx = np.arange(window_start, window_start + expected_len, dtype=int)
    sliced = _slice_feature_dict(full, idx)
    observed_seq = _residue_sequence(sliced)
    mismatches = _sequence_mismatches(observed_seq, expected_seq, offset=window_start)
    first_residue_index = None
    try:
        first_residue_index = int(sliced["residue_indices"][0])
    except (IndexError, TypeError, ValueError):
        pass
    if method == "pred_seq_exact_window" and first_residue_index is not None and first_residue_index != start:
        method = "pred_seq_exact_resnum_offset"

    # For sequence-window fallback, UniProt residue numbers are intentionally
    # provenance-only. They may not equal predicted PDB residue numbering.
    uniprot_positions = list(range(start, start + expected_len))
    return sliced, idx, uniprot_positions, {
        "method": method,
        "identity": identity,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches,
        "strict_error": strict_error,
        "pred_window_start0": int(window_start),
        "pred_window_start1": int(window_start + 1),
        "pred_window_end1": int(window_start + expected_len),
        "pred_first_residue_index": first_residue_index,
    }


def _load_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, dict):
        return {str(_to_builtin(key)): _to_builtin(item) for key, item in value.items()}
    return value


def _extract_one(
    row: dict[str, str],
    raw_root: Path,
    download_manifest: dict[str, dict[str, str]],
    output_root: Path,
    normalization_stats: dict[str, Any] | None,
    overwrite: bool,
    allow_pred_seq_window_fallback: bool,
    min_pred_seq_window_identity: float,
) -> dict[str, Any]:
    out = {key: "" for key in FIELDNAMES}
    pdb_id = (row.get("pdb_id") or "").lower()
    chain_id = row.get("chain_id") or ""
    seq_id = row.get("seq_id") or f"{pdb_id}__{chain_id}"
    accession = row.get("uniprot_acc") or ""
    out.update({"pdb_id": pdb_id, "chain_id": chain_id, "seq_id": seq_id, "uniprot_acc": accession, "status": "ERROR"})

    out_path = output_root / "pt" / pdb_id / f"{seq_id}_predcomplete.pt"
    if out_path.exists() and not overwrite:
        out.update({"status": "SKIP", "feature_path": str(out_path)})
        return out

    tmp_pdb: str | None = None
    try:
        paths = _find_pred_paths(accession, raw_root, download_manifest)
        pdb_path = paths["pdb"]
        if pdb_path is None or not pdb_path.exists():
            out["status"] = "NO_PDB"
            return out
        source = "tmalphafold" if str(pdb_path).endswith("_tmalphafold.pdb") else "alphafold"
        if str(pdb_path).endswith(".gz"):
            source = source.replace(".gz", "")
        out["source"] = source

        tmp_pdb = _write_sanitized_pdb(pdb_path)
        full = extract_complete_features_v4(Path(tmp_pdb), mode="complex", normalization_stats=normalization_stats)

        start, end = _parse_range(row.get("uniprot_range", ""))
        expected_len = int(row.get("len_seq") or len(row.get("pdbtm_seq") or ""))
        expected_seq = row.get("pdbtm_seq", "")
        try:
            sliced, _idx, uniprot_positions, mapping = _select_alignment(
                full,
                start=start,
                end=end,
                expected_len=expected_len,
                expected_seq=expected_seq,
                cigar=row.get("alignment_cigar", ""),
            )
        except (LengthMismatchError, AlignmentMismatchError) as strict_exc:
            if not allow_pred_seq_window_fallback:
                raise
            sliced, _idx, uniprot_positions, mapping = _select_predicted_sequence_window(
                full,
                start=start,
                expected_len=expected_len,
                expected_seq=expected_seq,
                min_identity=min_pred_seq_window_identity,
                strict_error=str(strict_exc),
            )

        raw_scalar = sliced["spatial_scalar_raw_features"]
        norm_scalar = sliced["spatial_scalar_features"]
        feature = {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "seq_id": seq_id,
            "uniprot_acc": accession,
            "uniprot_range": row.get("uniprot_range", ""),
            "alignment_cigar": row.get("alignment_cigar", ""),
            "source": source,
            "source_pdb_path": str(pdb_path),
            "pdbtm_seq": row.get("pdbtm_seq", ""),
            "pred_sasa_features": raw_scalar[:, 0:6].clone(),
            "pred_surface_features": raw_scalar[:, 6:20].clone(),
            "pred_structural_geometry_features": raw_scalar[:, 20:46].clone(),
            "pred_enhanced_features": raw_scalar[:, 46:58].clone(),
            "pred_physchem_features": sliced["physchem_features"],
            "pred_spatial_scalar_raw_features": raw_scalar,
            "pred_spatial_scalar_features": norm_scalar,
            "pred_spatial_vector_features": sliced["spatial_vector_features"],
            "pred_ca_coords": sliced["ca_coords"],
            "residue_names_1": _to_builtin(list(row.get("pdbtm_seq", ""))),
            "residue_names_3": _to_builtin(sliced["residue_names"]),
            "residue_indices": _to_builtin(sliced["residue_indices"]),
            "insertion_codes": _to_builtin(sliced.get("insertion_codes", [""] * expected_len)),
            "uniprot_positions": _to_builtin(uniprot_positions),
            "pred_mapping_method": _to_builtin(mapping["method"]),
            "pred_mapping_identity": _to_builtin(mapping["identity"]),
            "pred_mapping_n_mismatch": _to_builtin(mapping["n_mismatch"]),
            "pred_mapping_mismatches": _to_builtin(mapping["mismatches"]),
            "pred_mapping_strict_error": _to_builtin(mapping["strict_error"]),
            "pred_mapping_details": _to_builtin(mapping),
            "pred_chain_ids": _to_builtin(sliced["chain_ids"]),
            "spatial_scalar_feature_names": _to_builtin(sliced.get("spatial_scalar_feature_names", get_spatial_scalar_feature_names_v4())),
            "normalization_stats": str(normalization_stats.get("_path", "")) if normalization_stats else "",
            "feature_semantics": "sequence-plus-predicted-structure PLC-v4 complete features; no experimental/fixed PDB coordinates used as inputs",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
        torch.save(feature, tmp_path)
        os.replace(tmp_path, out_path)
        out.update(
            {
                "status": "OK",
                "feature_path": str(out_path),
                "n_residues": expected_len,
                "mapping_method": mapping["method"],
                "mapping_identity": f"{float(mapping['identity']):.8f}",
                "mapping_n_mismatch": int(mapping["n_mismatch"]),
                "mapping_error": mapping["strict_error"],
            }
        )
        return out
    except LengthMismatchError as exc:
        out["status"] = "LENGTH_MISMATCH"
        out["error"] = str(exc)
        return out
    except AlignmentMismatchError as exc:
        out["status"] = "ALIGNMENT_MISMATCH"
        out["error"] = str(exc)
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
        return out
    finally:
        if tmp_pdb:
            try:
                os.unlink(tmp_pdb)
            except OSError:
                pass


def _write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, manifest_path)
    summary = {
        "manifest": str(manifest_path),
        "n_total": len(rows),
        "statuses": Counter(row["status"] for row in rows),
        "sources": Counter(row["source"] for row in rows),
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
    parser.add_argument("--output-root", type=Path, default=Path("features/pred_struct_complete_v1"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--allow-nonstrict", action="store_true")
    parser.add_argument("--allow-pred-seq-window-fallback", action="store_true")
    parser.add_argument("--min-pred-seq-window-identity", type=float, default=0.99)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows_in = _strict_bridge_rows(args.bridge_csv, allow_nonstrict=args.allow_nonstrict, max_rows=args.max)
    if not rows_in:
        raise SystemExit(f"No bridge rows selected from {args.bridge_csv}")
    download_manifest = _download_manifest(args.download_manifest)
    normalization_stats = _load_stats(args.normalization_stats)
    if normalization_stats is not None:
        normalization_stats["_path"] = str(args.normalization_stats)

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for row in rows_in:
            rows.append(
                _extract_one(
                    row,
                    args.raw_root,
                    download_manifest,
                    args.output_root,
                    normalization_stats,
                    args.overwrite,
                    args.allow_pred_seq_window_fallback,
                    args.min_pred_seq_window_identity,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _extract_one,
                    row,
                    args.raw_root,
                    download_manifest,
                    args.output_root,
                    normalization_stats,
                    args.overwrite,
                    args.allow_pred_seq_window_fallback,
                    args.min_pred_seq_window_identity,
                )
                for row in rows_in
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (row["pdb_id"], row["chain_id"], row["seq_id"]))
    _write_manifest(rows, args.manifest or (args.output_root / "manifest.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
