#!/usr/bin/env python3
"""Build a strict PDBTM-chain to UniProt bridge table for predicted structures.

The table is intended to connect PDBTM residue labels to sequence-derived
AlphaFold/TmAlphaFold structures without using fixed or experimental PDB
coordinates as model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PDBe_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
TMALPHAFOLD_ENTRY_URL = "https://tmalphafold.ttk.hu/entry/{accession}"

FIELDNAMES = [
    "pdb_id",
    "chain_id",
    "seq_id",
    "pdbtm_seq",
    "len_seq",
    "uniprot_acc",
    "uniprot_id",
    "uniprot_range",
    "alignment_cigar",
    "coverage",
    "identity",
    "pdbe_coverage",
    "pdbe_identity",
    "n_equal",
    "n_mismatch",
    "n_insert",
    "n_delete",
    "max_indel_run",
    "strict_pass",
    "status",
    "error",
    "tmalphafold_entry_url",
    "alphafold_api_url",
    "alphafold_pdb_url",
    "source_dataset_row",
]


def _read_dataset(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            seq = (row.get("seq") or row.get("pdbtm_seq") or "").replace(" ", "").strip().upper()
            pdb_id = (row.get("prot") or row.get("pdb_id") or "").strip().lower()
            chain_id = (row.get("chain") or row.get("chain_id") or "").strip()
            if not pdb_id or not chain_id or not seq:
                continue
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "seq_id": row.get("seq_id") or f"{pdb_id}__{chain_id}",
                    "pdbtm_seq": seq,
                    "len_seq": len(seq),
                    "source_dataset_row": idx + 2,
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _request_json(url: str, timeout: float, retries: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PPC-pred-struct/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            return json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_exc!r}")


def _request_text(url: str, timeout: float, retries: int = 3) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PPC-pred-struct/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            return data.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_exc!r}")


def _cache_json(path: Path, url: str, timeout: float, offline: bool = False) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    if offline:
        raise FileNotFoundError(f"Missing cache file in offline mode: {path}")
    data = _request_json(url, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(json.dumps(data, sort_keys=True))
    os.replace(tmp_path, path)
    return data


def _cache_text(path: Path, url: str, timeout: float, offline: bool = False) -> str:
    if path.exists():
        return path.read_text()
    if offline:
        raise FileNotFoundError(f"Missing cache file in offline mode: {path}")
    text = _request_text(url, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(text)
    os.replace(tmp_path, path)
    return text


def _parse_fasta(text: str) -> str:
    return "".join(line.strip() for line in text.splitlines() if line and not line.startswith(">")).upper()


def _fetch_pdbe_mapping(pdb_id: str, cache_root: Path, timeout: float, offline: bool) -> dict[str, Any]:
    url = PDBe_URL.format(pdb_id=pdb_id.lower())
    return _cache_json(cache_root / f"{pdb_id.lower()}.json", url, timeout=timeout, offline=offline)


def _fetch_uniprot_sequence(accession: str, cache_root: Path, timeout: float, offline: bool) -> str:
    url = UNIPROT_FASTA_URL.format(accession=accession)
    text = _cache_text(cache_root / f"{accession}.fasta", url, timeout=timeout, offline=offline)
    return _parse_fasta(text)


def _fetch_alphafold_entry(accession: str, cache_root: Path, timeout: float, offline: bool) -> dict[str, Any] | None:
    url = ALPHAFOLD_API_URL.format(accession=accession)
    try:
        data = _cache_json(cache_root / f"{accession}.json", url, timeout=timeout, offline=offline)
    except Exception:
        return None
    if isinstance(data, list):
        for item in data:
            if str(item.get("uniprotAccession", "")).upper() == accession.upper():
                return item
        return data[0] if data else None
    return data if isinstance(data, dict) else None


def _mapping_candidates(pdbe_data: dict[str, Any], pdb_id: str, chain_id: str) -> list[dict[str, Any]]:
    root = pdbe_data.get(pdb_id.lower()) or pdbe_data.get(pdb_id.upper()) or {}
    uniprot_block = root.get("UniProt", {})
    candidates: list[dict[str, Any]] = []
    for accession, acc_data in uniprot_block.items():
        for mapping in acc_data.get("mappings", []):
            if str(mapping.get("chain_id", "")).strip() != chain_id:
                continue
            candidates.append(
                {
                    "uniprot_acc": accession,
                    "uniprot_id": acc_data.get("identifier") or acc_data.get("name") or "",
                    "unp_start": int(mapping.get("unp_start") or 0),
                    "unp_end": int(mapping.get("unp_end") or 0),
                    "pdbe_identity": float(mapping.get("identity") or 0.0),
                    "pdbe_coverage": float(mapping.get("coverage") or 0.0),
                }
            )
    candidates.sort(key=lambda x: (x["pdbe_identity"], x["pdbe_coverage"], x["unp_end"] - x["unp_start"]), reverse=True)
    return candidates


def _collapse_ops(ops: list[tuple[str, int]]) -> str:
    if not ops:
        return "0M"
    merged: list[tuple[str, int]] = []
    for op, count in ops:
        if count <= 0:
            continue
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + count)
        else:
            merged.append((op, count))
    return "".join(f"{count}{op}" for op, count in merged) or "0M"


def _align_strings(query: str, target: str) -> dict[str, Any]:
    if query == target:
        n = len(query)
        return {
            "alignment_cigar": f"{n}M",
            "coverage": 1.0 if n else 0.0,
            "identity": 1.0 if n else 0.0,
            "n_equal": n,
            "n_mismatch": 0,
            "n_insert": 0,
            "n_delete": 0,
            "max_indel_run": 0,
        }

    matcher = SequenceMatcher(a=query, b=target, autojunk=False)
    ops: list[tuple[str, int]] = []
    n_equal = 0
    n_mismatch = 0
    n_insert = 0
    n_delete = 0
    max_indel = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        q_len = i2 - i1
        t_len = j2 - j1
        if tag == "equal":
            ops.append(("M", q_len))
            n_equal += q_len
        elif tag == "replace":
            shared = min(q_len, t_len)
            if shared:
                ops.append(("X", shared))
                n_mismatch += shared
            if q_len > shared:
                ops.append(("D", q_len - shared))
                n_delete += q_len - shared
                max_indel = max(max_indel, q_len - shared)
            if t_len > shared:
                ops.append(("I", t_len - shared))
                n_insert += t_len - shared
                max_indel = max(max_indel, t_len - shared)
        elif tag == "delete":
            ops.append(("D", q_len))
            n_delete += q_len
            max_indel = max(max_indel, q_len)
        elif tag == "insert":
            ops.append(("I", t_len))
            n_insert += t_len
            max_indel = max(max_indel, t_len)
    denom = max(1, n_equal + n_mismatch + n_insert + n_delete)
    coverage = (n_equal + n_mismatch + n_delete) / max(1, len(query))
    return {
        "alignment_cigar": _collapse_ops(ops),
        "coverage": min(1.0, float(coverage)),
        "identity": float(n_equal / denom),
        "n_equal": n_equal,
        "n_mismatch": n_mismatch,
        "n_insert": n_insert,
        "n_delete": n_delete,
        "max_indel_run": max_indel,
    }


def _find_query_window(query: str, uniprot_sequence: str, pdbe_start: int, pdbe_end: int) -> tuple[str, int, int, str]:
    """Return the UniProt slice that should be aligned to the PDBTM sequence.

    PDBe mappings may include terminal residues that are present in UniProt but
    not in the PDBTM chain sequence.  If the PDBTM sequence is an exact
    contiguous substring of the UniProt sequence, use that precise range.
    """

    starts: list[int] = []
    search_at = 0
    while True:
        pos = uniprot_sequence.find(query, search_at)
        if pos < 0:
            break
        starts.append(pos)
        search_at = pos + 1
    if starts:
        target_start0 = max(0, pdbe_start - 1)
        best = min(starts, key=lambda pos: abs(pos - target_start0))
        return query, best + 1, best + len(query), "exact_substring"

    if pdbe_start > 0 and pdbe_end >= pdbe_start:
        target = uniprot_sequence[pdbe_start - 1 : pdbe_end]
        return target, pdbe_start, pdbe_end, "pdbe_range"
    return uniprot_sequence, 1, len(uniprot_sequence), "full_uniprot"


def _process_row(
    row: dict[str, Any],
    pdbe_cache: Path,
    uniprot_cache: Path,
    alphafold_cache: Path,
    timeout: float,
    offline: bool,
    identity_threshold: float,
    coverage_threshold: float,
    max_indel_run: int,
    skip_alphafold_api: bool,
) -> dict[str, Any]:
    out = {key: "" for key in FIELDNAMES}
    out.update(
        {
            "pdb_id": row["pdb_id"],
            "chain_id": row["chain_id"],
            "seq_id": row["seq_id"],
            "pdbtm_seq": row["pdbtm_seq"],
            "len_seq": row["len_seq"],
            "source_dataset_row": row["source_dataset_row"],
            "strict_pass": 0,
            "status": "ERROR",
        }
    )
    try:
        pdbe_data = _fetch_pdbe_mapping(row["pdb_id"], pdbe_cache, timeout=timeout, offline=offline)
        candidates = _mapping_candidates(pdbe_data, row["pdb_id"], row["chain_id"])
        if not candidates:
            out["status"] = "NO_PDBE_MAPPING"
            return out
        cand = candidates[0]
        accession = cand["uniprot_acc"]
        out.update(
            {
                "uniprot_acc": accession,
                "uniprot_id": cand["uniprot_id"],
                "pdbe_identity": cand["pdbe_identity"],
                "pdbe_coverage": cand["pdbe_coverage"],
                "tmalphafold_entry_url": TMALPHAFOLD_ENTRY_URL.format(accession=accession),
                "alphafold_api_url": ALPHAFOLD_API_URL.format(accession=accession),
            }
        )
        sequence = _fetch_uniprot_sequence(accession, uniprot_cache, timeout=timeout, offline=offline)
        start = int(cand["unp_start"])
        end = int(cand["unp_end"])
        target, actual_start, actual_end, window_status = _find_query_window(row["pdbtm_seq"], sequence, start, end)
        out["uniprot_range"] = f"{actual_start}-{actual_end}"
        aln = _align_strings(row["pdbtm_seq"], target)
        out.update(aln)
        if not skip_alphafold_api:
            af_entry = _fetch_alphafold_entry(accession, alphafold_cache, timeout=timeout, offline=offline)
            if af_entry:
                out["alphafold_pdb_url"] = af_entry.get("pdbUrl", "")
        strict = (
            float(out["identity"]) >= identity_threshold
            and float(out["coverage"]) >= coverage_threshold
            and int(out["max_indel_run"]) <= max_indel_run
        )
        out["strict_pass"] = int(strict)
        out["status"] = "OK" if strict else f"ALIGNMENT_FAIL:{window_status}"
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
        return out


def _process_prefetched_row(
    row: dict[str, Any],
    pdbe_by_pdb: dict[str, dict[str, Any]],
    pdbe_errors: dict[str, str],
    sequence_by_acc: dict[str, str],
    sequence_errors: dict[str, str],
    identity_threshold: float,
    coverage_threshold: float,
    max_indel_run: int,
) -> dict[str, Any]:
    out = {key: "" for key in FIELDNAMES}
    out.update(
        {
            "pdb_id": row["pdb_id"],
            "chain_id": row["chain_id"],
            "seq_id": row["seq_id"],
            "pdbtm_seq": row["pdbtm_seq"],
            "len_seq": row["len_seq"],
            "source_dataset_row": row["source_dataset_row"],
            "strict_pass": 0,
            "status": "ERROR",
        }
    )
    pdb_id = row["pdb_id"]
    if pdb_id in pdbe_errors:
        out["error"] = pdbe_errors[pdb_id]
        return out
    pdbe_data = pdbe_by_pdb.get(pdb_id)
    if not pdbe_data:
        out["status"] = "NO_PDBE_MAPPING"
        return out
    candidates = _mapping_candidates(pdbe_data, row["pdb_id"], row["chain_id"])
    if not candidates:
        out["status"] = "NO_PDBE_MAPPING"
        return out

    cand = candidates[0]
    accession = cand["uniprot_acc"]
    out.update(
        {
            "uniprot_acc": accession,
            "uniprot_id": cand["uniprot_id"],
            "pdbe_identity": cand["pdbe_identity"],
            "pdbe_coverage": cand["pdbe_coverage"],
            "tmalphafold_entry_url": TMALPHAFOLD_ENTRY_URL.format(accession=accession),
            "alphafold_api_url": ALPHAFOLD_API_URL.format(accession=accession),
        }
    )
    if accession in sequence_errors:
        out["error"] = sequence_errors[accession]
        return out
    sequence = sequence_by_acc.get(accession)
    if not sequence:
        out["status"] = "NO_UNIPROT_SEQUENCE"
        return out

    start = int(cand["unp_start"])
    end = int(cand["unp_end"])
    target, actual_start, actual_end, window_status = _find_query_window(row["pdbtm_seq"], sequence, start, end)
    out["uniprot_range"] = f"{actual_start}-{actual_end}"
    aln = _align_strings(row["pdbtm_seq"], target)
    out.update(aln)
    strict = (
        float(out["identity"]) >= identity_threshold
        and float(out["coverage"]) >= coverage_threshold
        and int(out["max_indel_run"]) <= max_indel_run
    )
    out["strict_pass"] = int(strict)
    out["status"] = "OK" if strict else f"ALIGNMENT_FAIL:{window_status}"
    return out


def _prefetch_pdbe(
    pdb_ids: list[str],
    pdbe_cache: Path,
    timeout: float,
    offline: bool,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    data_by_pdb: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    def task(pdb_id: str) -> tuple[str, dict[str, Any] | None, str]:
        try:
            return pdb_id, _fetch_pdbe_mapping(pdb_id, pdbe_cache, timeout=timeout, offline=offline), ""
        except Exception as exc:  # noqa: BLE001
            return pdb_id, None, repr(exc)

    if workers <= 1:
        iterator = (task(pdb_id) for pdb_id in pdb_ids)
        for idx, (pdb_id, data, error) in enumerate(iterator, 1):
            if idx % 250 == 0 or idx == len(pdb_ids):
                print(f"[bridge] PDBe {idx}/{len(pdb_ids)}", flush=True)
            if error:
                errors[pdb_id] = error
            elif data is not None:
                data_by_pdb[pdb_id] = data
        return data_by_pdb, errors

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task, pdb_id) for pdb_id in pdb_ids]
        for future in as_completed(futures):
            done += 1
            if done % 250 == 0 or done == len(futures):
                print(f"[bridge] PDBe {done}/{len(futures)}", flush=True)
            pdb_id, data, error = future.result()
            if error:
                errors[pdb_id] = error
            elif data is not None:
                data_by_pdb[pdb_id] = data
    return data_by_pdb, errors


def _prefetch_uniprot(
    accessions: list[str],
    uniprot_cache: Path,
    timeout: float,
    offline: bool,
    workers: int,
) -> tuple[dict[str, str], dict[str, str]]:
    sequence_by_acc: dict[str, str] = {}
    errors: dict[str, str] = {}

    def task(accession: str) -> tuple[str, str, str]:
        try:
            return accession, _fetch_uniprot_sequence(accession, uniprot_cache, timeout=timeout, offline=offline), ""
        except Exception as exc:  # noqa: BLE001
            return accession, "", repr(exc)

    if workers <= 1:
        for idx, (accession, sequence, error) in enumerate((task(acc) for acc in accessions), 1):
            if idx % 250 == 0 or idx == len(accessions):
                print(f"[bridge] UniProt {idx}/{len(accessions)}", flush=True)
            if error:
                errors[accession] = error
            else:
                sequence_by_acc[accession] = sequence
        return sequence_by_acc, errors

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task, accession) for accession in accessions]
        for future in as_completed(futures):
            done += 1
            if done % 250 == 0 or done == len(futures):
                print(f"[bridge] UniProt {done}/{len(futures)}", flush=True)
            accession, sequence, error = future.result()
            if error:
                errors[accession] = error
            else:
                sequence_by_acc[accession] = sequence
    return sequence_by_acc, errors


def _build_rows_staged(
    rows_in: list[dict[str, Any]],
    pdbe_cache: Path,
    uniprot_cache: Path,
    timeout: float,
    offline: bool,
    identity_threshold: float,
    coverage_threshold: float,
    max_indel_run: int,
    workers: int,
) -> list[dict[str, Any]]:
    pdb_ids = sorted({row["pdb_id"] for row in rows_in})
    print(f"[bridge] unique PDB IDs: {len(pdb_ids)} from chain rows: {len(rows_in)}", flush=True)
    pdbe_by_pdb, pdbe_errors = _prefetch_pdbe(pdb_ids, pdbe_cache, timeout, offline, workers)

    accessions: set[str] = set()
    for row in rows_in:
        data = pdbe_by_pdb.get(row["pdb_id"])
        if not data:
            continue
        candidates = _mapping_candidates(data, row["pdb_id"], row["chain_id"])
        if candidates:
            accessions.add(candidates[0]["uniprot_acc"])
    accession_list = sorted(accessions)
    print(f"[bridge] unique UniProt accessions: {len(accession_list)}", flush=True)
    sequence_by_acc, sequence_errors = _prefetch_uniprot(accession_list, uniprot_cache, timeout, offline, workers)

    print("[bridge] building chain-level alignments", flush=True)
    return [
        _process_prefetched_row(
            row,
            pdbe_by_pdb,
            pdbe_errors,
            sequence_by_acc,
            sequence_errors,
            identity_threshold,
            coverage_threshold,
            max_indel_run,
        )
        for row in rows_in
    ]


def _write_outputs(rows: list[dict[str, Any]], output_csv: Path, summary_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp_csv.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_csv, output_csv)
    summary = {
        "n_total": len(rows),
        "statuses": Counter(row["status"] for row in rows),
        "n_strict_pass": sum(int(row.get("strict_pass") or 0) for row in rows),
        "n_unique_uniprot": len({row["uniprot_acc"] for row in rows if row.get("uniprot_acc")}),
        "csv_path": str(output_csv),
    }
    tmp_json = summary_json.with_suffix(summary_json.suffix + ".tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp_json, summary_json)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/qc/tmk_no_len_limit/dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("features/pred_struct_bridge"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--identity-threshold", type=float, default=0.99)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--max-indel-run", type=int, default=10)
    parser.add_argument(
        "--skip-alphafold-api",
        action="store_true",
        help="Do not query AlphaFold DB during bridge building; download fallback will query it later if needed.",
    )
    parser.add_argument(
        "--row-level-fetch",
        action="store_true",
        help="Use legacy per-chain network tasks. Default is staged unique-PDB/UniProt prefetch.",
    )
    args = parser.parse_args()

    output_csv = args.output_csv or (args.output_dir / "bridge.csv")
    summary_json = args.summary_json or (args.output_dir / "summary.json")
    rows_in = _read_dataset(args.dataset_csv, max_rows=args.max)
    if not rows_in:
        raise SystemExit(f"No chain rows found in {args.dataset_csv}")

    pdbe_cache = args.output_dir / "cache" / "pdbe_mappings"
    uniprot_cache = args.output_dir / "cache" / "uniprot_fasta"
    alphafold_cache = args.output_dir / "cache" / "alphafold_api"
    if not args.row_level_fetch and args.skip_alphafold_api:
        results = _build_rows_staged(
            rows_in,
            pdbe_cache,
            uniprot_cache,
            args.timeout,
            args.offline,
            args.identity_threshold,
            args.coverage_threshold,
            args.max_indel_run,
            args.workers,
        )
    elif args.workers <= 1:
        results = []
        for row in rows_in:
            results.append(
                _process_row(
                    row,
                    pdbe_cache,
                    uniprot_cache,
                    alphafold_cache,
                    args.timeout,
                    args.offline,
                    args.identity_threshold,
                    args.coverage_threshold,
                    args.max_indel_run,
                    args.skip_alphafold_api,
                )
            )
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _process_row,
                    row,
                    pdbe_cache,
                    uniprot_cache,
                    alphafold_cache,
                    args.timeout,
                    args.offline,
                    args.identity_threshold,
                    args.coverage_threshold,
                    args.max_indel_run,
                    args.skip_alphafold_api,
                )
                for row in rows_in
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda x: (x["pdb_id"], x["chain_id"], x["source_dataset_row"]))
    _write_outputs(results, output_csv, summary_json)
    print(json.dumps(json.loads(summary_json.read_text()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
