#!/usr/bin/env python3
"""Download TmAlphaFold/AlphaFold predicted structures for bridge-table rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


TMALPHAFOLD_ENTRY_URL = "https://tmalphafold.ttk.hu/entry/{accession}"
TMALPHAFOLD_BASE = "https://tmalphafold.ttk.hu"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

TM_ENDPOINTS = {
    "trpdb": "tmalphafold.pdb",
    "tmdet": "tmdet.xml",
    "cctop": "cctop.xml",
    "eval": "eval.xml",
}

FIELDNAMES = [
    "uniprot_acc",
    "status",
    "source",
    "tmalphafold_entry_url",
    "tmalphafold_pdb_path",
    "tmdet_xml_path",
    "cctop_xml_path",
    "eval_xml_path",
    "alphafold_pdb_path",
    "alphafold_confidence_path",
    "alphafold_pae_path",
    "error",
]


def _request_bytes(url: str, timeout: float, retries: int = 3) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PPC-pred-struct/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_exc!r}")


def _request_text(url: str, timeout: float, retries: int = 3) -> str:
    return _request_bytes(url, timeout=timeout, retries=retries).decode("utf-8", errors="replace")


def _request_json(url: str, timeout: float, retries: int = 3) -> Any:
    return json.loads(_request_text(url, timeout=timeout, retries=retries))


def _atomic_write(path: Path, data: bytes, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)
    return True


def _maybe_gunzip(data: bytes) -> bytes:
    if data.startswith(b"\x1f\x8b"):
        return gzip.decompress(data)
    return data


def _read_accessions(bridge_csv: Path, require_strict: bool) -> list[str]:
    accessions: list[str] = []
    with bridge_csv.open("rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            acc = (row.get("uniprot_acc") or "").strip()
            if not acc:
                continue
            if require_strict and str(row.get("strict_pass", "")).strip() not in {"1", "true", "True"}:
                continue
            accessions.append(acc)
    return sorted(set(accessions))


def _tm_download_links(accession: str, timeout: float) -> dict[str, str]:
    url = TMALPHAFOLD_ENTRY_URL.format(accession=accession)
    html = _request_text(url, timeout=timeout)
    links: dict[str, str] = {}
    for href in re.findall(r'href="([^"]+)"', html):
        decoded = urllib.parse.unquote(href)
        for endpoint in TM_ENDPOINTS:
            if re.search(rf"/downloads/.+/{re.escape(accession)}/{endpoint}$", decoded):
                links[endpoint] = urllib.parse.urljoin(TMALPHAFOLD_BASE, decoded)
    return links


def _afdb_entry(accession: str, timeout: float) -> dict[str, Any] | None:
    try:
        data = _request_json(ALPHAFOLD_API_URL.format(accession=accession), timeout=timeout)
    except Exception:
        return None
    if isinstance(data, list):
        for item in data:
            if str(item.get("uniprotAccession", "")).upper() == accession.upper():
                return item
        return data[0] if data else None
    return data if isinstance(data, dict) else None


def _download_one(accession: str, output_root: Path, timeout: float, overwrite: bool, require_tmalphafold: bool) -> dict[str, str]:
    row = {key: "" for key in FIELDNAMES}
    row.update(
        {
            "uniprot_acc": accession,
            "status": "ERROR",
            "tmalphafold_entry_url": TMALPHAFOLD_ENTRY_URL.format(accession=accession),
        }
    )
    acc_dir = output_root / accession
    try:
        tm_links: dict[str, str] = {}
        try:
            tm_links = _tm_download_links(accession, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"TmAlphaFold lookup failed: {exc!r}"
        if tm_links:
            for endpoint, suffix in TM_ENDPOINTS.items():
                if endpoint not in tm_links:
                    continue
                path = acc_dir / f"{accession}_{suffix}"
                payload = _request_bytes(tm_links[endpoint], timeout=timeout)
                if endpoint == "trpdb":
                    payload = _maybe_gunzip(payload)
                _atomic_write(path, payload, overwrite=overwrite)
                if endpoint == "trpdb":
                    row["tmalphafold_pdb_path"] = str(path)
                elif endpoint == "tmdet":
                    row["tmdet_xml_path"] = str(path)
                elif endpoint == "cctop":
                    row["cctop_xml_path"] = str(path)
                elif endpoint == "eval":
                    row["eval_xml_path"] = str(path)
            if row["tmalphafold_pdb_path"]:
                row["source"] = "tmalphafold"
                row["status"] = "OK"
                return row

        if require_tmalphafold:
            row["status"] = "NO_TMALPHAFOLD"
            return row

        af_entry = _afdb_entry(accession, timeout=timeout)
        if not af_entry or not af_entry.get("pdbUrl"):
            row["status"] = "NO_AFDB"
            return row
        targets = {
            "alphafold_pdb_path": (af_entry.get("pdbUrl"), f"{accession}_alphafold.pdb"),
            "alphafold_confidence_path": (af_entry.get("plddtDocUrl"), f"{accession}_alphafold_confidence.json"),
            "alphafold_pae_path": (af_entry.get("paeDocUrl"), f"{accession}_alphafold_pae.json"),
        }
        for field, (url, filename) in targets.items():
            if not url:
                continue
            path = acc_dir / filename
            _atomic_write(path, _request_bytes(url, timeout=timeout), overwrite=overwrite)
            row[field] = str(path)
        row["source"] = "alphafold"
        row["status"] = "OK" if row["alphafold_pdb_path"] else "NO_AFDB"
        return row
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)
        return row


def _write_manifest(rows: list[dict[str, str]], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest.with_suffix(manifest.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, manifest)
    summary = {
        "n_total": len(rows),
        "statuses": Counter(row["status"] for row in rows),
        "sources": Counter(row["source"] for row in rows),
        "manifest": str(manifest),
    }
    summary_path = manifest.with_suffix(".summary.json")
    tmp_json = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp_json, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-csv", type=Path, default=Path("features/pred_struct_bridge/bridge.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("features/tmalphafold_raw"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-nonstrict", action="store_true")
    parser.add_argument("--require-tmalphafold", action="store_true")
    args = parser.parse_args()

    accessions = _read_accessions(args.bridge_csv, require_strict=not args.allow_nonstrict)
    if args.max is not None:
        accessions = accessions[: args.max]
    if not accessions:
        raise SystemExit(f"No accessions selected from {args.bridge_csv}")

    rows: list[dict[str, str]] = []
    if args.workers <= 1:
        for accession in accessions:
            rows.append(_download_one(accession, args.output_root, args.timeout, args.overwrite, args.require_tmalphafold))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_download_one, accession, args.output_root, args.timeout, args.overwrite, args.require_tmalphafold)
                for accession in accessions
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["uniprot_acc"])
    _write_manifest(rows, args.manifest or (args.output_root / "manifest.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
