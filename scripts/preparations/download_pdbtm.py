#!/usr/bin/env python3
"""Download PDBTM membrane protein structures and XML annotations."""

from __future__ import annotations

import argparse
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_ROOT = Path("/media/WDisk/ProtBind/PPC")
DEFAULT_ALPHA_URL = "https://pdbtm.unitmp.org/data/PDBTM/data/pdbtm_alpha.txt"
DEFAULT_ENTRY_BASE = "http://pdbtm.unitmp.org/api/v1/entry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Data root directory.")
    parser.add_argument("--alpha-url", default=DEFAULT_ALPHA_URL, help="PDBTM alpha list URL.")
    parser.add_argument("--entry-base", default=DEFAULT_ENTRY_BASE, help="PDBTM entry API base URL.")
    parser.add_argument("--list", dest="list_path", type=Path, default=None, help="Existing pdbtm_alpha list.")
    parser.add_argument("--download-list", action="store_true", help="Download/update the PDBTM alpha list first.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of unique PDB entries to download.")
    parser.add_argument("--only-list", action="store_true", help="Only download/list entries; do not fetch PDB/XML.")
    parser.add_argument("--insecure", action="store_true", help="Ignore TLS certificate errors.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between entries.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per file after the first attempt.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel entry downloads.")
    return parser.parse_args()


def urlretrieve(url: str, dest: Path, insecure: bool, retries: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context() if insecure else None
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, context=context, timeout=60) as response:
                data = response.read()
            if not data:
                raise RuntimeError(f"empty response from {url}")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"failed to download {url}: {last_error}")


def load_unique_pdb_ids(list_path: Path) -> list[str]:
    pdb_ids: list[str] = []
    seen: set[str] = set()

    for raw in list_path.read_text().splitlines():
        entry = raw.strip()
        if not entry:
            continue
        pdb_id = entry.split("_", 1)[0].lower()
        if pdb_id and pdb_id not in seen:
            seen.add(pdb_id)
            pdb_ids.append(pdb_id)

    return pdb_ids


def download_entry(
    pdb_id: str,
    entry_base: str,
    pdb_dir: Path,
    xml_dir: Path,
    insecure: bool,
    retries: int,
) -> list[str]:
    failures: list[str] = []
    targets = [
        (f"{entry_base}/{pdb_id}.trpdb", pdb_dir / f"{pdb_id}.pdb"),
        (f"{entry_base}/{pdb_id}.xml", xml_dir / f"{pdb_id}.xml"),
    ]
    for url, dest in targets:
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            urlretrieve(url, dest, insecure=insecure, retries=retries)
        except RuntimeError as exc:
            failures.append(f"{pdb_id}\t{dest.name}\t{exc}")
    return failures


def main() -> int:
    args = parse_args()
    root = args.root
    database_dir = root / "data" / "database"
    pdb_dir = root / "data" / "pdbtm" / "cplx"
    xml_dir = root / "data" / "xml"
    list_path = args.list_path or database_dir / "pdbtm_alpha_latest.txt"

    database_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    if args.download_list or not list_path.exists():
        print(f"Downloading PDBTM alpha list: {args.alpha_url}")
        urlretrieve(args.alpha_url, list_path, insecure=args.insecure, retries=args.retries)

    pdb_ids = load_unique_pdb_ids(list_path)
    if args.limit is not None:
        pdb_ids = pdb_ids[: args.limit]

    print(f"List: {list_path}")
    print(f"Unique PDB entries selected: {len(pdb_ids)}")
    if args.only_list:
        for pdb_id in pdb_ids[:20]:
            print(pdb_id)
        return 0

    failures: list[str] = []
    if args.workers <= 1:
        for idx, pdb_id in enumerate(pdb_ids, start=1):
            print(f"[{idx}/{len(pdb_ids)}] {pdb_id}")
            entry_failures = download_entry(
                pdb_id, args.entry_base, pdb_dir, xml_dir, args.insecure, args.retries
            )
            failures.extend(entry_failures)
            for failure in entry_failures:
                print(f"  FAIL {failure}")
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_pdb = {
                executor.submit(
                    download_entry,
                    pdb_id,
                    args.entry_base,
                    pdb_dir,
                    xml_dir,
                    args.insecure,
                    args.retries,
                ): pdb_id
                for pdb_id in pdb_ids
            }
            for idx, future in enumerate(as_completed(future_to_pdb), start=1):
                pdb_id = future_to_pdb[future]
                try:
                    entry_failures = future.result()
                except Exception as exc:
                    entry_failures = [f"{pdb_id}\tentry\t{exc}"]
                failures.extend(entry_failures)
                status = "OK" if not entry_failures else f"FAIL {len(entry_failures)}"
                print(f"[{idx}/{len(pdb_ids)}] {pdb_id} {status}")
                for failure in entry_failures:
                    print(f"  FAIL {failure}")
                if args.sleep:
                    time.sleep(args.sleep)

    fail_path = database_dir / "pdbtm_download_failures.txt"
    fail_path.write_text("\n".join(failures) + ("\n" if failures else ""))
    print(f"Finished. Failures: {len(failures)}")
    print(f"Failure log: {fail_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
