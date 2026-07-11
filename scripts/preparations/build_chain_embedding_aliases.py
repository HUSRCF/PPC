#!/usr/bin/env python3
"""Create per-chain aliases to exact-sequence embedding payloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _is_selected(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--unique-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    created = 0
    existing = 0
    missing: list[str] = []
    with args.chain_manifest.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not _is_selected(row.get("is_selected", "")):
                continue
            seq_id = row["selected_seq_id"].strip()
            unique_id = row["unique_id"].strip()
            digest = row["seq_sha256"].strip()
            target = args.unique_root / digest[:2] / f"{unique_id}.pt"
            if not target.exists():
                missing.append(seq_id)
                continue
            alias = args.output_root / f"{seq_id}.pt"
            relative = Path(os.path.relpath(target, alias.parent))
            if alias.is_symlink() and Path(os.readlink(alias)) == relative:
                existing += 1
                continue
            if alias.exists() or alias.is_symlink():
                raise FileExistsError(f"Alias exists with a different target: {alias}")
            alias.symlink_to(relative)
            created += 1
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} unique payloads; examples={missing[:20]}")
    summary = {"created": created, "existing": existing, "missing": 0, "output_root": str(args.output_root)}
    (args.output_root / "aliases_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
