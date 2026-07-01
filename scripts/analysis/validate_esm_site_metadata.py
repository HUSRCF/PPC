#!/usr/bin/env python3
"""One-time offline check that ESM payload residue/chain metadata matches label
and sequence-feature files, so training can skip the equivalent per-__getitem__
checks (strict_label_metadata / strict_sequence_feature_metadata).

Run this once after regenerating features, labels, or sequence-feature files
for a split. If it reports zero failures, it is safe to pass
--no-strict-label-metadata --no-strict-sequence-feature-metadata to
train_esm_site.py for that data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppcbind.data.esm_site_dataset import (  # noqa: E402
    _compare_label_metadata,
    _compare_sequence_feature_metadata,
    _discover_esm_paths,
    _esm_id,
    _load_labels,
    _torch_load,
)


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _check_one(
    esm_path_str: str,
    label_root_str: str | None,
    sequence_feature_root_str: str | None,
) -> dict[str, Any]:
    esm_path = Path(esm_path_str)
    pdb_id = _esm_id(esm_path)
    label_root = Path(label_root_str) if label_root_str else None
    sequence_feature_root = Path(sequence_feature_root_str) if sequence_feature_root_str else None
    try:
        data = _torch_load(esm_path)
        n_res = int(len(data.get("residue_names_1", data.get("residue_name_1", []))))
        if n_res == 0 and "embeddings" in data:
            n_res = int(data["embeddings"].shape[0])

        labels, label_data = _load_labels(label_root, pdb_id, n_res, required=label_root is not None)
        if label_root is not None:
            _compare_label_metadata(data, label_data, pdb_id, n_res)

        if sequence_feature_root is not None:
            residue_names = list(data.get("residue_names_1", data.get("residue_name_1", ["X"] * n_res)))
            chain_ids = list(data.get("chain_ids", data.get("chain_id", [""] * n_res)))
            candidates = (
                sequence_feature_root / pdb_id / f"{pdb_id}_seq.pt",
                sequence_feature_root / pdb_id / f"{pdb_id}_sequence.pt",
                sequence_feature_root / f"{pdb_id}_seq.pt",
                sequence_feature_root / f"{pdb_id}.pt",
            )
            seq_path = next((c for c in candidates if c.exists()), None)
            if seq_path is None:
                return {"pdb_id": pdb_id, "ok": False, "error": f"sequence feature file not found under {sequence_feature_root}"}
            seq_data = _torch_load(seq_path)
            if isinstance(seq_data, dict):
                _compare_sequence_feature_metadata(seq_data, residue_names, chain_ids, pdb_id)

        return {"pdb_id": pdb_id, "ok": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"pdb_id": pdb_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-root", required=True, type=Path)
    parser.add_argument("--label-root", default=None, type=Path)
    parser.add_argument("--sequence-feature-root", default=None, type=Path)
    parser.add_argument("--ids", default=None, type=Path, help="Optional id list; defaults to all files under --esm-root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", default=None, type=Path, help="Optional CSV path for per-id results")
    args = parser.parse_args()

    ids = _read_ids(args.ids) if args.ids is not None else None
    esm_paths = _discover_esm_paths(args.esm_root, ids)
    if not esm_paths:
        raise FileNotFoundError(f"No *_esm2.pt files found under {args.esm_root}")

    label_root_str = str(args.label_root) if args.label_root else None
    seq_root_str = str(args.sequence_feature_root) if args.sequence_feature_root else None

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_check_one, str(path), label_root_str, seq_root_str): path
            for path in esm_paths
        }
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 500 == 0 or done == len(futures):
                print(f"checked {done}/{len(futures)}", file=sys.stderr, flush=True)

    failures = [r for r in results if not r["ok"]]
    summary = {
        "n_checked": len(results),
        "n_ok": len(results) - len(failures),
        "n_failed": len(failures),
        "esm_root": str(args.esm_root),
        "label_root": label_root_str,
        "sequence_feature_root": seq_root_str,
    }
    print(json.dumps(summary, indent=2))
    for failure in failures[:50]:
        print(f"FAIL {failure['pdb_id']}: {failure['error']}", file=sys.stderr)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pdb_id", "ok", "error"])
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda r: r["pdb_id"]))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
