#!/usr/bin/env python3
"""Average exactly aligned residue-score NPZ files in probability or logit space."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    required = {"seq_ids", "labels", "scores"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"Missing arrays in {path}: {missing}")
    seq_ids = payload["seq_ids"]
    labels = payload["labels"]
    scores = payload["scores"]
    if "offsets" in payload.files:
        offsets = payload["offsets"].astype(np.int64, copy=False)
        lengths = np.diff(offsets).astype(np.int32, copy=False)
    elif "lengths" in payload.files:
        lengths = payload["lengths"].astype(np.int32, copy=False)
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    else:
        raise ValueError(f"{path} must contain offsets or lengths")
    if offsets.size != seq_ids.size + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != labels.size:
        raise ValueError(f"Invalid packed offsets in {path}")
    if labels.shape != scores.shape or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"Invalid packed score arrays in {path}")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"Scores are not finite probabilities in {path}")
    return {
        "seq_ids": seq_ids,
        "lengths": lengths,
        "offsets": offsets,
        "labels": labels.astype(np.uint8, copy=False),
        "scores": scores.astype(np.float64, copy=False),
    }


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1.0e-7, 1.0 - 1.0e-7)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--space", choices=("logit", "probability"), default="logit")
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()
    if len(args.input) < 2:
        raise ValueError("At least two input score files are required")

    payloads = [load(path) for path in args.input]
    reference = payloads[0]
    for path, payload in zip(args.input[1:], payloads[1:]):
        for key in ("seq_ids", "lengths", "offsets", "labels"):
            if not np.array_equal(reference[key], payload[key]):
                raise ValueError(f"Alignment mismatch for {key} in {path}")
    stacked = np.stack([payload["scores"] for payload in payloads], axis=0)
    if args.space == "logit":
        scores = sigmoid(np.mean(logit(stacked), axis=0))
    else:
        scores = np.mean(stacked, axis=0)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        seq_ids=reference["seq_ids"],
        lengths=reference["lengths"],
        offsets=reference["offsets"],
        labels=reference["labels"],
        scores=scores.astype(np.float32),
    )
    summary = {
        "method": args.method,
        "averaging_space": args.space,
        "n_members": len(args.input),
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.input
        ],
        "output": {"path": str(args.output_npz), "sha256": sha256_file(args.output_npz)},
        "coverage": {
            "chains": int(reference["seq_ids"].size),
            "residues": int(reference["labels"].size),
            "positives": int(reference["labels"].sum()),
        },
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
