"""Frozen, label-free MSA profiles and query-row encoder features for M18."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import time

from Bio import SeqIO
import numpy as np
import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ENCODER_NAME = "esm_msa1b_t12_100M_UR50S"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_alignment(text: str, query: str) -> list[str]:
    rows = []
    for record in SeqIO.parse(StringIO(text), "fasta"):
        row = "".join(c for c in str(record.seq) if not c.islower() and c != ".")
        if len(row) != len(query) or any(not ("A" <= c <= "Z" or c == "-") for c in row):
            raise ValueError(f"Invalid A3M match columns in {record.id}")
        rows.append(row)
    if not rows or rows[0] != query:
        raise ValueError("A3M first query does not exactly match frozen FASTA")
    return [rows[0], *sorted(set(rows[1:]) - {rows[0]})]


def encode_rows(rows: list[str]) -> np.ndarray:
    lookup = np.full(256, 21, dtype=np.uint8)
    for i, aa in enumerate(AMINO_ACIDS):
        lookup[ord(aa)] = i
    lookup[ord("-")] = 20
    return lookup[np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8)].reshape(len(rows), -1)


def profile_features(rows: list[str], query: str) -> np.ndarray:
    """24 bounded statistics + availability; Henikoff weights, Kish effective depth."""
    matrix = encode_rows(rows)
    n, length = matrix.shape
    result = np.zeros((length, 25), dtype=np.float32)
    if n < 2:
        return result
    counts = np.zeros((length, 20), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    for column in range(length):
        cnt = np.bincount(matrix[:, column], minlength=22)[:20]
        counts[column] = cnt
        diverse = np.count_nonzero(cnt)
        valid = matrix[:, column] < 20
        if diverse:
            weights[valid] += 1. / (diverse * cnt[matrix[valid, column]])
    if weights.sum() <= 0:
        return result
    weights /= weights.sum()
    valid = matrix < 20
    mass = weights @ valid
    for aa in range(20):
        result[:, aa] = np.divide(weights @ (matrix == aa), mass,
                                  out=np.zeros(length), where=mass > 0)
    result[:, 20] = weights @ (matrix == 20)
    freq = result[:, :20].astype(np.float64)
    result[:, 21] = -(freq * np.log(freq + 1.e-12)).sum(axis=1) / math.log(20)
    query_codes = encode_rows([query])[0]
    known_query = query_codes < 20
    result[known_query, 22] = freq[np.arange(length)[known_query], query_codes[known_query]]
    squares = (weights ** 2) @ valid
    kish = np.divide(mass ** 2, squares, out=np.zeros(length), where=squares > 0)
    result[:, 23] = np.log1p(kish) / math.log1p(16384)
    informative = valid.sum(axis=0) >= 2
    result[:, 24] = informative
    result[~informative] = 0
    return np.clip(result, 0., 1.)


def select_rows(rows: list[str], depth: int) -> list[str]:
    """Keep query; deterministically stratify other unique rows by query identity."""
    if len(rows) <= depth:
        return rows
    query = rows[0]
    ordered = sorted(rows[1:], key=lambda row: (sum(a == b for a, b in zip(row, query)), row))
    indices = np.linspace(0, len(ordered) - 1, depth - 1, dtype=int) if depth > 1 else []
    return [query, *[ordered[i] for i in indices]]


def window_starts(length: int, window: int = 1000, overlap: int = 128) -> list[int]:
    if not 0 <= overlap < window:
        raise ValueError("Require 0 <= overlap < window")
    if length <= window:
        return [0]
    return sorted(set([*range(0, length - window + 1, window - overlap), length - window]))


def encoder_features(model, converter, rows: list[str], device: str, depth: int) -> tuple[np.ndarray, dict]:
    length = len(rows[0])
    retries = []
    depths = sorted({depth, max(1, depth // 2), max(1, depth // 4), min(depth, 8), 1}, reverse=True)
    for current_depth in depths:
        selected = select_rows(rows, current_depth)
        try:
            features = np.zeros((length, 768), dtype=np.float32)
            coverage = np.zeros(length, dtype=np.float32)
            for start in window_starts(length):
                end = min(length, start + 1000)
                normalized = [(str(i), row[start:end].translate(str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X", "J": "X"}))) for i, row in enumerate(selected)]
                _, _, tokens = converter([normalized])
                tokens = tokens.to(device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    output = model(tokens, repr_layers=[12], return_contacts=False)
                value = output["representations"][12][0, 0, 1:1 + end - start].float().cpu().numpy()
                if value.shape != (end - start, 768) or not np.isfinite(value).all():
                    raise ValueError("Nonfinite or incorrect query-row representation")
                features[start:end] += value
                coverage[start:end] += 1
                del output, value, tokens
            if np.any(coverage == 0):
                raise ValueError("Incomplete full-chain window coverage")
            return features / coverage[:, None], {"selected_depth": len(selected), "oom_retries": retries,
                                                   "windows": window_starts(length), "window": 1000, "overlap": 128,
                                                   "ambiguous_token_policy": "UZOBJ->X; residue positions preserved"}
        except torch.cuda.OutOfMemoryError:
            retries.append(current_depth)
            torch.cuda.empty_cache()
    raise RuntimeError(f"OOM even at query-only depth; retries={retries}")


def load_inputs(index_path: Path, fasta_path: Path) -> tuple[list[dict], dict[str, str]]:
    with index_path.open() as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: row["union_seq_id"])
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    queries = {record.id: str(record.seq) for record in records}
    if len(queries) != len(records) or len(rows) != len(queries):
        raise ValueError("Duplicate IDs or inconsistent index/FASTA sizes")
    if {r["union_seq_id"] for r in rows} != set(queries):
        raise ValueError("Index/FASTA identity mismatch")
    for row in rows:
        if int(row["length"]) != len(queries[row["union_seq_id"]]):
            raise ValueError("Index/FASTA length mismatch")
    return rows, queries


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_payload(data: dict, row: dict, query: str, mode: str, source_hash: str,
                     checkpoint_hash: str | None, requested_depth: int) -> None:
    dimension = 25 if mode == "profile" else 768
    if data["sequence"] != query or tuple(data["embeddings"].shape) != (len(query), dimension):
        raise ValueError("Feature sequence/shape mismatch")
    if data["sequence_sha256"] != hashlib.sha256(query.encode()).hexdigest():
        raise ValueError("Feature query hash mismatch")
    if not torch.isfinite(data["embeddings"]).all():
        raise ValueError("Nonfinite cached feature")
    if (data["source_msa_sha256"] != row["msa_sha256"] or data["feature_mode"] != mode
            or data["source_code_sha256"] != source_hash
            or data.get("checkpoint_sha256") != checkpoint_hash):
        raise ValueError("Feature source/version/checkpoint mismatch")
    expected_depth = requested_depth if mode == "encoder" else None
    if data.get("requested_depth") != expected_depth:
        raise ValueError("Feature requested-depth mismatch")


def audit(root: Path, rows: list[dict], queries: dict[str, str], mode: str,
          checkpoint_hash: str | None = None, requested_depth: int = 64) -> dict:
    dimension = 25 if mode == "profile" else 768
    if mode == "encoder" and not checkpoint_hash:
        raise ValueError("Encoder audit requires an explicit checkpoint hash")
    source_hash = digest(Path(__file__))
    for row in rows:
        uid = row["union_seq_id"]
        data = torch.load(root / f"{uid}.pt", map_location="cpu", weights_only=True)
        validate_payload(data, row, queries[uid], mode, source_hash, checkpoint_hash, requested_depth)
    summary = {"mode": mode, "sequences": len(rows), "dimension": dimension,
               "residues": sum(map(len, queries.values())), "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
               "exact_queries": True, "finite": True, "labels_read": False, "full_length": True,
               "source_code_sha256": source_hash, "checkpoint_sha256": checkpoint_hash,
               "requested_depth": requested_depth if mode == "encoder" else None}
    atomic_json(root / "validation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "encoder", "audit-profile", "audit-encoder"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.depth < 1:
        parser.error("Invalid shard/depth")
    torch.set_num_threads(4)
    rows, queries = load_inputs(args.index, args.fasta)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if args.mode.startswith("audit-"):
        mode = args.mode.removeprefix("audit-")
        if mode == "encoder" and args.checkpoint is None:
            parser.error("Encoder audit requires the explicit official checkpoint")
        checkpoint_hash = digest(args.checkpoint) if mode == "encoder" else None
        print(json.dumps(audit(root, rows, queries, mode, checkpoint_hash, args.depth)), flush=True)
        return
    model = converter = None
    checkpoint_hash = None
    if args.mode == "encoder":
        import esm
        if args.checkpoint is None:
            parser.error("Encoder requires an explicit trusted official checkpoint")
        checkpoint_hash = digest(args.checkpoint)
        # Only this official model asset is loaded with pickle compatibility.
        model_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(ENCODER_NAME, model_data)
        model = model.eval().to(args.device)
        converter = alphabet.get_batch_converter()
    selected_rows = rows[args.shard::args.shards]
    if args.limit:
        selected_rows = selected_rows[:args.limit]
    failures = []
    source_hash = digest(Path(__file__))
    for number, row in enumerate(selected_rows, 1):
        uid = row["union_seq_id"]
        start = time.monotonic()
        path = root / f"{uid}.pt"
        try:
            if path.exists():
                cached = torch.load(path, map_location="cpu", weights_only=True)
                validate_payload(cached, row, queries[uid], args.mode, source_hash, checkpoint_hash, args.depth)
                print(json.dumps({"uid": uid, "status": "cached", "completed": number, "total": len(selected_rows)}), flush=True)
                continue
            raw = Path(row["msa_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["msa_sha256"]:
                raise ValueError("Frozen A3M source hash mismatch")
            alignment = read_alignment(raw.decode(), queries[uid])
            if args.mode == "profile":
                features = profile_features(alignment, queries[uid])
                detail = {"weighting": "Henikoff position weights after exact aligned-row deduplication", "depth_definition": "Kish effective depth, not HHsuite Neff", "dimension": 25}
            else:
                features, detail = encoder_features(model, converter, alignment, args.device, args.depth)
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite feature output")
            payload = {"embeddings": torch.from_numpy(features).to(torch.float32 if args.mode == "profile" else torch.float16),
                       "sequence": queries[uid], "sequence_sha256": hashlib.sha256(queries[uid].encode()).hexdigest(),
                       "source_msa_sha256": row["msa_sha256"], "source_code_sha256": source_hash,
                       "feature_mode": args.mode, "checkpoint_sha256": checkpoint_hash,
                       "requested_depth": args.depth if args.mode == "encoder" else None,
                       "unique_alignment_rows": len(alignment), "details": detail}
            validate_payload(payload, row, queries[uid], args.mode, source_hash, checkpoint_hash, args.depth)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            print(json.dumps({"uid": uid, "status": "done", "completed": number,
                              "total": len(selected_rows), "seconds": time.monotonic() - start, **detail}), flush=True)
        except Exception as exc:
            failures.append({"uid": uid, "error": repr(exc)})
            print(json.dumps({"uid": uid, "status": "failed", "error": repr(exc)}), flush=True)
    atomic_json(root / f"receipt_shard_{args.shard}.json", {"mode": args.mode, "shard": args.shard,
                "shards": args.shards, "attempted": len(selected_rows), "failures": failures,
                "source_code_sha256": source_hash, "checkpoint_sha256": checkpoint_hash})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
