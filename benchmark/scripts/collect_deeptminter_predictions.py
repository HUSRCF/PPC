#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_UNIQUE = 724
EXPECTED_CHAINS = 2392
EXPECTED_RESIDUES = 1_036_256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--unique-manifest", type=Path, required=True)
    parser.add_argument("--reverse-map", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--method", default="DeepTMInter_official_full")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Emit only fully QC-passing unique predictions. Final runs must omit this flag.",
    )
    parser.add_argument(
        "--smoke-run-ids",
        help="Comma-separated explicit run IDs; required with --allow-partial and forbidden otherwise.",
    )
    return parser.parse_args()


def sha256_text(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_unique_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "run_id",
            "prot_chain",
            "seq_sha256",
            "representative_seq_id",
            "length",
            "n_chains",
            "sequence",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"unexpected unique-manifest columns: {reader.fieldnames}")
        for row in reader:
            run_id = row["run_id"]
            if run_id in rows:
                raise ValueError(f"duplicate run_id in unique manifest: {run_id}")
            sequence = row["sequence"].strip()
            if row["prot_chain"] != "A":
                raise ValueError(f"{run_id}: expected synthetic chain A")
            if int(row["length"]) != len(sequence):
                raise ValueError(f"{run_id}: manifest length mismatch")
            if sha256_text(sequence) != row["seq_sha256"]:
                raise ValueError(f"{run_id}: manifest SHA256 mismatch")
            rows[run_id] = row
    if not rows:
        raise ValueError("empty unique manifest")
    return rows


def read_reverse_map(path: Path) -> tuple[dict[str, dict[str, str]], Counter[str]]:
    rows: dict[str, dict[str, str]] = {}
    run_counts: Counter[str] = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run_id", "seq_sha256", "seq_id", "pdb_id", "chain_id", "length"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"unexpected reverse-map columns: {reader.fieldnames}")
        for row in reader:
            seq_id = row["seq_id"]
            if seq_id in rows:
                raise ValueError(f"duplicate case-sensitive seq_id in reverse map: {seq_id}")
            expected_seq_id = f'{row["pdb_id"]}__{row["chain_id"]}'
            if seq_id != expected_seq_id:
                raise ValueError(f"{seq_id}: PDB/chain identity mismatch ({expected_seq_id})")
            rows[seq_id] = row
            run_counts[row["run_id"]] += 1
    if not rows:
        raise ValueError("empty reverse map")
    return rows, run_counts


def read_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"seq_id", "row_index", "position", "residue", "label"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"unexpected label columns: {reader.fieldnames}")
        for row in reader:
            rows[row["seq_id"]].append(row)
    for seq_id, chain_rows in rows.items():
        chain_rows.sort(key=lambda row: int(row["row_index"]))
        for expected_index, row in enumerate(chain_rows):
            if int(row["row_index"]) != expected_index:
                raise ValueError(f"{seq_id}: non-contiguous row_index at {expected_index}")
            if int(row["position"]) != expected_index + 1:
                raise ValueError(f"{seq_id}: non-contiguous 1-based position at {expected_index}")
            if len(row["residue"]) != 1:
                raise ValueError(f"{seq_id}: invalid residue at row {expected_index}")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"{seq_id}: non-binary label at row {expected_index}")
    if not rows:
        raise ValueError("empty labels")
    return dict(rows)


def qc_passes(path: Path, run_id: str) -> bool:
    complete = path / run_id / "complete.json"
    if not complete.is_file():
        return False
    payload = json.loads(complete.read_text())
    return payload.get("status") == "pass" and payload.get("run_id") == run_id


def read_prediction(path: Path, expected_sequence: str) -> list[float]:
    residues: list[str] = []
    scores: list[float] = []
    with path.open() as handle:
        for expected_position, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}: expected three tab-separated fields on line {expected_position}")
            position, residue, score_text = fields
            if int(position) != expected_position:
                raise ValueError(f"{path}: non-contiguous position on line {expected_position}")
            if len(residue) != 1:
                raise ValueError(f"{path}: invalid residue on line {expected_position}")
            score = float(score_text)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{path}: invalid score on line {expected_position}: {score_text}")
            residues.append(residue)
            scores.append(score)
    if "".join(residues) != expected_sequence:
        raise ValueError(f"{path}: prediction sequence mismatch")
    return scores


def main() -> None:
    args = parse_args()
    if args.allow_partial:
        if not args.smoke_run_ids:
            raise ValueError("--allow-partial requires explicit --smoke-run-ids")
        if "smoke" not in args.output_tsv.name or "smoke" not in args.summary_json.name:
            raise ValueError("partial outputs must contain 'smoke' in both output filenames")
    elif args.smoke_run_ids:
        raise ValueError("--smoke-run-ids is only valid with --allow-partial")

    manifest = read_unique_manifest(args.unique_manifest)
    reverse, reverse_counts = read_reverse_map(args.reverse_map)
    labels = read_labels(args.labels)

    n_label_residues = sum(len(rows) for rows in labels.values())
    if len(manifest) != EXPECTED_UNIQUE:
        raise ValueError(f"fixed universe requires {EXPECTED_UNIQUE} unique sequences")
    if len(reverse) != EXPECTED_CHAINS or len(labels) != EXPECTED_CHAINS:
        raise ValueError(f"fixed universe requires {EXPECTED_CHAINS} case-sensitive chains")
    if n_label_residues != EXPECTED_RESIDUES:
        raise ValueError(f"fixed universe requires {EXPECTED_RESIDUES} label residues")

    if set(reverse) != set(labels):
        only_reverse = sorted(set(reverse) - set(labels))[:20]
        only_labels = sorted(set(labels) - set(reverse))[:20]
        raise ValueError(
            "label/reverse-map universe mismatch: "
            f"reverse={len(reverse)} labels={len(labels)} "
            f"only_reverse={only_reverse} only_labels={only_labels}"
        )
    if set(reverse_counts) != set(manifest):
        raise ValueError("unique-manifest/reverse-map run_id universe mismatch")

    for run_id, row in manifest.items():
        if reverse_counts[run_id] != int(row["n_chains"]):
            raise ValueError(f"{run_id}: reverse-map n_chains mismatch")

    completed_run_ids: list[str] = []
    missing_run_ids: list[str] = []
    prediction_paths: dict[str, Path] = {}
    for run_id in sorted(manifest):
        path = args.prediction_dir / f"{run_id}A.deeptminter_full_chain"
        if qc_passes(args.qc_dir, run_id) and path.is_file():
            completed_run_ids.append(run_id)
            prediction_paths[run_id] = path
        else:
            missing_run_ids.append(run_id)

    if args.allow_partial:
        requested_run_ids = [item.strip() for item in args.smoke_run_ids.split(",") if item.strip()]
        if not requested_run_ids or len(requested_run_ids) != len(set(requested_run_ids)):
            raise ValueError("--smoke-run-ids must be a non-empty list without duplicates")
        unknown = sorted(set(requested_run_ids) - set(manifest))
        unavailable = sorted(set(requested_run_ids) - set(completed_run_ids))
        if unknown:
            raise ValueError(f"unknown smoke run IDs: {unknown}")
        if unavailable:
            raise ValueError(f"smoke run IDs are not fully QC-passing: {unavailable}")
        selected_run_ids = sorted(requested_run_ids)
    else:
        selected_run_ids = completed_run_ids

    if missing_run_ids and not args.allow_partial:
        raise FileNotFoundError(
            f"final collection requires complete coverage; missing {len(missing_run_ids)} "
            f"of {len(manifest)} run_ids, first={missing_run_ids[:20]}"
        )
    if not selected_run_ids:
        raise ValueError("no fully QC-passing predictions available")

    score_cache: dict[str, list[float]] = {}
    for run_id in selected_run_ids:
        score_cache[run_id] = read_prediction(
            prediction_paths[run_id], manifest[run_id]["sequence"]
        )

    selected_run_set = set(selected_run_ids)
    selected_seq_ids = sorted(
        seq_id for seq_id, mapping in reverse.items() if mapping["run_id"] in score_cache
    )
    expected_rows = sum(
        int(row["length"]) * reverse_counts[run_id]
        for run_id, row in manifest.items()
        if run_id in selected_run_set
    )
    if not args.allow_partial:
        if len(selected_run_ids) != EXPECTED_UNIQUE:
            raise ValueError("final output does not cover every unique sequence")
        if len(selected_seq_ids) != EXPECTED_CHAINS:
            raise ValueError("final output does not cover every case-sensitive chain")
        if expected_rows != EXPECTED_RESIDUES:
            raise ValueError("final expected residue count does not match fixed universe")

    output_rows = 0
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output_tsv.with_name(f"{args.output_tsv.name}.tmp.{os.getpid()}")
    with output_tmp.open("w", newline="") as handle:
        fieldnames = ["seq_id", "position", "residue", "score", "method"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for seq_id in selected_seq_ids:
            mapping = reverse[seq_id]
            run_id = mapping["run_id"]
            manifest_row = manifest[run_id]
            chain_rows = labels[seq_id]
            label_sequence = "".join(row["residue"] for row in chain_rows)
            if int(mapping["length"]) != len(chain_rows):
                raise ValueError(f"{seq_id}: reverse-map/label length mismatch")
            if mapping["seq_sha256"] != manifest_row["seq_sha256"]:
                raise ValueError(f"{seq_id}: reverse-map/manifest SHA256 mismatch")
            if label_sequence != manifest_row["sequence"]:
                raise ValueError(f"{seq_id}: label/manifest sequence mismatch")
            scores = score_cache[run_id]
            if len(scores) != len(chain_rows):
                raise ValueError(f"{seq_id}: score/label length mismatch")
            for row, score in zip(chain_rows, scores, strict=True):
                writer.writerow(
                    {
                        "seq_id": seq_id,
                        "position": row["position"],
                        "residue": row["residue"],
                        "score": format(score, ".17g"),
                        "method": args.method,
                    }
                )
                output_rows += 1
    if output_rows != expected_rows:
        raise ValueError(f"output row mismatch: {output_rows} != {expected_rows}")

    output_sha256 = sha256_file(output_tmp)
    summary = {
        "method": args.method,
        "scope": "partial_smoke" if args.allow_partial else "complete",
        "n_manifest_unique": len(manifest),
        "n_qc_completed_unique": len(completed_run_ids),
        "n_selected_unique": len(selected_run_ids),
        "n_missing_unique": len(missing_run_ids),
        "missing_run_ids_first20": missing_run_ids[:20],
        "n_manifest_chains": len(reverse),
        "n_output_chains": len(selected_seq_ids),
        "n_output_residues": output_rows,
        "output_tsv": str(args.output_tsv.resolve()),
        "output_sha256": output_sha256,
        "publication_contract": (
            "summary JSON is the commit marker; consumers must verify output_sha256"
        ),
        "qc": (
            "complete.json pass; exact case-sensitive chain universe; contiguous residue "
            "positions; exact sequence, length, and SHA256; finite scores in [0,1]"
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = args.summary_json.with_name(f"{args.summary_json.name}.tmp.{os.getpid()}")
    summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output_tmp.replace(args.output_tsv)
    summary_tmp.replace(args.summary_json)
    if sha256_file(args.output_tsv) != output_sha256:
        raise RuntimeError("published output SHA256 changed after atomic rename")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
