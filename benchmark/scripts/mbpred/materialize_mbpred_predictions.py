#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_UNIQUE = 724
EXPECTED_CHAINS = 2_392
EXPECTED_RESIDUES = 1_036_256
RF_TREES = 500
SCORE_GRID_ATOL = 1e-8

UNIQUE_COLUMNS = (
    "run_id",
    "prot_chain",
    "seq_sha256",
    "representative_seq_id",
    "length",
    "n_chains",
    "sequence",
)
REVERSE_COLUMNS = ("run_id", "seq_sha256", "seq_id", "pdb_id", "chain_id", "length")
LABEL_COLUMNS = ("seq_id", "row_index", "position", "residue", "label")
RF_SCORE_COLUMNS = ("", "Res", "AA", "Score")
OUTPUT_COLUMNS = ("seq_id", "position", "residue", "score", "method")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SEQUENCE_RE = re.compile(r"[A-Z]+\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SUMMARY_SCHEMA = "mbpred_adapted_residue_predictions/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize strict MBPred-adapted residue predictions by reverse-mapping "
            "per-unique official-style RF score CSV files to the canonical test chains."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--unique-manifest", type=Path, required=True)
    parser.add_argument("--reverse-map", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument(
        "--score-pattern",
        default="{run_id}.csv",
        help="Relative score path template under --score-dir; must contain {run_id} once.",
    )
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--method", default="MBPred_adapted")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit an explicitly allowlisted smoke subset; forbidden for final publication.",
    )
    parser.add_argument(
        "--smoke-run-ids",
        help="Comma-separated run IDs; required with --allow-partial and forbidden otherwise.",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_positive_int(text: str, context: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{context}: expected integer, got {text!r}") from exc
    if value <= 0:
        raise ValueError(f"{context}: expected positive integer, got {value}")
    return value


def require_exact_header(fieldnames: list[str] | None, expected: tuple[str, ...], context: str) -> None:
    if tuple(fieldnames or ()) != expected:
        raise ValueError(f"{context}: unexpected columns {fieldnames}; expected {list(expected)}")


def require_complete_dict_row(
    row: dict[str | None, str | list[str] | None], expected: tuple[str, ...], context: str
) -> None:
    if set(row) != set(expected) or any(row[column] is None for column in expected):
        raise ValueError(f"{context}: missing or extra delimited field")


def read_unique_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_exact_header(reader.fieldnames, UNIQUE_COLUMNS, str(path))
        for line_number, row in enumerate(reader, start=2):
            require_complete_dict_row(row, UNIQUE_COLUMNS, f"{path}:{line_number}")
            run_id = row["run_id"]
            if not RUN_ID_RE.fullmatch(run_id):
                raise ValueError(f"{path}:{line_number}: unsafe run_id {run_id!r}")
            if run_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate run_id {run_id}")
            if row["prot_chain"] != "A":
                raise ValueError(f"{path}:{line_number}: {run_id} must use synthetic chain A")
            sequence = row["sequence"]
            if not SEQUENCE_RE.fullmatch(sequence):
                raise ValueError(f"{path}:{line_number}: {run_id} has invalid sequence")
            length = parse_positive_int(row["length"], f"{path}:{line_number}: length")
            n_chains = parse_positive_int(row["n_chains"], f"{path}:{line_number}: n_chains")
            if length != len(sequence):
                raise ValueError(f"{path}:{line_number}: {run_id} length/sequence mismatch")
            if not SHA256_RE.fullmatch(row["seq_sha256"]):
                raise ValueError(f"{path}:{line_number}: {run_id} has invalid SHA256 text")
            if sha256_text(sequence) != row["seq_sha256"]:
                raise ValueError(f"{path}:{line_number}: {run_id} sequence SHA256 mismatch")
            if not row["representative_seq_id"]:
                raise ValueError(f"{path}:{line_number}: {run_id} lacks representative_seq_id")
            rows[run_id] = row
    if not rows:
        raise ValueError(f"{path}: empty unique manifest")
    return rows


def read_reverse_map(path: Path) -> tuple[dict[str, dict[str, str]], Counter[str]]:
    rows: dict[str, dict[str, str]] = {}
    run_counts: Counter[str] = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_exact_header(reader.fieldnames, REVERSE_COLUMNS, str(path))
        for line_number, row in enumerate(reader, start=2):
            require_complete_dict_row(row, REVERSE_COLUMNS, f"{path}:{line_number}")
            seq_id = row["seq_id"]
            if not seq_id:
                raise ValueError(f"{path}:{line_number}: empty seq_id")
            if seq_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate case-sensitive seq_id {seq_id}")
            expected_seq_id = f'{row["pdb_id"]}__{row["chain_id"]}'
            if seq_id != expected_seq_id:
                raise ValueError(
                    f"{path}:{line_number}: seq_id {seq_id!r} does not equal {expected_seq_id!r}"
                )
            if not RUN_ID_RE.fullmatch(row["run_id"]):
                raise ValueError(f"{path}:{line_number}: unsafe run_id {row['run_id']!r}")
            if not SHA256_RE.fullmatch(row["seq_sha256"]):
                raise ValueError(f"{path}:{line_number}: invalid SHA256 text")
            parse_positive_int(row["length"], f"{path}:{line_number}: length")
            rows[seq_id] = row
            run_counts[row["run_id"]] += 1
    if not rows:
        raise ValueError(f"{path}: empty reverse map")
    return rows, run_counts


def read_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_header(reader.fieldnames, LABEL_COLUMNS, str(path))
        for line_number, row in enumerate(reader, start=2):
            require_complete_dict_row(row, LABEL_COLUMNS, f"{path}:{line_number}")
            if not row["seq_id"]:
                raise ValueError(f"{path}:{line_number}: empty seq_id")
            rows[row["seq_id"]].append(row)
    if not rows:
        raise ValueError(f"{path}: empty labels")
    for seq_id, chain_rows in rows.items():
        try:
            chain_rows.sort(key=lambda row: int(row["row_index"]))
        except ValueError as exc:
            raise ValueError(f"{path}: {seq_id} has non-integer row_index") from exc
        for expected_index, row in enumerate(chain_rows):
            if int(row["row_index"]) != expected_index:
                raise ValueError(f"{path}: {seq_id} non-contiguous/duplicate row_index at {expected_index}")
            if int(row["position"]) != expected_index + 1:
                raise ValueError(f"{path}: {seq_id} non-contiguous 1-based position at {expected_index + 1}")
            if not re.fullmatch(r"[A-Z]", row["residue"]):
                raise ValueError(f"{path}: {seq_id} invalid residue at position {expected_index + 1}")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"{path}: {seq_id} non-binary label at position {expected_index + 1}")
    return dict(rows)


def validate_canonical_universe(
    manifest: dict[str, dict[str, str]],
    reverse: dict[str, dict[str, str]],
    reverse_counts: Counter[str],
    labels: dict[str, list[dict[str, str]]],
) -> None:
    n_label_residues = sum(len(chain_rows) for chain_rows in labels.values())
    if len(manifest) != EXPECTED_UNIQUE:
        raise ValueError(f"fixed universe requires {EXPECTED_UNIQUE} unique sequences, got {len(manifest)}")
    if len(reverse) != EXPECTED_CHAINS or len(labels) != EXPECTED_CHAINS:
        raise ValueError(
            f"fixed universe requires {EXPECTED_CHAINS} case-sensitive chains; "
            f"reverse={len(reverse)} labels={len(labels)}"
        )
    if n_label_residues != EXPECTED_RESIDUES:
        raise ValueError(
            f"fixed universe requires {EXPECTED_RESIDUES} label residues, got {n_label_residues}"
        )
    if set(reverse) != set(labels):
        only_reverse = sorted(set(reverse) - set(labels))[:20]
        only_labels = sorted(set(labels) - set(reverse))[:20]
        raise ValueError(
            "label/reverse-map case-sensitive chain universe mismatch: "
            f"only_reverse={only_reverse} only_labels={only_labels}"
        )
    if set(reverse_counts) != set(manifest):
        only_reverse = sorted(set(reverse_counts) - set(manifest))[:20]
        only_manifest = sorted(set(manifest) - set(reverse_counts))[:20]
        raise ValueError(
            "manifest/reverse-map run_id universe mismatch: "
            f"only_reverse={only_reverse} only_manifest={only_manifest}"
        )

    for run_id, manifest_row in manifest.items():
        if reverse_counts[run_id] != int(manifest_row["n_chains"]):
            raise ValueError(f"{run_id}: reverse-map n_chains mismatch")
        representative = manifest_row["representative_seq_id"]
        if representative not in reverse or reverse[representative]["run_id"] != run_id:
            raise ValueError(f"{run_id}: representative_seq_id is not mapped back to this run")

    for seq_id, mapping in reverse.items():
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
        if sha256_text(label_sequence) != mapping["seq_sha256"]:
            raise ValueError(f"{seq_id}: label/reverse-map SHA256 mismatch")


def validate_score_pattern(pattern: str) -> None:
    if pattern.count("{run_id}") != 1:
        raise ValueError("--score-pattern must contain {run_id} exactly once")
    remainder = pattern.replace("{run_id}", "")
    if "{" in remainder or "}" in remainder:
        raise ValueError("--score-pattern contains an unsupported format field")
    rendered = Path(pattern.format(run_id="dtm0001"))
    if rendered.is_absolute() or ".." in rendered.parts:
        raise ValueError("--score-pattern must remain a relative path below --score-dir")


def score_path(score_dir: Path, pattern: str, run_id: str) -> Path:
    return score_dir / pattern.format(run_id=run_id)


def parse_smoke_selection(args: argparse.Namespace, manifest: dict[str, dict[str, str]]) -> list[str]:
    if args.allow_partial:
        if not args.smoke_run_ids:
            raise ValueError("--allow-partial requires explicit --smoke-run-ids")
        if "smoke" not in args.output_tsv.name.lower() or "smoke" not in args.summary_json.name.lower():
            raise ValueError("partial outputs must contain 'smoke' in both output filenames")
        selected = [item.strip() for item in args.smoke_run_ids.split(",") if item.strip()]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("--smoke-run-ids must be non-empty and contain no duplicates")
        unknown = sorted(set(selected) - set(manifest))
        if unknown:
            raise ValueError(f"unknown smoke run IDs: {unknown}")
        return sorted(selected)
    if args.smoke_run_ids:
        raise ValueError("--smoke-run-ids is only valid with --allow-partial")
    return sorted(manifest)


def read_rf_score_file(path: Path, expected_sequence: str) -> tuple[list[int], str]:
    votes: list[int] = []
    residues: list[str] = []
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: RF score CSV is not valid UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path}: empty RF score file") from exc
        if tuple(header) != RF_SCORE_COLUMNS:
            raise ValueError(
                f"{path}: unexpected RF score columns {header}; expected {list(RF_SCORE_COLUMNS)}"
            )
        for expected_position, fields in enumerate(reader, start=1):
            if len(fields) != len(RF_SCORE_COLUMNS):
                raise ValueError(
                    f"{path}:{expected_position + 1}: expected four CSV fields, got {len(fields)}"
                )
            # R write.csv emits a blank header for this column, then numeric row names.
            row_name, residue, position_text, score_text = fields
            try:
                row_name_value = int(row_name)
                position = int(position_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{expected_position + 1}: invalid row/position integer") from exc
            if row_name_value != expected_position:
                raise ValueError(f"{path}:{expected_position + 1}: non-contiguous R row name")
            if position != expected_position:
                raise ValueError(f"{path}:{expected_position + 1}: non-contiguous 1-based AA position")
            if not re.fullmatch(r"[A-Z]", residue):
                raise ValueError(f"{path}:{expected_position + 1}: invalid Res residue {residue!r}")
            try:
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{expected_position + 1}: invalid score {score_text!r}") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{path}:{expected_position + 1}: score outside finite [0,1]")
            scaled = score * RF_TREES
            rounded_votes = round(scaled)
            if abs(scaled - rounded_votes) > SCORE_GRID_ATOL:
                raise ValueError(
                    f"{path}:{expected_position + 1}: score {score_text!r} is not on the 1/{RF_TREES} grid"
                )
            if not 0 <= rounded_votes <= RF_TREES:
                raise ValueError(f"{path}:{expected_position + 1}: invalid RF vote count")
            residues.append(residue)
            votes.append(rounded_votes)
    observed_sequence = "".join(residues)
    if observed_sequence != expected_sequence:
        raise ValueError(
            f"{path}: exact score/manifest sequence mismatch "
            f"(score_len={len(observed_sequence)}, expected_len={len(expected_sequence)})"
        )
    return votes, hashlib.sha256(raw).hexdigest()


def score_bundle_sha256(score_hashes: dict[str, str]) -> str:
    canonical = "".join(f"{run_id}\t{score_hashes[run_id]}\n" for run_id in sorted(score_hashes))
    return sha256_text(canonical)


def main() -> None:
    args = parse_args()
    validate_score_pattern(args.score_pattern)
    if not args.method or any(char in args.method for char in "\t\r\n"):
        raise ValueError("--method must be non-empty and contain no tab/newline")
    if args.output_tsv.resolve() == args.summary_json.resolve():
        raise ValueError("--output-tsv and --summary-json must be different paths")

    manifest = read_unique_manifest(args.unique_manifest)
    reverse, reverse_counts = read_reverse_map(args.reverse_map)
    labels = read_labels(args.labels)
    validate_canonical_universe(manifest, reverse, reverse_counts, labels)
    selected_run_ids = parse_smoke_selection(args, manifest)
    selected_run_set = set(selected_run_ids)

    all_score_paths = {
        run_id: score_path(args.score_dir, args.score_pattern, run_id) for run_id in manifest
    }
    missing_selected = [run_id for run_id in selected_run_ids if not all_score_paths[run_id].is_file()]
    if missing_selected:
        mode = "smoke" if args.allow_partial else "final"
        raise FileNotFoundError(
            f"{mode} materialization lacks {len(missing_selected)} selected RF score files; "
            f"first={missing_selected[:20]}"
        )
    available_run_ids = sorted(run_id for run_id, path in all_score_paths.items() if path.is_file())
    if not args.allow_partial and len(available_run_ids) != EXPECTED_UNIQUE:
        missing = sorted(set(manifest) - set(available_run_ids))
        raise FileNotFoundError(
            f"final materialization requires all {EXPECTED_UNIQUE} RF score files; "
            f"missing={len(missing)} first={missing[:20]}"
        )

    score_votes: dict[str, list[int]] = {}
    score_hashes: dict[str, str] = {}
    for run_id in selected_run_ids:
        votes, file_sha256 = read_rf_score_file(
            all_score_paths[run_id], manifest[run_id]["sequence"]
        )
        score_votes[run_id] = votes
        score_hashes[run_id] = file_sha256

    selected_seq_ids = sorted(
        seq_id for seq_id, mapping in reverse.items() if mapping["run_id"] in selected_run_set
    )
    expected_rows = sum(len(labels[seq_id]) for seq_id in selected_seq_ids)
    if not args.allow_partial:
        if len(selected_run_ids) != EXPECTED_UNIQUE:
            raise ValueError("final output does not cover every unique sequence")
        if len(selected_seq_ids) != EXPECTED_CHAINS:
            raise ValueError("final output does not cover every case-sensitive chain")
        if expected_rows != EXPECTED_RESIDUES:
            raise ValueError("final expected residue count does not match fixed universe")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output_tsv.with_name(f"{args.output_tsv.name}.tmp.{os.getpid()}")
    summary_tmp = args.summary_json.with_name(f"{args.summary_json.name}.tmp.{os.getpid()}")
    if output_tmp.exists() or summary_tmp.exists():
        raise FileExistsError("refusing to overwrite stale temporary publication file")

    output_rows = 0
    try:
        with output_tmp.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
            writer.writeheader()
            for seq_id in selected_seq_ids:
                run_id = reverse[seq_id]["run_id"]
                chain_rows = labels[seq_id]
                votes = score_votes[run_id]
                if len(votes) != len(chain_rows):
                    raise ValueError(f"{seq_id}: score/label length mismatch")
                for row, vote_count in zip(chain_rows, votes, strict=True):
                    writer.writerow(
                        {
                            "seq_id": seq_id,
                            "position": row["position"],
                            "residue": row["residue"],
                            "score": format(vote_count / RF_TREES, ".17g"),
                            "method": args.method,
                        }
                    )
                    output_rows += 1
        if output_rows != expected_rows:
            raise ValueError(f"output row mismatch: {output_rows} != {expected_rows}")

        output_sha256 = sha256_file(output_tmp)
        canonical_inputs = {
            "labels": {"path": str(args.labels.resolve()), "sha256": sha256_file(args.labels)},
            "unique_manifest": {
                "path": str(args.unique_manifest.resolve()),
                "sha256": sha256_file(args.unique_manifest),
            },
            "reverse_map": {
                "path": str(args.reverse_map.resolve()),
                "sha256": sha256_file(args.reverse_map),
            },
        }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "method": args.method,
            "scope": "partial_smoke" if args.allow_partial else "complete",
            "fixed_universe": {
                "unique_sequences": EXPECTED_UNIQUE,
                "case_sensitive_chains": EXPECTED_CHAINS,
                "residue_rows": EXPECTED_RESIDUES,
            },
            "rf_contract": {
                "trees": RF_TREES,
                "positive_score_grid": f"1/{RF_TREES}",
                "grid_absolute_tolerance": SCORE_GRID_ATOL,
                "input_columns": list(RF_SCORE_COLUMNS),
            },
            "canonical_inputs": canonical_inputs,
            "score_dir": str(args.score_dir.resolve()),
            "score_pattern": args.score_pattern,
            "n_manifest_unique": len(manifest),
            "n_available_score_files": len(available_run_ids),
            "n_selected_unique": len(selected_run_ids),
            "n_output_chains": len(selected_seq_ids),
            "n_output_residues": output_rows,
            "selected_run_ids": selected_run_ids,
            "score_file_sha256": dict(sorted(score_hashes.items())),
            "score_bundle_sha256": score_bundle_sha256(score_hashes),
            "output_tsv": str(args.output_tsv.resolve()),
            "output_sha256": output_sha256,
            "publication_contract": (
                "This summary JSON is the commit marker; consumers must verify output_sha256 "
                "and the canonical/source score hashes before using the TSV."
            ),
            "qc": (
                "fixed 724/2392/1036256 canonical universe; case-sensitive chain mapping; "
                "exact sequence SHA256, 1-based position and residue alignment; no missing or "
                "duplicate chain/residue; finite [0,1] class-1 votes on the 1/500 RF grid"
            ),
        }
        summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        # The JSON is the commit marker. Invalidate any previous marker before
        # publishing a new TSV so interruption can only leave an uncommitted
        # TSV, never a stale marker that appears to bless it.
        args.summary_json.unlink(missing_ok=True)
        output_tmp.replace(args.output_tsv)
        summary_tmp.replace(args.summary_json)
        if sha256_file(args.output_tsv) != output_sha256:
            raise RuntimeError("published output SHA256 changed after atomic rename")
    finally:
        output_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
