#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator


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
SUMMARY_KEYS = {
    "schema",
    "method",
    "scope",
    "fixed_universe",
    "rf_contract",
    "canonical_inputs",
    "score_dir",
    "score_pattern",
    "n_manifest_unique",
    "n_available_score_files",
    "n_selected_unique",
    "n_output_chains",
    "n_output_residues",
    "selected_run_ids",
    "score_file_sha256",
    "score_bundle_sha256",
    "output_tsv",
    "output_sha256",
    "publication_contract",
    "qc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently validate a materialized MBPred-adapted prediction TSV against "
            "the canonical universe and its per-unique RF score files."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--unique-manifest", type=Path, required=True)
    parser.add_argument("--reverse-map", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--score-pattern", default="{run_id}.csv")
    parser.add_argument("--predictions-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-method", default="MBPred_adapted")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Validate an explicitly allowlisted smoke subset instead of a publishable full run.",
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


def require_header(actual: list[str] | None, expected: tuple[str, ...], context: str) -> None:
    if tuple(actual or ()) != expected:
        raise ValueError(f"{context}: unexpected columns {actual}; expected {list(expected)}")


def require_complete_row(
    row: dict[str | None, str | list[str] | None], expected: tuple[str, ...], context: str
) -> None:
    if set(row) != set(expected) or any(row[column] is None for column in expected):
        raise ValueError(f"{context}: missing or extra delimited field")


def positive_int(text: str, context: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{context}: expected integer, got {text!r}") from exc
    if value <= 0:
        raise ValueError(f"{context}: expected positive integer, got {value}")
    return value


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_header(reader.fieldnames, UNIQUE_COLUMNS, str(path))
        for line_number, row in enumerate(reader, start=2):
            require_complete_row(row, UNIQUE_COLUMNS, f"{path}:{line_number}")
            run_id = row["run_id"]
            if not RUN_ID_RE.fullmatch(run_id):
                raise ValueError(f"{path}:{line_number}: unsafe run_id {run_id!r}")
            if run_id in manifest:
                raise ValueError(f"{path}:{line_number}: duplicate run_id {run_id}")
            if row["prot_chain"] != "A":
                raise ValueError(f"{path}:{line_number}: {run_id} must use synthetic chain A")
            sequence = row["sequence"]
            if not SEQUENCE_RE.fullmatch(sequence):
                raise ValueError(f"{path}:{line_number}: invalid sequence")
            length = positive_int(row["length"], f"{path}:{line_number}: length")
            positive_int(row["n_chains"], f"{path}:{line_number}: n_chains")
            if length != len(sequence):
                raise ValueError(f"{path}:{line_number}: length/sequence mismatch")
            if not SHA256_RE.fullmatch(row["seq_sha256"]):
                raise ValueError(f"{path}:{line_number}: invalid SHA256 text")
            if sha256_text(sequence) != row["seq_sha256"]:
                raise ValueError(f"{path}:{line_number}: sequence SHA256 mismatch")
            if not row["representative_seq_id"]:
                raise ValueError(f"{path}:{line_number}: empty representative_seq_id")
            manifest[run_id] = row
    if not manifest:
        raise ValueError(f"{path}: empty manifest")
    return manifest


def read_reverse(path: Path) -> tuple[dict[str, dict[str, str]], Counter[str]]:
    reverse: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_header(reader.fieldnames, REVERSE_COLUMNS, str(path))
        for line_number, row in enumerate(reader, start=2):
            require_complete_row(row, REVERSE_COLUMNS, f"{path}:{line_number}")
            seq_id = row["seq_id"]
            if not seq_id or seq_id in reverse:
                raise ValueError(f"{path}:{line_number}: empty/duplicate case-sensitive seq_id {seq_id!r}")
            if seq_id != f'{row["pdb_id"]}__{row["chain_id"]}':
                raise ValueError(f"{path}:{line_number}: PDB/chain identity mismatch for {seq_id}")
            if not RUN_ID_RE.fullmatch(row["run_id"]):
                raise ValueError(f"{path}:{line_number}: unsafe run_id")
            if not SHA256_RE.fullmatch(row["seq_sha256"]):
                raise ValueError(f"{path}:{line_number}: invalid SHA256 text")
            positive_int(row["length"], f"{path}:{line_number}: length")
            reverse[seq_id] = row
            counts[row["run_id"]] += 1
    if not reverse:
        raise ValueError(f"{path}: empty reverse map")
    return reverse, counts


def read_label_rows(path: Path) -> dict[str, list[tuple[int, str]]]:
    labels: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_header(reader.fieldnames, LABEL_COLUMNS, str(path))
        for line_number, row in enumerate(reader, start=2):
            require_complete_row(row, LABEL_COLUMNS, f"{path}:{line_number}")
            seq_id = row["seq_id"]
            if not seq_id:
                raise ValueError(f"{path}:{line_number}: empty seq_id")
            try:
                row_index = int(row["row_index"])
                position = int(row["position"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-integer row/position") from exc
            if not re.fullmatch(r"[A-Z]", row["residue"]):
                raise ValueError(f"{path}:{line_number}: invalid residue")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number}: non-binary label")
            labels[seq_id].append((row_index, row["residue"], position))
    if not labels:
        raise ValueError(f"{path}: empty labels")

    normalized: dict[str, list[tuple[int, str]]] = {}
    for seq_id, chain_rows in labels.items():
        chain_rows.sort(key=lambda item: item[0])
        normalized_rows: list[tuple[int, str]] = []
        for expected_index, (row_index, residue, position) in enumerate(chain_rows):
            if row_index != expected_index or position != expected_index + 1:
                raise ValueError(f"{path}: {seq_id} has duplicate/non-contiguous residue coordinates")
            normalized_rows.append((position, residue))
        normalized[seq_id] = normalized_rows
    return normalized


def validate_universe(
    manifest: dict[str, dict[str, str]],
    reverse: dict[str, dict[str, str]],
    reverse_counts: Counter[str],
    labels: dict[str, list[tuple[int, str]]],
) -> None:
    residue_count = sum(len(rows) for rows in labels.values())
    if (len(manifest), len(reverse), len(labels), residue_count) != (
        EXPECTED_UNIQUE,
        EXPECTED_CHAINS,
        EXPECTED_CHAINS,
        EXPECTED_RESIDUES,
    ):
        raise ValueError(
            "canonical hard-count mismatch: "
            f"unique={len(manifest)} reverse_chains={len(reverse)} "
            f"label_chains={len(labels)} residues={residue_count}"
        )
    if set(reverse) != set(labels):
        raise ValueError("case-sensitive reverse-map and label chain universes differ")
    if set(reverse_counts) != set(manifest):
        raise ValueError("reverse-map and manifest run_id universes differ")

    for run_id, row in manifest.items():
        if reverse_counts[run_id] != int(row["n_chains"]):
            raise ValueError(f"{run_id}: n_chains mismatch")
        representative = row["representative_seq_id"]
        if representative not in reverse or reverse[representative]["run_id"] != run_id:
            raise ValueError(f"{run_id}: representative_seq_id mapping mismatch")

    for seq_id, mapping in reverse.items():
        manifest_row = manifest[mapping["run_id"]]
        sequence = "".join(residue for _, residue in labels[seq_id])
        if int(mapping["length"]) != len(sequence):
            raise ValueError(f"{seq_id}: reverse-map/label length mismatch")
        if sequence != manifest_row["sequence"]:
            raise ValueError(f"{seq_id}: label/manifest sequence mismatch")
        if mapping["seq_sha256"] != manifest_row["seq_sha256"]:
            raise ValueError(f"{seq_id}: reverse-map/manifest hash mismatch")
        if sha256_text(sequence) != mapping["seq_sha256"]:
            raise ValueError(f"{seq_id}: label/reverse-map hash mismatch")


def validate_pattern(pattern: str) -> None:
    if pattern.count("{run_id}") != 1:
        raise ValueError("--score-pattern must contain {run_id} exactly once")
    remainder = pattern.replace("{run_id}", "")
    if "{" in remainder or "}" in remainder:
        raise ValueError("--score-pattern contains an unsupported format field")
    rendered = Path(pattern.format(run_id="dtm0001"))
    if rendered.is_absolute() or ".." in rendered.parts:
        raise ValueError("--score-pattern must remain below --score-dir")


def select_run_ids(args: argparse.Namespace, manifest: dict[str, dict[str, str]]) -> list[str]:
    if args.allow_partial:
        if not args.smoke_run_ids:
            raise ValueError("--allow-partial requires explicit --smoke-run-ids")
        if (
            "smoke" not in args.predictions_tsv.name.lower()
            or "smoke" not in args.summary_json.name.lower()
        ):
            raise ValueError("partial inputs must contain 'smoke' in both filenames")
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


def score_to_votes(score_text: str, context: str) -> int:
    try:
        score = float(score_text)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid score {score_text!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{context}: score outside finite [0,1]")
    scaled = score * RF_TREES
    votes = round(scaled)
    if abs(scaled - votes) > SCORE_GRID_ATOL:
        raise ValueError(f"{context}: score is not on the 1/{RF_TREES} RF grid")
    if not 0 <= votes <= RF_TREES:
        raise ValueError(f"{context}: vote count outside [0,{RF_TREES}]")
    return votes


def read_score(path: Path, expected_sequence: str) -> tuple[list[int], str]:
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
            raise ValueError(f"{path}: empty score file") from exc
        if tuple(header) != RF_SCORE_COLUMNS:
            raise ValueError(f"{path}: unexpected RF score columns {header}")
        for expected_position, fields in enumerate(reader, start=1):
            if len(fields) != 4:
                raise ValueError(f"{path}:{expected_position + 1}: expected four fields")
            # The first header is blank because R writes numeric row names into that column.
            row_name, residue, position_text, score_text = fields
            try:
                row_number = int(row_name)
                position = int(position_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{expected_position + 1}: invalid row/position") from exc
            if row_number != expected_position or position != expected_position:
                raise ValueError(f"{path}:{expected_position + 1}: non-contiguous coordinates")
            if not re.fullmatch(r"[A-Z]", residue):
                raise ValueError(f"{path}:{expected_position + 1}: invalid residue")
            residues.append(residue)
            votes.append(score_to_votes(score_text, f"{path}:{expected_position + 1}"))
    if "".join(residues) != expected_sequence:
        raise ValueError(f"{path}: exact score/manifest sequence mismatch")
    return votes, hashlib.sha256(raw).hexdigest()


def bundle_hash(score_hashes: dict[str, str]) -> str:
    text = "".join(f"{run_id}\t{score_hashes[run_id]}\n" for run_id in sorted(score_hashes))
    return sha256_text(text)


def expected_rows(
    selected_seq_ids: list[str],
    reverse: dict[str, dict[str, str]],
    labels: dict[str, list[tuple[int, str]]],
    score_votes: dict[str, list[int]],
) -> Iterator[tuple[str, int, str, int]]:
    for seq_id in selected_seq_ids:
        run_id = reverse[seq_id]["run_id"]
        chain_rows = labels[seq_id]
        votes = score_votes[run_id]
        if len(chain_rows) != len(votes):
            raise ValueError(f"{seq_id}: score/label length mismatch")
        for (position, residue), vote_count in zip(chain_rows, votes, strict=True):
            yield seq_id, position, residue, vote_count


def validate_summary(
    summary_path: Path,
    args: argparse.Namespace,
    selected_run_ids: list[str],
    selected_seq_ids: list[str],
    output_rows: int,
    available_score_files: int,
    score_hashes: dict[str, str],
    output_sha256: str,
) -> dict[str, object]:
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or set(summary) != SUMMARY_KEYS:
        actual_keys = set(summary) if isinstance(summary, dict) else set()
        raise ValueError(
            f"{summary_path}: unexpected summary keys; "
            f"missing={sorted(SUMMARY_KEYS - actual_keys)} extra={sorted(actual_keys - SUMMARY_KEYS)}"
        )
    expected_scope = "partial_smoke" if args.allow_partial else "complete"
    scalar_expected = {
        "schema": SUMMARY_SCHEMA,
        "method": args.expected_method,
        "scope": expected_scope,
        "score_dir": str(args.score_dir.resolve()),
        "score_pattern": args.score_pattern,
        "n_manifest_unique": EXPECTED_UNIQUE,
        "n_available_score_files": available_score_files,
        "n_selected_unique": len(selected_run_ids),
        "n_output_chains": len(selected_seq_ids),
        "n_output_residues": output_rows,
        "score_bundle_sha256": bundle_hash(score_hashes),
        "output_tsv": str(args.predictions_tsv.resolve()),
        "output_sha256": output_sha256,
    }
    for key, expected in scalar_expected.items():
        if summary[key] != expected:
            raise ValueError(f"{summary_path}: {key} mismatch: {summary[key]!r} != {expected!r}")

    expected_universe = {
        "unique_sequences": EXPECTED_UNIQUE,
        "case_sensitive_chains": EXPECTED_CHAINS,
        "residue_rows": EXPECTED_RESIDUES,
    }
    if summary["fixed_universe"] != expected_universe:
        raise ValueError(f"{summary_path}: fixed_universe mismatch")
    expected_rf_contract = {
        "trees": RF_TREES,
        "positive_score_grid": f"1/{RF_TREES}",
        "grid_absolute_tolerance": SCORE_GRID_ATOL,
        "input_columns": list(RF_SCORE_COLUMNS),
    }
    if summary["rf_contract"] != expected_rf_contract:
        raise ValueError(f"{summary_path}: rf_contract mismatch")
    expected_inputs = {
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
    if summary["canonical_inputs"] != expected_inputs:
        raise ValueError(f"{summary_path}: canonical input path/hash mismatch")
    if summary["selected_run_ids"] != selected_run_ids:
        raise ValueError(f"{summary_path}: selected_run_ids mismatch")
    if summary["score_file_sha256"] != dict(sorted(score_hashes.items())):
        raise ValueError(f"{summary_path}: per-score SHA256 mapping mismatch")
    if not isinstance(summary["publication_contract"], str) or "output_sha256" not in summary[
        "publication_contract"
    ]:
        raise ValueError(f"{summary_path}: invalid publication contract")
    if not isinstance(summary["qc"], str) or "1/500" not in summary["qc"]:
        raise ValueError(f"{summary_path}: invalid QC declaration")
    return summary


def main() -> None:
    args = parse_args()
    validate_pattern(args.score_pattern)
    if not args.expected_method or any(char in args.expected_method for char in "\t\r\n"):
        raise ValueError("--expected-method must be non-empty and contain no tab/newline")
    if not args.predictions_tsv.is_file() or not args.summary_json.is_file():
        raise FileNotFoundError("prediction TSV and summary JSON must both exist")

    manifest = read_manifest(args.unique_manifest)
    reverse, reverse_counts = read_reverse(args.reverse_map)
    labels = read_label_rows(args.labels)
    validate_universe(manifest, reverse, reverse_counts, labels)
    selected_run_ids = select_run_ids(args, manifest)
    selected_set = set(selected_run_ids)

    paths = {
        run_id: args.score_dir / args.score_pattern.format(run_id=run_id) for run_id in manifest
    }
    available_score_files = sum(path.is_file() for path in paths.values())
    missing = [run_id for run_id in selected_run_ids if not paths[run_id].is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} selected RF score files; first={missing[:20]}")
    if not args.allow_partial and available_score_files != EXPECTED_UNIQUE:
        raise FileNotFoundError(
            f"full validation requires {EXPECTED_UNIQUE} RF score files, got {available_score_files}"
        )

    score_votes: dict[str, list[int]] = {}
    score_hashes: dict[str, str] = {}
    for run_id in selected_run_ids:
        votes, digest = read_score(paths[run_id], manifest[run_id]["sequence"])
        score_votes[run_id] = votes
        score_hashes[run_id] = digest

    selected_seq_ids = sorted(
        seq_id for seq_id, row in reverse.items() if row["run_id"] in selected_set
    )
    expected_row_count = sum(len(labels[seq_id]) for seq_id in selected_seq_ids)
    if not args.allow_partial and (
        len(selected_run_ids), len(selected_seq_ids), expected_row_count
    ) != (EXPECTED_UNIQUE, EXPECTED_CHAINS, EXPECTED_RESIDUES):
        raise ValueError("full selected universe does not satisfy canonical hard counts")

    output_sha256 = sha256_file(args.predictions_tsv)
    output_rows = 0
    observed_chains: set[str] = set()
    expected_iterator = expected_rows(selected_seq_ids, reverse, labels, score_votes)
    with args.predictions_tsv.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{args.predictions_tsv}: empty prediction TSV") from exc
        if tuple(header) != OUTPUT_COLUMNS:
            raise ValueError(f"{args.predictions_tsv}: unexpected output columns {header}")
        sentinel = object()
        for output_rows, pair in enumerate(
            itertools.zip_longest(expected_iterator, reader, fillvalue=sentinel), start=1
        ):
            expected, observed = pair
            line_number = output_rows + 1
            if expected is sentinel:
                raise ValueError(f"{args.predictions_tsv}:{line_number}: unexpected extra residue row")
            if observed is sentinel:
                raise ValueError(f"{args.predictions_tsv}:{line_number}: missing residue row")
            if len(observed) != len(OUTPUT_COLUMNS):
                raise ValueError(f"{args.predictions_tsv}:{line_number}: expected five TSV fields")
            seq_id, position_text, residue, score_text, method = observed
            expected_seq_id, expected_position, expected_residue, expected_votes = expected
            try:
                position = int(position_text)
            except ValueError as exc:
                raise ValueError(f"{args.predictions_tsv}:{line_number}: invalid position") from exc
            observed_votes = score_to_votes(score_text, f"{args.predictions_tsv}:{line_number}")
            if (seq_id, position, residue, observed_votes) != (
                expected_seq_id,
                expected_position,
                expected_residue,
                expected_votes,
            ):
                raise ValueError(
                    f"{args.predictions_tsv}:{line_number}: prediction/source mismatch; "
                    f"observed={(seq_id, position, residue, observed_votes)!r} "
                    f"expected={(expected_seq_id, expected_position, expected_residue, expected_votes)!r}"
                )
            if method != args.expected_method:
                raise ValueError(f"{args.predictions_tsv}:{line_number}: method mismatch")
            observed_chains.add(seq_id)

    if output_rows != expected_row_count:
        raise ValueError(f"output row count mismatch: {output_rows} != {expected_row_count}")
    if observed_chains != set(selected_seq_ids):
        raise ValueError("output has a missing or unexpected case-sensitive chain")
    if sha256_file(args.predictions_tsv) != output_sha256:
        raise RuntimeError("prediction TSV changed while it was being validated")

    validate_summary(
        args.summary_json,
        args,
        selected_run_ids,
        selected_seq_ids,
        output_rows,
        available_score_files,
        score_hashes,
        output_sha256,
    )
    result = {
        "status": "pass",
        "scope": "partial_smoke" if args.allow_partial else "complete",
        "method": args.expected_method,
        "n_unique": len(selected_run_ids),
        "n_chains": len(selected_seq_ids),
        "n_residues": output_rows,
        "output_sha256": output_sha256,
        "score_bundle_sha256": bundle_hash(score_hashes),
        "checks": (
            "canonical hard counts; exact case-sensitive mapping; independent source-score "
            "reconstruction; no duplicate/missing residues; finite 1/500-grid scores; commit marker"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
