#!/usr/bin/env python3
"""Build the 38 residue features consumed by the official MBPred RF model.

This is a compatibility implementation of ``MBPred.pm``.  Two apparent bugs
in the official implementation are intentionally retained:

* PSSM accumulators are not reset between sequence positions.  After each
  position they are divided by the MSA depth, then reused at the next one.
* CumStr column names do not describe the values written under them.  The
  checkpoint sees full-DI, segment-DI, full-MI, segment-MI in the columns named
  Diseq, Miseq, Diseg, Miseg (and likewise for their maxima).

Only Python's standard library is required.  Rows belonging to Phobius SIGNAL
features and one-residue I/M/O segments are omitted, matching the official
pipeline's eventual incomplete-row removal.

The official ``RF_test.R`` later removes ``Res_num`` and ``Res_name`` before
prediction.  Its CSV output unfortunately renames the residue letter ``Res``
and the one-based position ``AA``; downstream adapters must use those value
semantics rather than interpreting the names literally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


AA_ORDER = tuple("AILVFWYNCQMSTDERHKGP")
PHYSICAL_COLUMNS = (
    "hydrophobicity",
    "polarity",
    "charge",
    "residue_volume",
    "isoelectric_point",
    "steric_parameter",
)
CHECKPOINT_COLUMNS = (
    "Entropy",
    "Diseq",
    "Miseq",
    "Diseg",
    "Miseg",
    "Diseqmax",
    "Miseqmax",
    "Disegmax",
    "Misegmax",
    "RelPos1",
    "RelPos2",
    *AA_ORDER,
    *PHYSICAL_COLUMNS,
    "Seg",
)
OUTPUT_COLUMNS = ("Res_num", "Res_name", *CHECKPOINT_COLUMNS)

if len(CHECKPOINT_COLUMNS) != 38:
    raise RuntimeError(f"internal error: expected 38 features, got {len(CHECKPOINT_COLUMNS)}")


class FeatureBuildError(ValueError):
    """Raised when an input violates the official feature contract."""


@dataclass(frozen=True)
class Segment:
    segment_id: int
    label: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class CouplingAggregate:
    full_di_sum: tuple[float, ...]
    segment_di_sum: tuple[float | None, ...]
    full_mi_sum: tuple[float, ...]
    segment_mi_sum: tuple[float | None, ...]
    full_di_max: tuple[float, ...]
    segment_di_max: tuple[float | None, ...]
    full_mi_max: tuple[float, ...]
    segment_mi_max: tuple[float | None, ...]
    pair_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_single_fasta(path: Path) -> tuple[str, str]:
    identifier: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    raise FeatureBuildError(
                        f"{path}:{line_number}: expected one FASTA record"
                    )
                identifier = line[1:].split()[0]
                if not identifier:
                    raise FeatureBuildError(f"{path}:{line_number}: empty FASTA identifier")
                continue
            if identifier is None:
                raise FeatureBuildError(
                    f"{path}:{line_number}: sequence appears before FASTA header"
                )
            if any(char.isspace() for char in line):
                raise FeatureBuildError(f"{path}:{line_number}: whitespace inside sequence")
            chunks.append(line.upper())

    if identifier is None or not chunks:
        raise FeatureBuildError(f"{path}: empty FASTA")
    sequence = "".join(chunks)
    if not sequence.isalpha():
        raise FeatureBuildError(f"{path}: FASTA sequence contains non-letter characters")
    return identifier, sequence


def read_aaindex(path: Path) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    with path.open("r", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, 1):
            fields = raw.split()
            if not fields:
                continue
            if fields[0] == "PPtype":
                if tuple(fields[1:]) != PHYSICAL_COLUMNS:
                    raise FeatureBuildError(
                        f"{path}:{line_number}: unexpected AAindex header {fields[1:]}"
                    )
                continue
            if len(fields) != 7 or len(fields[0]) != 1:
                raise FeatureBuildError(
                    f"{path}:{line_number}: expected residue plus six properties"
                )
            residue = fields[0]
            if residue in result:
                raise FeatureBuildError(f"{path}:{line_number}: duplicate residue {residue}")
            try:
                values = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise FeatureBuildError(
                    f"{path}:{line_number}: non-numeric AAindex value"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise FeatureBuildError(f"{path}:{line_number}: non-finite AAindex value")
            result[residue] = values

    missing = sorted(set(AA_ORDER) - result.keys())
    extra = sorted(result.keys() - set(AA_ORDER))
    if missing or extra:
        raise FeatureBuildError(f"{path}: AAindex mismatch; missing={missing}, extra={extra}")
    return result


def historical_pssm_and_entropy(
    alignment_path: Path, sequence: str
) -> tuple[list[tuple[float, ...]], list[float], int, dict[str, int]]:
    """Reproduce the cumulative accumulator at MBPred.pm:119-168."""

    aa_to_index = {aa: index for index, aa in enumerate(AA_ORDER)}
    counts = [[0] * len(AA_ORDER) for _ in sequence]
    ignored: Counter[str] = Counter()
    depth = 0
    query: str | None = None

    with alignment_path.open("r", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = raw.rstrip("\r\n")
            if not row:
                raise FeatureBuildError(f"{alignment_path}:{line_number}: blank alignment row")
            if len(row) != len(sequence):
                raise FeatureBuildError(
                    f"{alignment_path}:{line_number}: alignment length {len(row)} != "
                    f"sequence length {len(sequence)}"
                )
            if query is None:
                query = row
            depth += 1
            for position, symbol in enumerate(row):
                aa_index = aa_to_index.get(symbol)
                if aa_index is None:
                    ignored[symbol] += 1
                else:
                    counts[position][aa_index] += 1

    if depth == 0:
        raise FeatureBuildError(f"{alignment_path}: empty alignment")
    if query != sequence:
        raise FeatureBuildError(
            f"{alignment_path}: first alignment row does not exactly match FASTA sequence"
        )

    accumulator = [0.0] * len(AA_ORDER)
    pssm: list[tuple[float, ...]] = []
    entropy: list[float] = []
    for position_counts in counts:
        for index, count in enumerate(position_counts):
            accumulator[index] = (accumulator[index] + count) / depth
        pssm.append(tuple(accumulator))
        entropy.append(
            -sum(value * math.log2(value) for value in accumulator if value != 0.0)
        )

    return pssm, entropy, depth, dict(sorted(ignored.items()))


_PHOBIUS_FEATURE_RE = re.compile(
    r"^FT\s+(?P<kind>DOMAIN|TRANSMEM|SIGNAL)\s+"
    r"(?P<start>\d+)\s+(?P<end>\d+)(?:\s+(?P<annotation>.*?))?\s*$"
)


def parse_phobius(
    path: Path, sequence_length: int
) -> tuple[list[Segment], list[int | None], list[str | None], set[int]]:
    raw_segments: list[tuple[str, int, int]] = []
    signal_positions: set[int] = set()
    signal_region_positions: set[int] = set()

    with path.open("r", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            match = _PHOBIUS_FEATURE_RE.match(line)
            if match is None:
                continue
            kind = match.group("kind")
            start = int(match.group("start"))
            end = int(match.group("end"))
            if not (1 <= start <= end <= sequence_length):
                raise FeatureBuildError(
                    f"{path}:{line_number}: invalid Phobius interval {start}-{end}"
                )

            if kind == "SIGNAL":
                signal_positions.update(range(start, end + 1))
                continue
            if kind == "TRANSMEM":
                label = "M"
            else:
                annotation = (match.group("annotation") or "").upper()
                if "NON CYTOPLASMIC" in annotation:
                    label = "O"
                elif "CYTOPLASMIC" in annotation:
                    label = "I"
                elif annotation in {"N-REGION.", "H-REGION.", "C-REGION."}:
                    signal_region_positions.update(range(start, end + 1))
                    continue
                else:
                    raise FeatureBuildError(
                        f"{path}:{line_number}: unrecognized DOMAIN annotation "
                        f"{match.group('annotation')!r}"
                    )
            raw_segments.append((label, start, end))

    signal_region_outside_signal = signal_region_positions - signal_positions
    if signal_region_outside_signal:
        preview = sorted(signal_region_outside_signal)[:10]
        raise FeatureBuildError(
            f"{path}: signal-region DOMAIN residues are outside SIGNAL; "
            f"first={preview}"
        )

    if not raw_segments:
        raise FeatureBuildError(f"{path}: no I/M/O Phobius segments")

    raw_segments.sort(key=lambda item: (item[1], item[2], item[0]))
    segment_by_position: list[int | None] = [None] * sequence_length
    label_by_position: list[str | None] = [None] * sequence_length
    segments: list[Segment] = []
    for segment_id, (label, start, end) in enumerate(raw_segments):
        segment = Segment(segment_id, label, start, end)
        segments.append(segment)
        for position in range(start, end + 1):
            index = position - 1
            if segment_by_position[index] is not None:
                raise FeatureBuildError(
                    f"{path}: overlapping I/M/O segments at residue {position}"
                )
            if position in signal_positions:
                raise FeatureBuildError(
                    f"{path}: SIGNAL overlaps I/M/O segment at residue {position}"
                )
            segment_by_position[index] = segment_id
            label_by_position[index] = label

    unexplained = [
        position
        for position, segment_id in enumerate(segment_by_position, 1)
        if segment_id is None and position not in signal_positions
    ]
    if unexplained:
        preview = unexplained[:10]
        raise FeatureBuildError(
            f"{path}: {len(unexplained)} residues are neither I/M/O nor SIGNAL; "
            f"first={preview}"
        )
    return segments, segment_by_position, label_by_position, signal_positions


def _push_top_eight(heap: list[float], value: float) -> None:
    if len(heap) < 8:
        heapq.heappush(heap, value)
    elif value > heap[0]:
        heapq.heapreplace(heap, value)


def _heap_sums(heaps: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(math.fsum(heap) for heap in heaps)


def _heap_maxima(
    heaps: Sequence[Sequence[float]], *, allow_empty: bool
) -> tuple[float | None, ...]:
    maxima: list[float | None] = []
    for heap in heaps:
        if heap:
            maxima.append(max(heap))
        elif allow_empty:
            maxima.append(None)
        else:
            raise FeatureBuildError("coupling matrix leaves a residue without full-chain pairs")
    return tuple(maxima)


def aggregate_couplings(
    path: Path, sequence: str, segment_by_position: Sequence[int | None]
) -> CouplingAggregate:
    length = len(sequence)
    if length < 2:
        raise FeatureBuildError("MBPred coupling features require a sequence of length >= 2")
    if len(segment_by_position) != length:
        raise FeatureBuildError("internal error: segment map length mismatch")

    full_di: list[list[float]] = [[] for _ in sequence]
    full_mi: list[list[float]] = [[] for _ in sequence]
    segment_di: list[list[float]] = [[] for _ in sequence]
    segment_mi: list[list[float]] = [[] for _ in sequence]
    pair_count = 0
    expected_pairs = length * (length - 1) // 2
    seen_pairs = bytearray((expected_pairs + 7) // 8)

    with path.open("r", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("Your"):
                continue
            fields = line.split()
            if len(fields) != 6:
                raise FeatureBuildError(
                    f"{path}:{line_number}: expected six FreeContact columns"
                )
            try:
                left = int(fields[0])
                right = int(fields[2])
                mi = float(fields[4])
                di = float(fields[5])
            except ValueError as exc:
                raise FeatureBuildError(
                    f"{path}:{line_number}: malformed coupling row"
                ) from exc
            if not (1 <= left < right <= length):
                raise FeatureBuildError(
                    f"{path}:{line_number}: expected 1 <= i < j <= {length}, got "
                    f"{left}, {right}"
                )
            # Triangular row-major index for (left, right), using 1-based input
            # positions. A bitset detects duplicates without imposing order or
            # retaining O(L^2) Python objects.
            preceding = (left - 1) * length - ((left - 1) * left) // 2
            pair_index = preceding + (right - left - 1)
            byte_index, bit_index = divmod(pair_index, 8)
            bit = 1 << bit_index
            if seen_pairs[byte_index] & bit:
                raise FeatureBuildError(
                    f"{path}:{line_number}: duplicate coupling pair ({left}, {right})"
                )
            seen_pairs[byte_index] |= bit
            if fields[1] != sequence[left - 1] or fields[3] != sequence[right - 1]:
                raise FeatureBuildError(
                    f"{path}:{line_number}: residue letters do not match FASTA"
                )
            if not (math.isfinite(mi) and math.isfinite(di)):
                raise FeatureBuildError(f"{path}:{line_number}: non-finite coupling score")

            left_index = left - 1
            right_index = right - 1
            for index in (left_index, right_index):
                _push_top_eight(full_di[index], di)
                _push_top_eight(full_mi[index], mi)

            left_segment = segment_by_position[left_index]
            if left_segment is not None and left_segment == segment_by_position[right_index]:
                for index in (left_index, right_index):
                    _push_top_eight(segment_di[index], di)
                    _push_top_eight(segment_mi[index], mi)
            pair_count += 1

    if pair_count != expected_pairs:
        raise FeatureBuildError(
            f"{path}: incomplete coupling matrix: {pair_count} pairs, expected "
            f"{expected_pairs}"
        )

    full_di_max = _heap_maxima(full_di, allow_empty=False)
    full_mi_max = _heap_maxima(full_mi, allow_empty=False)
    return CouplingAggregate(
        full_di_sum=_heap_sums(full_di),
        segment_di_sum=tuple(
            math.fsum(heap) if heap else None for heap in segment_di
        ),
        full_mi_sum=_heap_sums(full_mi),
        segment_mi_sum=tuple(
            math.fsum(heap) if heap else None for heap in segment_mi
        ),
        full_di_max=tuple(float(value) for value in full_di_max),
        segment_di_max=_heap_maxima(segment_di, allow_empty=True),
        full_mi_max=tuple(float(value) for value in full_mi_max),
        segment_mi_max=_heap_maxima(segment_mi, allow_empty=True),
        pair_count=pair_count,
    )


def _format_value(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeatureBuildError(f"refusing to write non-finite feature {value}")
        return format(value, ".17g")
    return str(value)


def _write_tsv_atomic(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _format_value(row[column]) for column in OUTPUT_COLUMNS})
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise
    return count


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise


def build_features(
    *,
    fasta_path: Path,
    alignment_path: Path,
    coupling_path: Path,
    phobius_path: Path,
    aaindex_path: Path,
    output_path: Path,
    qc_path: Path,
    require_full_residue_coverage: bool = False,
) -> dict[str, object]:
    protein_id, sequence = read_single_fasta(fasta_path)
    aaindex = read_aaindex(aaindex_path)
    unsupported = sorted(set(sequence) - aaindex.keys())
    if unsupported:
        raise FeatureBuildError(
            f"{fasta_path}: residues lack official AAindex properties: {unsupported}"
        )

    pssm, entropy, msa_depth, ignored_alignment_symbols = historical_pssm_and_entropy(
        alignment_path, sequence
    )
    segments, segment_by_position, label_by_position, signal_positions = parse_phobius(
        phobius_path, len(sequence)
    )
    couplings = aggregate_couplings(coupling_path, sequence, segment_by_position)
    segment_lookup = {segment.segment_id: segment for segment in segments}

    omitted_singleton_positions = [
        position
        for position, segment_id in enumerate(segment_by_position, 1)
        if segment_id is not None and segment_lookup[segment_id].length == 1
    ]
    omitted_positions = sorted(signal_positions | set(omitted_singleton_positions))
    if require_full_residue_coverage and omitted_positions:
        raise FeatureBuildError(
            f"official feature semantics omit {len(omitted_positions)} residues; "
            f"SIGNAL={len(signal_positions)}, singleton_segment="
            f"{len(omitted_singleton_positions)}"
        )

    rows: list[dict[str, object]] = []
    for index, residue in enumerate(sequence):
        position = index + 1
        segment_id = segment_by_position[index]
        if segment_id is None:
            if position not in signal_positions:
                raise FeatureBuildError(f"internal error: unexplained residue {position}")
            continue
        segment = segment_lookup[segment_id]
        if segment.length == 1:
            continue

        segment_di_sum = couplings.segment_di_sum[index]
        segment_mi_sum = couplings.segment_mi_sum[index]
        segment_di_max = couplings.segment_di_max[index]
        segment_mi_max = couplings.segment_mi_max[index]
        if None in (segment_di_sum, segment_mi_sum, segment_di_max, segment_mi_max):
            raise FeatureBuildError(
                f"residue {position}: non-singleton segment lacks coupling features"
            )

        row: dict[str, object] = {
            "Res_num": position,
            "Res_name": residue,
            "Entropy": entropy[index],
            # Preserve MBPred.pm:516/565 header/value mismatch.
            "Diseq": couplings.full_di_sum[index],
            "Miseq": segment_di_sum,
            "Diseg": couplings.full_mi_sum[index],
            "Miseg": segment_mi_sum,
            "Diseqmax": couplings.full_di_max[index],
            "Miseqmax": segment_di_max,
            "Disegmax": couplings.full_mi_max[index],
            "Misegmax": segment_mi_max,
            "RelPos1": (position - segment.start + 1) / segment.length,
            "RelPos2": position / len(sequence),
            "Seg": label_by_position[index],
        }
        row.update(zip(AA_ORDER, pssm[index], strict=True))
        row.update(zip(PHYSICAL_COLUMNS, aaindex[residue], strict=True))
        if set(row) != set(OUTPUT_COLUMNS):
            missing = sorted(set(OUTPUT_COLUMNS) - row.keys())
            extra = sorted(row.keys() - set(OUTPUT_COLUMNS))
            raise FeatureBuildError(
                f"internal output schema mismatch: missing={missing}, extra={extra}"
            )
        rows.append(row)

    expected_output_rows = len(sequence) - len(omitted_positions)
    if len(rows) != expected_output_rows:
        raise FeatureBuildError(
            f"internal row count mismatch: {len(rows)} != {expected_output_rows}"
        )
    if not rows:
        raise FeatureBuildError("no complete MBPred feature rows remain")

    written_rows = _write_tsv_atomic(output_path, rows)
    if written_rows != expected_output_rows:
        raise FeatureBuildError(
            f"output row count mismatch: {written_rows} != {expected_output_rows}"
        )

    label_counts = Counter(label for label in label_by_position if label is not None)
    summary: dict[str, object] = {
        "schema_version": 1,
        "compatibility_target": "official MBPred.pm historical semantics",
        "protein_id": protein_id,
        "sequence_length": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "msa_depth": msa_depth,
        "alignment_ignored_symbol_counts": ignored_alignment_symbols,
        "coupling_pair_count": couplings.pair_count,
        "expected_coupling_pair_count": len(sequence) * (len(sequence) - 1) // 2,
        "segment_count": len(segments),
        "segment_label_residue_counts": dict(sorted(label_counts.items())),
        "signal_omitted_positions": sorted(signal_positions),
        "singleton_segment_omitted_positions": omitted_singleton_positions,
        "output_row_count": written_rows,
        "checkpoint_feature_count": len(CHECKPOINT_COLUMNS),
        "checkpoint_columns": list(CHECKPOINT_COLUMNS),
        "historical_quirks_preserved": [
            "cumulative_pssm_accumulator",
            "cumstr_header_value_order_mismatch",
        ],
        "inputs": {
            "fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
            "alignment": {
                "path": str(alignment_path),
                "sha256": sha256_file(alignment_path),
            },
            "couplings": {
                "path": str(coupling_path),
                "sha256": sha256_file(coupling_path),
            },
            "phobius": {"path": str(phobius_path), "sha256": sha256_file(phobius_path)},
            "aaindex": {"path": str(aaindex_path), "sha256": sha256_file(aaindex_path)},
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    _write_json_atomic(qc_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path, help="gap-preserving .aln")
    parser.add_argument("--couplings", required=True, type=Path, help="six-column .evfold")
    parser.add_argument("--phobius", required=True, type=Path, help="Phobius long output")
    parser.add_argument("--aaindex", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--qc-json",
        type=Path,
        help="default: <output>.qc.json",
    )
    parser.add_argument(
        "--require-full-residue-coverage",
        action="store_true",
        help="fail instead of faithfully omitting SIGNAL/singleton-segment rows",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    qc_path = args.qc_json or Path(f"{args.output}.qc.json")
    summary = build_features(
        fasta_path=args.fasta,
        alignment_path=args.alignment,
        coupling_path=args.couplings,
        phobius_path=args.phobius,
        aaindex_path=args.aaindex,
        output_path=args.output,
        qc_path=qc_path,
        require_full_residue_coverage=args.require_full_residue_coverage,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
