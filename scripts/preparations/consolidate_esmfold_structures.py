#!/usr/bin/env python3
"""Consolidate ESMFold PDBs by exact input sequence with reversible mappings."""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V", "UNK": "X",
}


@dataclass(frozen=True)
class PDBInfo:
    sequence: str
    seq_sha256: str
    file_sha256: str
    chain_ids: str
    mean_plddt_ca: float
    min_plddt_ca: float
    max_plddt_ca: float


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdb(path: Path) -> PDBInfo:
    residues: list[str] = []
    chain_order: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    plddt: list[float] = []
    with path.open("rt", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue
            chain = line[21:22].strip() or "_"
            residue_key = (chain, line[22:26].strip(), line[26:27].strip())
            if residue_key in seen:
                continue
            seen.add(residue_key)
            residue_name = line[17:20].strip().upper()
            residues.append(AA3_TO_1.get(residue_name, "X"))
            if chain not in chain_order:
                chain_order.append(chain)
            try:
                plddt.append(float(line[60:66]))
            except ValueError:
                pass
    if not residues:
        raise ValueError(f"No CA residues found in {path}")
    sequence = "".join(residues)
    scores = plddt or [float("nan")]
    return PDBInfo(
        sequence=sequence,
        seq_sha256=sha256_text(sequence),
        file_sha256=sha256_file(path),
        chain_ids=";".join(chain_order),
        mean_plddt_ca=mean(scores),
        min_plddt_ca=min(scores),
        max_plddt_ca=max(scores),
    )


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target) or sha256_file(source) == sha256_file(target):
            return "existing"
        raise FileExistsError(f"Existing target differs from source: {target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
        return "copy"


def expected_sequences(old_manifest: Path, supplement_manifest: Path) -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    for source_name, path, id_column in (
        ("legacy_unique", old_manifest, "unique_id"),
        ("strict_supplement", supplement_manifest, "missing_uid"),
    ):
        for row in read_rows(path, "\t"):
            source_id = row[id_column]
            sequence = row["sequence"].strip().upper()
            seq_sha256 = row["seq_sha256"].strip().lower()
            if sha256_text(sequence) != seq_sha256:
                raise ValueError(f"Manifest sequence hash mismatch: {path}:{source_id}")
            record = {
                "source_id": source_id,
                "source_name": source_name,
                "sequence": sequence,
                "seq_sha256": seq_sha256,
                "n_residues": str(len(sequence)),
            }
            previous = expected.get(source_id)
            if previous is not None and previous["seq_sha256"] != seq_sha256:
                raise ValueError(f"Conflicting source ID {source_id}")
            expected[source_id] = record
    return expected


def source_group(path: Path, old_root: Path, supplement_root: Path) -> tuple[str, int]:
    try:
        rel = path.relative_to(old_root)
        first = rel.parts[0] if len(rel.parts) > 1 else "root"
        priority = 0 if first == "union_pdb" else 2
        return f"legacy_{first}", priority
    except ValueError:
        rel = path.relative_to(supplement_root)
        first = rel.parts[0] if len(rel.parts) > 1 else "root"
        return f"strict_supplement_{first}", 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--old-pdb-root", required=True, type=Path)
    parser.add_argument("--supplement-manifest", required=True, type=Path)
    parser.add_argument("--supplement-pdb-root", required=True, type=Path)
    parser.add_argument("--all-chains", required=True, type=Path)
    parser.add_argument("--final-chain-manifest", required=True, type=Path)
    args = parser.parse_args()

    target = args.target.resolve()
    structures_dir = target / "structures"
    manifests_dir = target / "manifests"
    qc_dir = target / "qc"
    structures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    expected = expected_sequences(args.old_manifest, args.supplement_manifest)
    source_paths = sorted(args.old_pdb_root.rglob("*.pdb")) + sorted(args.supplement_pdb_root.rglob("*.pdb"))
    if not source_paths:
        raise ValueError("No ESMFold PDB files found")

    inode_cache: dict[tuple[int, int], PDBInfo] = {}
    inventory: list[dict[str, object]] = []
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    unmapped_sources: list[dict[str, object]] = []
    sequence_mismatches: list[dict[str, object]] = []

    for path in source_paths:
        stat = path.stat()
        inode_key = (stat.st_dev, stat.st_ino)
        info = inode_cache.get(inode_key)
        if info is None:
            info = inspect_pdb(path)
            inode_cache[inode_key] = info
        source_id = path.stem
        expected_record = expected.get(source_id)
        group, priority = source_group(path, args.old_pdb_root, args.supplement_pdb_root)
        expected_sha = expected_record["seq_sha256"] if expected_record else ""
        row: dict[str, object] = {
            "source_group": group,
            "source_id": source_id,
            "source_path": str(path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size_bytes": stat.st_size,
            "file_sha256": info.file_sha256,
            "expected_seq_sha256": expected_sha,
            "observed_seq_sha256": info.seq_sha256,
            "n_residues": len(info.sequence),
            "sequence_match": int(bool(expected_sha) and expected_sha == info.seq_sha256),
            "source_priority": priority,
        }
        inventory.append(row)
        if expected_record is None:
            unmapped_sources.append(row)
            continue
        if expected_sha != info.seq_sha256:
            sequence_mismatches.append(row)
            continue
        by_sequence[expected_sha].append({"path": path, "info": info, "source": row})

    if unmapped_sources or sequence_mismatches:
        write_tsv(qc_dir / "unmapped_sources.tsv", unmapped_sources, list(inventory[0]))
        write_tsv(qc_dir / "sequence_mismatches.tsv", sequence_mismatches, list(inventory[0]))
        raise ValueError(
            f"PDB QC failed: unmapped={len(unmapped_sources)} sequence_mismatches={len(sequence_mismatches)}"
        )

    canonical: dict[str, dict[str, object]] = {}
    conflicts: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    for seq_sha256, candidates in sorted(by_sequence.items()):
        candidates.sort(key=lambda item: (int(item["source"]["source_priority"]), str(item["path"])))
        primary = candidates[0]
        info = primary["info"]
        assert isinstance(info, PDBInfo)
        source_path = primary["path"]
        assert isinstance(source_path, Path)
        source_ids = sorted({str(item["source"]["source_id"]) for item in candidates})
        file_hashes = sorted({str(item["info"].file_sha256) for item in candidates})
        structure_id = f"esmfold_L{len(info.sequence):05d}_{seq_sha256[:16]}"
        target_path = structures_dir / f"{structure_id}.pdb"
        mode = link_or_copy(source_path, target_path)
        record: dict[str, object] = {
            "structure_id": structure_id,
            "target_path": str(target_path.relative_to(target)),
            "seq_sha256": seq_sha256,
            "n_residues": len(info.sequence),
            "source_ids": ";".join(source_ids),
            "source_occurrences": len(candidates),
            "distinct_file_models": len(file_hashes),
            "primary_source_group": primary["source"]["source_group"],
            "primary_source_path": str(source_path),
            "file_sha256": info.file_sha256,
            "pdb_chain_ids": info.chain_ids,
            "mean_plddt_ca": f"{info.mean_plddt_ca:.6f}",
            "min_plddt_ca": f"{info.min_plddt_ca:.6f}",
            "max_plddt_ca": f"{info.max_plddt_ca:.6f}",
            "storage_mode": mode,
        }
        canonical[seq_sha256] = record
        structure_rows.append(record)
        if len(file_hashes) > 1:
            conflicts.append(
                {
                    "seq_sha256": seq_sha256,
                    "structure_id": structure_id,
                    "distinct_file_models": len(file_hashes),
                    "file_sha256s": ";".join(file_hashes),
                    "source_paths": ";".join(str(item["path"]) for item in candidates),
                }
            )

    for row in inventory:
        record = canonical.get(str(row["expected_seq_sha256"]))
        row["structure_id"] = record["structure_id"] if record else ""
        row["target_path"] = record["target_path"] if record else ""
        row["selected_as_primary"] = int(
            bool(record) and str(row["source_path"]) == str(record["primary_source_path"])
        )

    all_chain_rows = read_rows(args.all_chains, "\t")
    all_chain_map: list[dict[str, object]] = []
    all_chain_index: dict[str, dict[str, str]] = {}
    folded_chain_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    pair_index: dict[tuple[str, str], dict[str, str]] = {}
    folded_pair_index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_chain_rows:
        seq_hash = row["seq_sha256"].lower()
        record = canonical.get(seq_hash)
        selected_seq_id = row.get("selected_seq_id", "")
        mapping: dict[str, object] = {
            "pdb_id": row["pdb_id"],
            "chain_id": row["chain_id"],
            "seq_id": selected_seq_id,
            "split": row.get("split", ""),
            "is_selected": row.get("is_selected", ""),
            "n_residues": row["n_residues"],
            "seq_sha256": seq_hash,
            "sequence_unique_id": row["unique_id"],
            "structure_id": record["structure_id"] if record else "",
            "structure_path": record["target_path"] if record else "",
            "structure_available": int(record is not None),
        }
        all_chain_map.append(mapping)
        pair_index[(row["pdb_id"].lower(), row["chain_id"])] = row
        folded_pair_index[(row["pdb_id"].lower(), row["chain_id"].lower())].append(row)
        if selected_seq_id:
            all_chain_index[selected_seq_id] = row
            folded_chain_index[selected_seq_id.lower()].append(row)

    final_rows = read_rows(args.final_chain_manifest, ",")
    final_map: list[dict[str, object]] = []
    final_unmatched: list[dict[str, object]] = []
    for row in final_rows:
        chain_record = all_chain_index.get(row["seq_id"])
        if chain_record is None:
            chain_record = pair_index.get((row["pdb_id"].lower(), row["chain_id"]))
        if chain_record is None:
            candidates = folded_chain_index.get(row["seq_id"].lower(), [])
            if len(candidates) == 1:
                chain_record = candidates[0]
        if chain_record is None:
            candidates = folded_pair_index.get((row["pdb_id"].lower(), row["chain_id"].lower()), [])
            if len(candidates) == 1:
                chain_record = candidates[0]
        if chain_record is not None:
            mapped_sequence = chain_record["sequence"].strip().upper()
            mapped_sha1 = hashlib.sha1(mapped_sequence.encode("ascii")).hexdigest()
            if mapped_sha1 != row["seq_sha1"].lower():
                raise ValueError(
                    f"Final chain sequence mismatch for {row['seq_id']}: "
                    f"manifest_sha1={row['seq_sha1']} mapped_sha1={mapped_sha1}"
                )
            if len(mapped_sequence) != int(row["n_residues"]):
                raise ValueError(
                    f"Final chain length mismatch for {row['seq_id']}: "
                    f"manifest={row['n_residues']} mapped={len(mapped_sequence)}"
                )
        seq_hash = chain_record["seq_sha256"].lower() if chain_record else ""
        record = canonical.get(seq_hash)
        mapping = {
            "split": row["split"],
            "seq_id": row["seq_id"],
            "pdb_id": row["pdb_id"],
            "chain_id": row["chain_id"],
            "n_residues": row["n_residues"],
            "component_id": row["component_id"],
            "exact_group_id": row["exact_group_id"],
            "seq_sha1": row["seq_sha1"],
            "seq_sha256": seq_hash,
            "sequence_unique_id": chain_record["unique_id"] if chain_record else "",
            "structure_id": record["structure_id"] if record else "",
            "structure_path": record["target_path"] if record else "",
            "structure_available": int(record is not None),
        }
        final_map.append(mapping)
        if chain_record is None or record is None:
            final_unmatched.append(mapping)

    structure_fields = list(structure_rows[0])
    inventory_fields = list(inventory[0])
    all_chain_fields = list(all_chain_map[0])
    final_fields = list(final_map[0])
    write_tsv(manifests_dir / "structure_manifest.tsv", structure_rows, structure_fields)
    write_tsv(manifests_dir / "source_inventory.tsv", inventory, inventory_fields)
    write_tsv(manifests_dir / "sequence_dedup_map.tsv", structure_rows, structure_fields)
    write_tsv(manifests_dir / "all_chain_to_structure.tsv", all_chain_map, all_chain_fields)
    write_tsv(manifests_dir / "final_chain_to_structure.tsv", final_map, final_fields)
    write_tsv(qc_dir / "structure_model_conflicts.tsv", conflicts, [
        "seq_sha256", "structure_id", "distinct_file_models", "file_sha256s", "source_paths"
    ])
    write_tsv(qc_dir / "final_chain_unmatched.tsv", final_unmatched, final_fields)

    final_unique = {str(row["seq_sha256"]) for row in final_map if row["seq_sha256"]}
    final_covered_unique = {
        str(row["seq_sha256"]) for row in final_map if int(row["structure_available"]) == 1
    }
    summary = {
        "target": str(target),
        "source_pdb_occurrences": len(source_paths),
        "source_unique_inodes": len(inode_cache),
        "manifest_sequence_ids": len(expected),
        "canonical_unique_structures": len(structure_rows),
        "canonical_distinct_sequences": len(canonical),
        "structure_model_conflicts": len(conflicts),
        "unmapped_source_pdbs": len(unmapped_sources),
        "sequence_mismatch_pdbs": len(sequence_mismatches),
        "all_chain_rows": len(all_chain_map),
        "all_chain_rows_with_structure": sum(int(row["structure_available"]) for row in all_chain_map),
        "final_chain_rows": len(final_map),
        "final_chain_rows_with_structure": sum(int(row["structure_available"]) for row in final_map),
        "final_unique_sequences": len(final_unique),
        "final_unique_sequences_with_structure": len(final_covered_unique),
        "final_unmatched_rows": len(final_unmatched),
        "storage_policy": "hardlink on same filesystem; copy fallback across filesystems",
        "sequence_qc": "PDB CA-derived sequence must exactly match manifest SHA-256",
    }
    (qc_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    readme = f"""# ESMFold structure archive

This directory is a non-destructive, sequence-keyed consolidation of the ESMFold
predictions used by the PPC project. Source files remain in their original benchmark
directories. Canonical PDB names are stable functions of exact sequence length and
SHA-256, not PDB IDs or UniProt bridges.

## Coverage

- Canonical unique structures: {summary['canonical_unique_structures']}
- Raw source PDB occurrences: {summary['source_pdb_occurrences']}
- Final split chains mapped: {summary['final_chain_rows_with_structure']} / {summary['final_chain_rows']}
- Final split unique sequences mapped: {summary['final_unique_sequences_with_structure']} / {summary['final_unique_sequences']}
- Sequence/PDB mismatches: {summary['sequence_mismatch_pdbs']}
- Distinct-model conflicts for one exact sequence: {summary['structure_model_conflicts']}

## Layout

- `structures/`: one canonical PDB per exact amino-acid sequence.
- `manifests/structure_manifest.tsv`: canonical structure metadata and pLDDT summary.
- `manifests/source_inventory.tsv`: every scanned source occurrence and its provenance.
- `manifests/sequence_dedup_map.tsv`: exact-sequence deduplication map.
- `manifests/all_chain_to_structure.tsv`: complete chain occurrence mapping from the feature universe.
- `manifests/final_chain_to_structure.tsv`: current chain-filtered SI30 split mapping.
- `qc/summary.json`: machine-readable coverage and QC summary.
- `qc/structure_model_conflicts.tsv`: same-sequence files with differing coordinate hashes.
- `qc/final_chain_unmatched.tsv`: final chains lacking a reversible structure mapping.

## Sequence and structure policy

The sequence key is SHA-256 of the complete experimental-design chain sequence used as
the folding input. For every PDB, the archive reconstructs a sequence from ordered CA
residues and requires an exact SHA-256 match to the input manifest. No PDBTM -> UniProt
-> AlphaFold/TmAlphaFold bridge is used.

The predictions came from several resumed/sharded ESMFold jobs. Their exact invocation
logs remain under the PPC benchmark run directories; therefore this archive does not
pretend that one global batching/chunk setting generated every file. Batching parameters
affect throughput and memory, while the biological input identity is captured here by
the exact sequence hash.

## Primary sources

- `{args.old_manifest}`
- `{args.old_pdb_root}`
- `{args.supplement_manifest}`
- `{args.supplement_pdb_root}`
- `{args.all_chains}`
- `{args.final_chain_manifest}`

Generated by `consolidate_esmfold_structures.py`. Re-running is idempotent when source
content is unchanged.
"""
    (target / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
