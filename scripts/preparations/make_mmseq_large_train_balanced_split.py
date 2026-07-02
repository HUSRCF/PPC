#!/usr/bin/env python3
"""Build a large-train, balanced-val/test split from an MMseq component split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def read_ids(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(text)
    os.replace(tmp, path)


def write_ids(path: Path, ids: list[str]) -> None:
    write_text_atomic(path, "\n".join(sorted(ids)) + ("\n" if ids else ""))


def read_label_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = (row.get("pdb_id") or "").lower()
            if pdb_id:
                rows[pdb_id] = row
    return rows


def read_pred_coverage(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = (row.get("pdb_id") or "").lower()
            if not pdb_id:
                continue
            try:
                out[pdb_id] = float(row.get("coverage_residue") or 0.0)
            except ValueError:
                out[pdb_id] = 0.0
    return out


def row_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key) or 0))
    except ValueError:
        return 0


def load_complex_clusters(path: Path, allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = (row.get("pdb_id") or "").lower()
            if pdb_id in allowed_ids:
                rows[pdb_id] = row
    missing = sorted(allowed_ids - set(rows))
    if missing:
        examples = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} IDs are missing from complex_clusters.csv, examples: {examples}")
    return rows


def component_stats(
    complex_rows: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    coverage: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for pdb_id, row in complex_rows.items():
        grouped[str(row["component_id"])].append(pdb_id)

    components: list[dict[str, Any]] = []
    for component_id, pdb_ids in grouped.items():
        pdb_ids = sorted(pdb_ids)
        clusters: set[str] = set()
        residues = positives = negatives = 0
        cov_values: list[float] = []
        for pdb_id in pdb_ids:
            row = complex_rows[pdb_id]
            clusters.update(cluster for cluster in row.get("clusters", "").split(";") if cluster)
            label = labels.get(pdb_id, {})
            residues += row_int(label, "n_residues")
            positives += row_int(label, "n_positive")
            negatives += row_int(label, "n_negative")
            if pdb_id in coverage:
                cov_values.append(float(coverage[pdb_id]))
        components.append(
            {
                "component_id": component_id,
                "pdb_ids": pdb_ids,
                "clusters": sorted(clusters),
                "n_complexes": len(pdb_ids),
                "n_clusters": len(clusters),
                "n_residues": residues,
                "n_positive": positives,
                "n_negative": negatives,
                "positive_fraction": positives / max(1, positives + negatives),
                "coverage_residue_mean": sum(cov_values) / max(1, len(cov_values)),
            }
        )
    return components


def split_totals(assigned: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for split, components in assigned.items():
        n_complexes = sum(int(c["n_complexes"]) for c in components)
        n_residues = sum(int(c["n_residues"]) for c in components)
        n_positive = sum(int(c["n_positive"]) for c in components)
        n_negative = sum(int(c["n_negative"]) for c in components)
        cov_weight = sum(float(c["coverage_residue_mean"]) * int(c["n_complexes"]) for c in components)
        totals[split] = {
            "n_components": float(len(components)),
            "n_complexes": float(n_complexes),
            "n_residues": float(n_residues),
            "n_positive": float(n_positive),
            "n_negative": float(n_negative),
            "positive_fraction": n_positive / max(1, n_positive + n_negative),
            "coverage_residue_mean": cov_weight / max(1, n_complexes),
        }
    return totals


def objective(
    assigned: dict[str, list[dict[str, Any]]],
    target: dict[str, dict[str, float]],
    global_positive_fraction: float,
    global_coverage_mean: float,
    weights: dict[str, float],
) -> float:
    totals = split_totals(assigned)
    score = 0.0
    for split in SPLITS:
        current = totals[split]
        want = target[split]
        score += weights["residue"] * abs(current["n_residues"] - want["n_residues"]) / max(1.0, want["n_residues"])
        score += weights["complex"] * abs(current["n_complexes"] - want["n_complexes"]) / max(1.0, want["n_complexes"])
        if current["n_complexes"] > 0:
            score += weights["positive"] * abs(current["positive_fraction"] - global_positive_fraction)
            score += weights["coverage"] * abs(current["coverage_residue_mean"] - global_coverage_mean)
        else:
            score += weights["empty_split"]
    return score


def greedy_assign(
    components: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    large_min_complexes: int,
    seed: int,
    weights: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    test_frac = 1.0 - train_frac - val_frac
    if train_frac <= 0 or val_frac <= 0 or test_frac <= 0:
        raise ValueError("train/val/test fractions must all be positive")

    total_complexes = sum(int(c["n_complexes"]) for c in components)
    total_residues = sum(int(c["n_residues"]) for c in components)
    total_positive = sum(int(c["n_positive"]) for c in components)
    total_negative = sum(int(c["n_negative"]) for c in components)
    total_cov = sum(float(c["coverage_residue_mean"]) * int(c["n_complexes"]) for c in components)
    fractions = {"train": train_frac, "val": val_frac, "test": test_frac}
    target = {
        split: {
            "n_complexes": total_complexes * frac,
            "n_residues": total_residues * frac,
        }
        for split, frac in fractions.items()
    }
    global_positive_fraction = total_positive / max(1, total_positive + total_negative)
    global_coverage_mean = total_cov / max(1, total_complexes)

    rng = random.Random(seed)
    ordered = sorted(
        components,
        key=lambda c: (-int(c["n_complexes"]), -int(c["n_residues"]), rng.random()),
    )
    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    remaining: list[dict[str, Any]] = []
    for component in ordered:
        if int(component["n_complexes"]) >= large_min_complexes:
            component["assignment_reason"] = f"large_component_ge_{large_min_complexes}"
            assigned["train"].append(component)
        else:
            remaining.append(component)

    for component in remaining:
        best_split = None
        best_score = math.inf
        for split in SPLITS:
            trial = {name: list(value) for name, value in assigned.items()}
            trial[split].append(component)
            score = objective(trial, target, global_positive_fraction, global_coverage_mean, weights)
            if score < best_score:
                best_split = split
                best_score = score
        assert best_split is not None
        component["assignment_reason"] = "greedy_balance"
        assigned[best_split].append(component)

    return assigned


def write_component_csv(path: Path, assigned: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = [
        "component_id",
        "split",
        "assignment_reason",
        "n_complexes",
        "n_clusters",
        "n_residues",
        "n_positive",
        "n_negative",
        "positive_fraction",
        "coverage_residue_mean",
        "pdb_ids",
        "clusters",
    ]
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in SPLITS:
            for component in sorted(assigned[split], key=lambda c: int(c["component_id"])):
                writer.writerow(
                    {
                        "component_id": component["component_id"],
                        "split": split,
                        "assignment_reason": component["assignment_reason"],
                        "n_complexes": component["n_complexes"],
                        "n_clusters": component["n_clusters"],
                        "n_residues": component["n_residues"],
                        "n_positive": component["n_positive"],
                        "n_negative": component["n_negative"],
                        "positive_fraction": component["positive_fraction"],
                        "coverage_residue_mean": component["coverage_residue_mean"],
                        "pdb_ids": ";".join(component["pdb_ids"]),
                        "clusters": ";".join(component["clusters"]),
                    }
                )
    os.replace(tmp, path)


def write_complex_csv(path: Path, assigned: dict[str, list[dict[str, Any]]], complex_rows: dict[str, dict[str, Any]]) -> None:
    component_to_split = {
        str(component["component_id"]): split
        for split, components in assigned.items()
        for component in components
    }
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp.open("w", newline="") as handle:
        fieldnames = ["pdb_id", "split", "component_id", "n_chains", "n_clusters", "clusters"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pdb_id in sorted(complex_rows):
            row = complex_rows[pdb_id]
            component_id = str(row["component_id"])
            writer.writerow(
                {
                    "pdb_id": pdb_id,
                    "split": component_to_split[component_id],
                    "component_id": component_id,
                    "n_chains": row.get("n_chains", ""),
                    "n_clusters": row.get("n_clusters", ""),
                    "clusters": row.get("clusters", ""),
                }
            )
    os.replace(tmp, path)


def build_summary(args: argparse.Namespace, assigned: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = split_totals(assigned)
    split_ids = {
        split: sorted(pdb_id for component in components for pdb_id in component["pdb_ids"])
        for split, components in assigned.items()
    }
    split_clusters = {
        split: set(cluster for component in components for cluster in component["clusters"])
        for split, components in assigned.items()
    }
    split_components = {
        split: {str(component["component_id"]) for component in components}
        for split, components in assigned.items()
    }

    intersections: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        intersections[f"{left}_{right}"] = {
            "cluster_n": len(split_clusters[left] & split_clusters[right]),
            "cluster_examples": sorted(split_clusters[left] & split_clusters[right])[:20],
            "component_n": len(split_components[left] & split_components[right]),
            "component_examples": sorted(split_components[left] & split_components[right])[:20],
        }

    return {
        "strategy": "large_train_balanced_valtest",
        "base_split_dir": str(args.base_split_dir),
        "source_split_dir": str(args.source_split_dir),
        "label_manifest": str(args.label_manifest),
        "predstruct_manifest": str(args.predstruct_manifest) if args.predstruct_manifest else "",
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": 1.0 - args.train_frac - args.val_frac,
        },
        "large_min_complexes": args.large_min_complexes,
        "weights": {
            "residue": args.weight_residue,
            "complex": args.weight_complex,
            "positive": args.weight_positive,
            "coverage": args.weight_coverage,
            "empty_split": args.weight_empty_split,
        },
        "n_ids": sum(len(ids) for ids in split_ids.values()),
        "n_components": sum(len(v) for v in split_components.values()),
        "split_stats": {
            split: {
                **totals[split],
                "n_chain_clusters": len(split_clusters[split]),
                "n_ids": len(split_ids[split]),
            }
            for split in SPLITS
        },
        "intersections": intersections,
        "largest_components": [
            {
                "component_id": component["component_id"],
                "split": split,
                "n_complexes": component["n_complexes"],
                "n_residues": component["n_residues"],
                "positive_fraction": component["positive_fraction"],
                "coverage_residue_mean": component["coverage_residue_mean"],
                "assignment_reason": component["assignment_reason"],
                "examples": component["pdb_ids"][:20],
            }
            for split, components in assigned.items()
            for component in sorted(components, key=lambda c: int(c["n_complexes"]), reverse=True)[:10]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split-dir", required=True, type=Path)
    parser.add_argument("--source-split-dir", required=True, type=Path)
    parser.add_argument("--label-manifest", required=True, type=Path)
    parser.add_argument("--predstruct-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--large-min-complexes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-residue", type=float, default=1.0)
    parser.add_argument("--weight-complex", type=float, default=0.7)
    parser.add_argument("--weight-positive", type=float, default=2.0)
    parser.add_argument("--weight-coverage", type=float, default=0.5)
    parser.add_argument("--weight-empty-split", type=float, default=100.0)
    args = parser.parse_args()

    all_ids_path = args.source_split_dir / "all_ids.txt"
    if all_ids_path.exists():
        source_ids = read_ids(all_ids_path)
    else:
        source_ids = []
        for split in SPLITS:
            source_ids.extend(read_ids(args.source_split_dir / f"{split}_ids.txt"))
    allowed_ids = set(source_ids)
    if len(allowed_ids) != len(source_ids):
        raise ValueError("source split contains duplicated IDs")

    labels = read_label_manifest(args.label_manifest)
    missing_labels = sorted(allowed_ids - set(labels))
    if missing_labels:
        examples = ", ".join(missing_labels[:10])
        raise ValueError(f"{len(missing_labels)} source IDs are missing labels, examples: {examples}")

    coverage = read_pred_coverage(args.predstruct_manifest)
    complex_rows = load_complex_clusters(args.base_split_dir / "complex_clusters.csv", allowed_ids)
    components = component_stats(complex_rows, labels, coverage)
    weights = {
        "residue": args.weight_residue,
        "complex": args.weight_complex,
        "positive": args.weight_positive,
        "coverage": args.weight_coverage,
        "empty_split": args.weight_empty_split,
    }
    assigned = greedy_assign(components, args.train_frac, args.val_frac, args.large_min_complexes, args.seed, weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {
        split: sorted(pdb_id for component in assigned[split] for pdb_id in component["pdb_ids"])
        for split in SPLITS
    }
    for split, ids in split_ids.items():
        write_ids(args.output_dir / f"{split}_ids.txt", ids)
    write_ids(args.output_dir / "all_ids.txt", [pdb_id for ids in split_ids.values() for pdb_id in ids])
    write_component_csv(args.output_dir / "components.csv", assigned)
    write_complex_csv(args.output_dir / "complex_clusters.csv", assigned, complex_rows)

    summary = build_summary(args, assigned)
    write_text_atomic(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    leaked = any(value["cluster_n"] or value["component_n"] for value in summary["intersections"].values())
    return 2 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
