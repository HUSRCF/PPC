#!/usr/bin/env python3
"""Build complex-level cluster-disjoint train/val/test splits from MMseqs clusters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


class DSU:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def _read_manifest(path: Path, include_zero_positive: bool) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = row["pdb_id"].lower()
            if row.get("status") not in {"OK", "SKIP"}:
                continue
            if not include_zero_positive and int(row.get("n_positive") or 0) <= 0:
                continue
            rows[pdb_id] = row
    return rows


def _read_chain_metadata(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    seq_rows: dict[str, dict[str, Any]] = {}
    pdb_to_seq: dict[str, list[str]] = defaultdict(list)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = row["seq_id"]
            pdb_id = row["pdb_id"].lower()
            seq_rows[seq_id] = row
            pdb_to_seq[pdb_id].append(seq_id)
    return seq_rows, pdb_to_seq


def _read_mmseq_clusters(path: Path, known_seq_ids: set[str]) -> tuple[dict[str, str], dict[str, list[str]], int]:
    seq_to_cluster: dict[str, str] = {}
    cluster_to_seq: dict[str, list[str]] = defaultdict(list)
    n_unknown_rows = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            if member not in known_seq_ids:
                n_unknown_rows += 1
                continue
            seq_to_cluster[member] = rep
            cluster_to_seq[rep].append(member)
    for seq_id in known_seq_ids:
        if seq_id not in seq_to_cluster:
            seq_to_cluster[seq_id] = seq_id
            cluster_to_seq[seq_id].append(seq_id)
    return seq_to_cluster, cluster_to_seq, n_unknown_rows


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _write_ids(path: Path, ids: list[str]) -> None:
    _write_text_atomic(path, "\n".join(ids) + ("\n" if ids else ""))


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def _row_int(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def _component_metrics(pdb_ids: list[str], clusters: set[str], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    residues = sum(_row_int(manifest[pdb_id], "n_residues") for pdb_id in pdb_ids)
    positives = sum(_row_int(manifest[pdb_id], "n_positive") for pdb_id in pdb_ids)
    negatives = sum(_row_int(manifest[pdb_id], "n_negative") for pdb_id in pdb_ids)
    return {
        "n_complexes": len(pdb_ids),
        "n_clusters": len(clusters),
        "n_residues": residues,
        "n_positive": positives,
        "n_negative": negatives,
        "positive_fraction": float(positives / max(1, positives + negatives)),
    }


def _split_score(current: dict[str, int], target: dict[str, float], split: str, weight: int) -> tuple[float, int]:
    after = dict(current)
    after[split] += weight
    score = 0.0
    for name, value in after.items():
        denom = max(1.0, target[name])
        score += abs(value - target[name]) / denom
    return score, after[split]


def _assign_components(
    components: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
    weight_key: str,
) -> None:
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    total_weight = sum(int(component[weight_key]) for component in components)
    target = {
        "train": total_weight * train_frac,
        "val": total_weight * val_frac,
        "test": total_weight * test_frac,
    }
    current = {"train": 0, "val": 0, "test": 0}
    rng = random.Random(seed)
    ordered = sorted(
        components,
        key=lambda component: (-int(component[weight_key]), rng.random()),
    )
    split_order = ["train", "val", "test"]
    for component in ordered:
        weight = int(component[weight_key])
        best_split = min(
            split_order,
            key=lambda split: (_split_score(current, target, split, weight), split_order.index(split)),
        )
        component["split"] = best_split
        current[best_split] += weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--chain-metadata", required=True, type=Path)
    parser.add_argument("--cluster-tsv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-by", choices=["samples", "residues"], default="samples")
    parser.add_argument("--include-zero-positive", action="store_true")
    args = parser.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0 or args.train_frac + args.val_frac >= 1.0:
        raise ValueError("--train-frac and --val-frac must leave a positive test fraction")

    manifest = _read_manifest(args.manifest, include_zero_positive=args.include_zero_positive)
    seq_rows, pdb_to_seq = _read_chain_metadata(args.chain_metadata)
    seq_to_cluster, cluster_to_seq, n_unknown_cluster_rows = _read_mmseq_clusters(args.cluster_tsv, set(seq_rows))

    eligible_ids = sorted(pdb_id for pdb_id in manifest if pdb_id in pdb_to_seq)
    missing_chain_ids = sorted(pdb_id for pdb_id in manifest if pdb_id not in pdb_to_seq)
    dsu = DSU(eligible_ids)

    cluster_to_pdbs: dict[str, set[str]] = defaultdict(set)
    pdb_to_clusters: dict[str, set[str]] = {}
    for pdb_id in eligible_ids:
        clusters = {seq_to_cluster[seq_id] for seq_id in pdb_to_seq[pdb_id]}
        pdb_to_clusters[pdb_id] = clusters
        for cluster_id in clusters:
            cluster_to_pdbs[cluster_id].add(pdb_id)

    for pdbs in cluster_to_pdbs.values():
        ordered = sorted(pdbs)
        if len(ordered) <= 1:
            continue
        first = ordered[0]
        for pdb_id in ordered[1:]:
            dsu.union(first, pdb_id)

    root_to_pdbs: dict[str, list[str]] = defaultdict(list)
    for pdb_id in eligible_ids:
        root_to_pdbs[dsu.find(pdb_id)].append(pdb_id)

    components: list[dict[str, Any]] = []
    for component_index, pdb_ids in enumerate(sorted(root_to_pdbs.values(), key=lambda ids: (len(ids), ids[0]), reverse=True)):
        pdb_ids = sorted(pdb_ids)
        clusters: set[str] = set()
        for pdb_id in pdb_ids:
            clusters.update(pdb_to_clusters[pdb_id])
        metrics = _component_metrics(pdb_ids, clusters, manifest)
        weight_key = "n_complexes" if args.weight_by == "samples" else "n_residues"
        component = {
            "component_id": component_index,
            "pdb_ids": pdb_ids,
            "clusters": sorted(clusters),
            "weight": metrics[weight_key],
            **metrics,
        }
        components.append(component)

    weight_key = "n_complexes" if args.weight_by == "samples" else "n_residues"
    _assign_components(components, args.train_frac, args.val_frac, args.seed, weight_key)

    split_to_ids = {"train": [], "val": [], "test": []}
    split_to_clusters = {"train": set(), "val": set(), "test": set()}
    for component in components:
        split = component["split"]
        split_to_ids[split].extend(component["pdb_ids"])
        split_to_clusters[split].update(component["clusters"])
    for split in split_to_ids:
        split_to_ids[split] = sorted(split_to_ids[split])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_ids(args.output_dir / "train_ids.txt", split_to_ids["train"])
    _write_ids(args.output_dir / "val_ids.txt", split_to_ids["val"])
    _write_ids(args.output_dir / "test_ids.txt", split_to_ids["test"])

    component_csv = args.output_dir / "components.csv"
    tmp_component = component_csv.parent / f".{component_csv.name}.{uuid.uuid4().hex}.tmp"
    with tmp_component.open("w", newline="") as handle:
        fieldnames = [
            "component_id",
            "split",
            "n_complexes",
            "n_clusters",
            "n_residues",
            "n_positive",
            "n_negative",
            "positive_fraction",
            "pdb_ids",
            "clusters",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for component in sorted(components, key=lambda item: item["component_id"]):
            writer.writerow(
                {
                    key: component[key]
                    for key in fieldnames
                    if key not in {"pdb_ids", "clusters"}
                }
                | {
                    "pdb_ids": ";".join(component["pdb_ids"]),
                    "clusters": ";".join(component["clusters"]),
                }
            )
    os.replace(tmp_component, component_csv)

    complex_csv = args.output_dir / "complex_clusters.csv"
    tmp_complex = complex_csv.parent / f".{complex_csv.name}.{uuid.uuid4().hex}.tmp"
    pdb_to_split = {pdb_id: split for split, ids in split_to_ids.items() for pdb_id in ids}
    root_to_component = {
        pdb_id: component["component_id"]
        for component in components
        for pdb_id in component["pdb_ids"]
    }
    with tmp_complex.open("w", newline="") as handle:
        fieldnames = ["pdb_id", "split", "component_id", "n_chains", "n_clusters", "clusters"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pdb_id in eligible_ids:
            clusters = sorted(pdb_to_clusters[pdb_id])
            writer.writerow(
                {
                    "pdb_id": pdb_id,
                    "split": pdb_to_split[pdb_id],
                    "component_id": root_to_component[pdb_id],
                    "n_chains": len(pdb_to_seq[pdb_id]),
                    "n_clusters": len(clusters),
                    "clusters": ";".join(clusters),
                }
            )
    os.replace(tmp_complex, complex_csv)

    intersections: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        key = f"{left}_{right}"
        intersections[key] = sorted(split_to_clusters[left] & split_to_clusters[right])

    split_stats: dict[str, Any] = {}
    for split, ids in split_to_ids.items():
        positives = sum(_row_int(manifest[pdb_id], "n_positive") for pdb_id in ids)
        negatives = sum(_row_int(manifest[pdb_id], "n_negative") for pdb_id in ids)
        residues = sum(_row_int(manifest[pdb_id], "n_residues") for pdb_id in ids)
        split_components = [component for component in components if component["split"] == split]
        split_stats[split] = {
            "n_complexes": len(ids),
            "n_components": len(split_components),
            "n_chain_clusters": len(split_to_clusters[split]),
            "n_residues": residues,
            "n_positive": positives,
            "n_negative": negatives,
            "positive_fraction": float(positives / max(1, positives + negatives)),
            "component_size": _stats([int(component["n_complexes"]) for component in split_components]),
        }

    summary = {
        "manifest": str(args.manifest),
        "chain_metadata": str(args.chain_metadata),
        "cluster_tsv": str(args.cluster_tsv),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "weight_by": args.weight_by,
        "include_zero_positive": args.include_zero_positive,
        "n_manifest_eligible": len(manifest),
        "n_eligible_with_chains": len(eligible_ids),
        "n_missing_chain_metadata": len(missing_chain_ids),
        "missing_chain_metadata_examples": missing_chain_ids[:20],
        "n_chain_records": len(seq_rows),
        "n_mmseq_clusters": len(cluster_to_seq),
        "n_unknown_cluster_rows": n_unknown_cluster_rows,
        "n_components": len(components),
        "component_size": _stats([int(component["n_complexes"]) for component in components]),
        "largest_components": [
            {
                "component_id": component["component_id"],
                "split": component["split"],
                "n_complexes": component["n_complexes"],
                "n_clusters": component["n_clusters"],
                "n_residues": component["n_residues"],
                "n_positive": component["n_positive"],
                "examples": component["pdb_ids"][:20],
            }
            for component in sorted(components, key=lambda item: int(item["n_complexes"]), reverse=True)[:10]
        ],
        "split_stats": split_stats,
        "cluster_intersections": {
            key: {"n": len(value), "examples": value[:20]}
            for key, value in intersections.items()
        },
    }
    _write_text_atomic(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    leaked = sum(len(value) for value in intersections.values())
    return 0 if leaked == 0 and not missing_chain_ids else 2 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
