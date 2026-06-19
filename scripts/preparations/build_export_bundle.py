#!/usr/bin/env python3
"""Build a lightweight PPC metadata/result export bundle.

The bundle intentionally excludes large tensor/PDB files. It contains split IDs,
per-sample tags, chain-to-UniProt/predicted-structure bridge tags, experiment
configs, result CSVs/figures, and checksums for easy transfer and auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {row[key].lower(): row for row in reader if row.get(key)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_files(src: Path, dst: Path, suffixes: set[str] | None = None) -> int:
    if not src.exists():
        return 0
    n = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        n += 1
    return n


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def has_path(root: Path, rel: str | None) -> bool:
    return bool(rel) and (root / rel).exists()


def build(args: argparse.Namespace) -> Path:
    root = Path(args.project_root).resolve()
    out = Path(args.output).resolve()
    split_dir = root / args.split_dir
    label_manifest_path = root / "features/contact_labels/manifest.csv"
    predseq_manifest_path = root / "features/pred_struct_sequence_scalar_relaxed/manifest.csv"
    bridge_path = root / "features/pred_struct_bridge/bridge.csv"
    mmseq_chain_meta = root / "features/mmseq30/chain_metadata.csv"
    mmseq_cluster = root / "features/mmseq30/mmseq30_cluster.tsv"

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("splits", "tags", "manifests", "qc", "results", "configs", "checksums"):
        (out / sub).mkdir(exist_ok=True)

    split_ids: dict[str, list[str]] = {}
    split_of: dict[str, str] = {}
    for name in ("train", "val", "test"):
        ids = read_ids(split_dir / f"{name}_ids.txt")
        split_ids[name] = ids
        for pid in ids:
            if pid in split_of:
                raise ValueError(f"duplicate pdb_id across splits: {pid}")
            split_of[pid] = name
        copy_if_exists(split_dir / f"{name}_ids.txt", out / "splits" / f"{name}_ids.txt")
    copy_if_exists(split_dir / "summary.json", out / "splits" / "summary.json")

    label_rows = read_csv_by_key(label_manifest_path, "pdb_id")
    predseq_rows = read_csv_by_key(predseq_manifest_path, "pdb_id")
    bridge_rows = read_csv_rows(bridge_path)
    bridge_by_pdb: dict[str, list[dict[str, str]]] = {}
    for row in bridge_rows:
        pid = (row.get("pdb_id") or "").lower()
        if pid:
            bridge_by_pdb.setdefault(pid, []).append(row)

    sample_rows: list[dict[str, object]] = []
    for pid in sorted(split_of):
        label = label_rows.get(pid, {})
        pred = predseq_rows.get(pid, {})
        chains = bridge_by_pdb.get(pid, [])
        ok_chains = [r for r in chains if r.get("status") == "OK"]
        strict_chains = [r for r in chains if str(r.get("strict_pass", "")).lower() in {"1", "true", "yes"}]
        uniprots = sorted({r.get("uniprot_acc", "") for r in ok_chains if r.get("uniprot_acc")})
        uniprot_ids = sorted({r.get("uniprot_id", "") for r in ok_chains if r.get("uniprot_id")})
        chain_statuses = sorted({r.get("status", "") for r in chains if r.get("status")})
        mapping_methods = pred.get("mapping_methods", "")
        row = {
            "pdb_id": pid,
            "split": split_of[pid],
            "label_status": label.get("status", ""),
            "n_residues": label.get("n_residues", pred.get("n_residues", "")),
            "n_chains": label.get("n_chains", pred.get("n_chains", "")),
            "n_positive": label.get("n_positive", ""),
            "n_negative": label.get("n_negative", ""),
            "positive_fraction": label.get("positive_fraction", ""),
            "predseq_status": pred.get("status", "MISSING"),
            "predseq_n_features": pred.get("n_features", ""),
            "predseq_coverage_residue": pred.get("coverage_residue", ""),
            "predseq_available_chains": pred.get("n_available_chains", ""),
            "predseq_available_residues": pred.get("n_available_residues", ""),
            "predseq_mapping_methods": mapping_methods,
            "has_label_pt": has_path(root, label.get("label_path")),
            "has_esm_final_pt": (root / "features/esm2_t33_650M_UR50D/pt" / pid / f"{pid}_esm2.pt").exists(),
            "has_esm_mlc_pt": (root / "features/esm2_t33_650M_UR50D_mlc/pt" / pid / f"{pid}_esm2.pt").exists(),
            "has_predseq_pt": has_path(root, pred.get("output_path")),
            "bridge_n_chains": len(chains),
            "bridge_n_ok_chains": len(ok_chains),
            "bridge_n_strict_pass_chains": len(strict_chains),
            "uniprot_accs": ";".join(uniprots),
            "uniprot_ids": ";".join(uniprot_ids),
            "bridge_statuses": ";".join(chain_statuses),
            "usable_for_final_seq_only": True,
            "usable_for_mlc_contact": (root / "features/esm2_t33_650M_UR50D_mlc/pt" / pid / f"{pid}_esm2.pt").exists(),
            "usable_for_predstruct_scalar": pred.get("status") == "OK" and has_path(root, pred.get("output_path")),
        }
        sample_rows.append(row)

    sample_fields = [
        "pdb_id", "split", "label_status", "n_residues", "n_chains", "n_positive", "n_negative", "positive_fraction",
        "predseq_status", "predseq_n_features", "predseq_coverage_residue", "predseq_available_chains", "predseq_available_residues",
        "predseq_mapping_methods", "has_label_pt", "has_esm_final_pt", "has_esm_mlc_pt", "has_predseq_pt",
        "bridge_n_chains", "bridge_n_ok_chains", "bridge_n_strict_pass_chains", "uniprot_accs", "uniprot_ids", "bridge_statuses",
        "usable_for_final_seq_only", "usable_for_mlc_contact", "usable_for_predstruct_scalar",
    ]
    write_csv(out / "tags/sample_tags.csv", sample_rows, sample_fields)

    split_membership = [{"pdb_id": pid, "split": split_of[pid]} for pid in sorted(split_of)]
    write_csv(out / "splits/split_membership.csv", split_membership, ["pdb_id", "split"])

    chain_export: list[dict[str, object]] = []
    chain_fields = [
        "pdb_id", "split", "chain_id", "seq_id", "len_seq", "uniprot_acc", "uniprot_id", "uniprot_range",
        "coverage", "identity", "pdbe_coverage", "pdbe_identity", "alignment_cigar", "n_mismatch", "n_insert", "n_delete",
        "strict_pass", "status", "tmalphafold_entry_url", "alphafold_api_url", "alphafold_pdb_url",
    ]
    for row in bridge_rows:
        pid = (row.get("pdb_id") or "").lower()
        if pid not in split_of:
            continue
        chain_export.append({field: (split_of[pid] if field == "split" else row.get(field, "")) for field in chain_fields})
    write_csv(out / "tags/chain_tags.csv", chain_export, chain_fields)

    # Copy lightweight manifests and summaries.
    manifest_files = [
        "features/contact_labels/manifest.csv",
        "features/contact_labels/manifest.summary.json",
        "features/pred_struct_sequence_scalar_relaxed/manifest.csv",
        "features/pred_struct_sequence_scalar_relaxed/manifest.summary.json",
        "features/pred_struct_bridge/bridge.csv",
        "features/mmseq30/chain_metadata.csv",
        "features/mmseq30/mmseq30_cluster.tsv",
        "features/mmseq30/chain_fasta.summary.json",
        "features/tmalphafold_raw/manifest.csv",
        "features/tmalphafold_raw/manifest.summary.json",
    ]
    for rel in manifest_files:
        src = root / rel
        if src.exists():
            copy_if_exists(src, out / "manifests" / rel.replace("/", "__"))

    qc_files = [
        "features/pdbfixer/pdbfixer_report.csv",
        "features/pdbfixer/pdbfixer_report_sanitized_nohetconect.csv",
    ]
    for rel in qc_files:
        src = root / rel
        if src.exists():
            copy_if_exists(src, out / "qc" / Path(rel).name)
    copy_tree_files(root / "features/qc", out / "qc/features_qc", suffixes={".csv", ".json", ".txt", ".tsv"})

    # Copy result summaries and publication figures.
    for rel in [
        "figures/model_results_all_runs.csv",
        "figures/model_results_curated_relaxed.csv",
        "figures/model_results_curated_relaxed_summary.csv",
        "figures/TABLE_model_results_relaxed.tex",
        "figures/latex_includes.tex",
        "figures/fig_model_f1_horizontal_bar.pdf",
        "figures/fig_model_f1_horizontal_bar.png",
        "figures/fig_metrics_matrix_auc_topk.pdf",
        "figures/fig_metrics_matrix_auc_topk.png",
    ]:
        src = root / rel
        if src.exists():
            copy_if_exists(src, out / "results" / Path(rel).name)

    copy_tree_files(root / "configs/experiments", out / "configs/experiments", suffixes={".yaml", ".yml", ".md", ".tsv"})
    copy_tree_files(root / "runs/job_manifests", out / "manifests/job_manifests", suffixes={".tsv", ".csv", ".json", ".txt"})

    label_summary = {}
    label_summary_file = root / "features/contact_labels/manifest.summary.json"
    if label_summary_file.exists():
        try:
            label_summary = json.loads(label_summary_file.read_text())
        except json.JSONDecodeError:
            label_summary = {}
    label_definition = {
        "positive_contact_cutoff_angstrom": label_summary.get("cutoff", 5.0),
        "heavy_atom_only": label_summary.get("heavy_only", True),
        "source_manifest_summary": "features/contact_labels/manifest.summary.json",
    }

    counts = {
        "train": len(split_ids["train"]),
        "val": len(split_ids["val"]),
        "test": len(split_ids["test"]),
        "total": len(split_of),
        "sample_tags": len(sample_rows),
        "chain_tags": len(chain_export),
        "usable_for_final_seq_only": sum(1 for r in sample_rows if r["usable_for_final_seq_only"]),
        "usable_for_mlc_contact": sum(1 for r in sample_rows if r["usable_for_mlc_contact"]),
        "usable_for_predstruct_scalar": sum(1 for r in sample_rows if r["usable_for_predstruct_scalar"]),
    }
    version = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "git_commit": git_commit(),
        "split_dir": args.split_dir,
        "counts": counts,
        "label_definition": label_definition,
        "excluded_by_design": ["large tensor .pt files", "PDB/mmCIF structure files", "AlphaFold/TmAlphaFold cache JSON files"],
    }
    (out / "VERSION.json").write_text(json.dumps(version, indent=2, sort_keys=True) + "\n")

    readme = f"""# PPC metadata export bundle

Created UTC: `{version['created_utc']}`  
Git commit: `{version['git_commit']}`  
Split source: `{args.split_dir}`

This bundle is a lightweight transfer package for reproducibility, auditing, and downstream result export. It intentionally excludes large tensor/PDB files.

## Key files

- `splits/train_ids.txt`, `splits/val_ids.txt`, `splits/test_ids.txt`: locked complex-level split IDs.
- `splits/split_membership.csv`: one row per PDB complex with split assignment.
- `tags/sample_tags.csv`: one row per PDB complex with labels, feature availability, predicted-structure coverage, and usability flags.
- `tags/chain_tags.csv`: one row per PDB chain with UniProt/TmAlphaFold/AlphaFold mapping and alignment quality.
- `manifests/`: source manifests copied from `features/` plus submitted job manifests.
- `configs/experiments/`: archived YAML configs grouped by experiment family.
- `results/`: summary CSVs, LaTeX snippets, and model comparison figures.
- `checksums/sha256_manifest.tsv`: checksums for all files in this bundle.

## Label Definition

Positive residue labels are generated from complex structures using heavy-atom contacts with cutoff `{label_definition['positive_contact_cutoff_angstrom']}` Angstrom. `heavy_atom_only={label_definition['heavy_atom_only']}`.

## Counts

```json
{json.dumps(counts, indent=2, sort_keys=True)}
```

## Transfer notes

Use `sample_tags.csv` to decide which samples are valid for a model family before copying large feature tensors. For example, `usable_for_predstruct_scalar=True` marks complexes that have relaxed predicted-structure scalar features aligned to the split.
"""
    (out / "README.md").write_text(readme)

    checksum_rows = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and "checksums" not in p.parts):
        checksum_rows.append({"sha256": sha256_file(path), "path": str(path.relative_to(out)), "bytes": path.stat().st_size})
    write_csv(out / "checksums/sha256_manifest.tsv", checksum_rows, ["sha256", "path", "bytes"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--split-dir", default="features/contact_labels/splits_mmseq30_tmk_no_len_limit_predstruct_relaxed")
    parser.add_argument("--output", default="exports/ppc_metadata_release_latest")
    args = parser.parse_args()
    out = build(args)
    print(out)


if __name__ == "__main__":
    main()
