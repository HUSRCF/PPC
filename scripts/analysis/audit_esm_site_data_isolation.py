#!/usr/bin/env python3
"""Audit split, cache-path, and training-loop isolation for ESM site runs."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml


SPLITS = ("train", "val", "test")


def _project_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row_key = str(row.get(key, "")).strip()
            if not row_key:
                raise ValueError(f"{path}: row has empty {key!r}")
            if row_key in rows:
                raise ValueError(f"{path}: duplicate {key}={row_key!r}")
            rows[row_key] = row
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairwise_intersections(values: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(values[left] & values[right])
            output[f"{left}-{right}"] = {"count": len(overlap), "examples": overlap[:20]}
    return output


def _first_existing(candidates: tuple[Path, ...]) -> Path | None:
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _source_esm(root: Path, sample_id: str, row: dict[str, str]) -> Path | None:
    del sample_id
    pdb_id = row["pdb_id"].lower()
    return _first_existing(
        (
            root / pdb_id / f"{pdb_id}_esm2.pt",
            root / pdb_id / f"{pdb_id}_protein.pt",
            root / f"{pdb_id}_esm2.pt",
            root / f"{pdb_id}_protein.pt",
        )
    )


def _label(root: Path, sample_id: str, row: dict[str, str]) -> Path | None:
    del sample_id
    pdb_id = row["pdb_id"].lower()
    return _first_existing(
        (
            root / pdb_id / f"{pdb_id}_labels.pt",
            root / pdb_id / f"{pdb_id}_contact.pt",
            root / f"{pdb_id}_labels.pt",
            root / f"{pdb_id}.pt",
        )
    )


def _sequence_feature(root: Path, sample_id: str, row: dict[str, str]) -> Path | None:
    del row
    return _first_existing(
        (
            root / sample_id / f"{sample_id}_seq.pt",
            root / sample_id / f"{sample_id}_sequence.pt",
            root / f"{sample_id}_seq.pt",
            root / f"{sample_id}.pt",
        )
    )


def _embedding(root: Path, sample_id: str, row: dict[str, str]) -> Path | None:
    del row
    names = (
        f"{sample_id}_prottrans.pt",
        f"{sample_id}_protbert.pt",
        f"{sample_id}_protbert_bfd.pt",
        f"{sample_id}_embeddings.pt",
        f"{sample_id}.pt",
    )
    return _first_existing(tuple(root / sample_id / name for name in names) + tuple(root / name for name in names))


def _contact(root: Path, sample_id: str, row: dict[str, str]) -> Path | None:
    del row
    names = (
        f"{sample_id}_contact_graph.pt",
        f"{sample_id}_predcontact.pt",
        f"{sample_id}_pred_struct_contact.pt",
        f"{sample_id}.pt",
    )
    return _first_existing(tuple(root / sample_id / name for name in names) + tuple(root / name for name in names))


def _resolved_path_audit(
    root: Path | None,
    resolver: Callable[[Path, str, dict[str, str]], Path | None],
    ids: dict[str, list[str]],
    rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if root is None:
        return {"configured": False, "root": None}
    paths: dict[str, set[str]] = {split: set() for split in SPLITS}
    missing: dict[str, list[str]] = {split: [] for split in SPLITS}
    symlinks = 0
    for split in SPLITS:
        for sample_id in ids[split]:
            path = resolver(root, sample_id, rows[sample_id])
            if path is None:
                missing[split].append(sample_id)
                continue
            symlinks += int(path.is_symlink())
            paths[split].add(str(path.resolve()))
    return {
        "configured": True,
        "root": str(root),
        "resolved_unique_paths": {split: len(paths[split]) for split in SPLITS},
        "resolved_symlink_references": symlinks,
        "missing": {split: {"count": len(items), "examples": items[:20]} for split, items in missing.items()},
        "cross_split_resolved_path_overlap": _pairwise_intersections(paths),
    }


def _function_facts(training_script: Path, function_name: str) -> dict[str, Any]:
    source = training_script.read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name
    )
    segment = ast.get_source_segment(source, node) or ""
    calls = [
        ".".join(part for part in (getattr(call.func.value, "id", ""), call.func.attr) if part)
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]
    return {
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "calls_model_eval": "model.eval" in calls,
        "uses_torch_no_grad": "torch.no_grad" in calls,
        "contains_backward_call": any(call.endswith(".backward") or call == "backward" for call in calls),
        "contains_optimizer_or_scaler_step": any(call in {"optimizer.step", "scaler.step"} for call in calls),
        "checks_logits_requires_grad": "logits.requires_grad" in segment,
        "checks_parameter_versions": "parameter._version" in segment,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ESM Site Data and Gradient Isolation Audit",
        "",
        f"- Overall status: **{report['status'].upper()}**",
        f"- Config: `{report['inputs']['config']}`",
        f"- Training script: `{report['inputs']['training_script']}`",
        f"- Loaded chain counts: `{report['split_counts']}`",
        f"- Test loaded by training config: `{report['training_protocol']['test_split_loaded']}`",
        "",
        "## Split Isolation",
        "",
    ]
    for key, item in report["identity_audits"].items():
        counts = ", ".join(f"{pair}={value['count']}" for pair, value in item["cross_split"].items())
        lines.append(f"- `{key}` cross-split overlap: {counts}")
    lines.extend(["", "## Resolved File Isolation", ""])
    for key, item in report["resolved_path_audits"].items():
        if not item["configured"]:
            continue
        missing = sum(value["count"] for value in item["missing"].values())
        overlaps = sum(value["count"] for value in item["cross_split_resolved_path_overlap"].values())
        lines.append(f"- `{key}`: missing={missing}; cross-split resolved-path overlaps={overlaps}")
    facts = report["training_protocol"]["evaluate_function"]
    lines.extend(
        [
            "",
            "## Gradient Isolation",
            "",
            f"- `_evaluate` calls `model.eval()`: `{facts['calls_model_eval']}`",
            f"- `_evaluate` runs under `torch.no_grad()`: `{facts['uses_torch_no_grad']}`",
            f"- `_evaluate` contains backward: `{facts['contains_backward_call']}`",
            f"- `_evaluate` contains optimizer/scaler step: `{facts['contains_optimizer_or_scaler_step']}`",
            f"- Runtime logits/parameter-version assertions present: `{facts['checks_logits_requires_grad'] and facts['checks_parameter_versions']}`",
            "",
            "## SI30 Evidence",
            "",
        ]
    )
    si = report.get("si30_evidence")
    if si:
        lines.extend(
            [
                f"- Unique exact sequences: `{si.get('n_unique_sequences')}`",
                f"- Recalled cross-split pairs: `{si.get('n_cross_split_candidate_pairs')}`",
                f"- SI >= 0.30 violations: `{si.get('n_si30_violations')}`",
                f"- Maximum recalled cross-split global SI: `{si.get('max_cross_split_global_identity')}`",
                f"- Scope limitation: {si.get('candidate_recall_limitation')}",
            ]
        )
    else:
        lines.append("- No SI30 summary was supplied to this run.")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, default=Path("scripts/training/train_esm_site.py"))
    parser.add_argument("--si-summary", type=Path)
    parser.add_argument("--si-violations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = _project_path(project_root, args.config)
    training_script = _project_path(project_root, args.training_script)
    output_dir = _project_path(project_root, args.output_dir)
    assert config_path is not None and training_script is not None and output_dir is not None
    config = yaml.safe_load(config_path.read_text())
    data = config.get("data", {})
    training = config.get("training", {})

    split_dir = _project_path(project_root, data["split_dir"])
    chain_manifest_path = _project_path(project_root, data["chain_filter_manifest"])
    training_manifest_path = _project_path(project_root, data["manifest"])
    assert split_dir is not None and chain_manifest_path is not None and training_manifest_path is not None
    split_paths = {
        split: next(
            path
            for path in (split_dir / f"{split}_chain_ids.txt", split_dir / f"{split}_ids.txt")
            if path.exists()
        )
        for split in SPLITS
    }
    ids = {split: _read_ids(path) for split, path in split_paths.items()}
    chain_rows = _read_rows(chain_manifest_path, "seq_id")
    training_rows = _read_rows(training_manifest_path, "pdb_id")

    failures: list[str] = []
    duplicate_ids = {
        split: sorted(key for key, count in Counter(values).items() if count > 1)
        for split, values in ids.items()
    }
    for split, duplicates in duplicate_ids.items():
        if duplicates:
            failures.append(f"{split} split contains {len(duplicates)} duplicate IDs")
    id_sets = {split: set(values) for split, values in ids.items()}
    id_overlap = _pairwise_intersections(id_sets)
    if any(item["count"] for item in id_overlap.values()):
        failures.append("chain IDs overlap across splits")
    union_ids = set().union(*id_sets.values())
    missing_manifest = sorted(union_ids - set(chain_rows))
    extra_manifest = sorted(set(chain_rows) - union_ids)
    if missing_manifest or extra_manifest:
        failures.append(
            f"chain manifest differs from split union: missing={len(missing_manifest)} extra={len(extra_manifest)}"
        )
    missing_training_manifest = sorted(union_ids - set(training_rows))
    extra_training_manifest = sorted(set(training_rows) - union_ids)
    if missing_training_manifest or extra_training_manifest:
        failures.append(
            "training manifest differs from split union: "
            f"missing={len(missing_training_manifest)} extra={len(extra_training_manifest)}"
        )

    split_mismatches: list[str] = []
    for split in SPLITS:
        for sample_id in ids[split]:
            if sample_id not in chain_rows:
                continue
            if chain_rows[sample_id].get("split") != split:
                split_mismatches.append(sample_id)
            if sample_id in training_rows and training_rows[sample_id].get("split") != split:
                split_mismatches.append(sample_id)
    if split_mismatches:
        failures.append(f"manifest split field disagrees for {len(set(split_mismatches))} chains")

    identity_audits: dict[str, Any] = {
        "seq_id": {"cross_split": id_overlap},
    }
    for key in ("pdb_id", "component_id", "seq_sha1", "exact_group_id", "representative_id"):
        values = {
            split: {chain_rows[sample_id].get(key, "") for sample_id in ids[split] if sample_id in chain_rows}
            - {""}
            for split in SPLITS
        }
        cross_split = _pairwise_intersections(values)
        identity_audits[key] = {
            "unique_counts": {split: len(value) for split, value in values.items()},
            "cross_split": cross_split,
        }
        if any(item["count"] for item in cross_split.values()):
            failures.append(f"{key} overlaps across splits")

    root_specs = {
        "source_esm": (data.get("esm_root"), _source_esm),
        "labels": (data.get("label_root"), _label),
        "sequence_features": (data.get("sequence_feature_root"), _sequence_feature),
        "primary_embeddings": (data.get("primary_embedding_root"), _embedding),
        "prottrans_embeddings": (data.get("prottrans_embedding_root"), _embedding),
        "contact_graphs": (data.get("contact_graph_root"), _contact),
        "aux_contact_graphs": (data.get("aux_contact_graph_root"), _contact),
    }
    resolved_path_audits = {
        name: _resolved_path_audit(_project_path(project_root, root_value), resolver, ids, chain_rows)
        for name, (root_value, resolver) in root_specs.items()
    }
    for name, audit in resolved_path_audits.items():
        if not audit["configured"]:
            continue
        missing_count = sum(item["count"] for item in audit["missing"].values())
        overlap_count = sum(item["count"] for item in audit["cross_split_resolved_path_overlap"].values())
        if missing_count:
            failures.append(f"{name} has {missing_count} unresolved files")
        if overlap_count:
            failures.append(f"{name} has {overlap_count} cross-split resolved file overlaps")

    evaluate_facts = _function_facts(training_script, "_evaluate")
    required_eval_facts = (
        evaluate_facts["calls_model_eval"]
        and evaluate_facts["uses_torch_no_grad"]
        and not evaluate_facts["contains_backward_call"]
        and not evaluate_facts["contains_optimizer_or_scaler_step"]
        and evaluate_facts["checks_logits_requires_grad"]
        and evaluate_facts["checks_parameter_versions"]
    )
    if not required_eval_facts:
        failures.append("_evaluate does not satisfy strict no-gradient/no-update static checks")
    if bool(training.get("eval_test_each_epoch", True)):
        failures.append("training config loads/evaluates the test split during model development")

    si_summary_path = _project_path(project_root, args.si_summary)
    si_violations_path = _project_path(project_root, args.si_violations)
    si_evidence = json.loads(si_summary_path.read_text()) if si_summary_path is not None else None
    if si_evidence is not None and int(si_evidence.get("n_si30_violations", -1)) != 0:
        failures.append("SI30 summary reports cross-split violations")
    if si_violations_path is not None:
        violation_lines = sum(1 for _ in si_violations_path.open())
        if violation_lines != 1:
            failures.append(f"SI30 violations file has {violation_lines - 1} data rows")

    report = {
        "status": "pass" if not failures else "fail",
        "inputs": {
            "config": str(config_path),
            "training_script": str(training_script),
            "split_dir": str(split_dir),
            "chain_manifest": str(chain_manifest_path),
            "training_manifest": str(training_manifest_path),
            "sha256": {
                "config": _sha256(config_path),
                "training_script": _sha256(training_script),
                "chain_manifest": _sha256(chain_manifest_path),
                "training_manifest": _sha256(training_manifest_path),
                **{f"{split}_ids": _sha256(path) for split, path in split_paths.items()},
            },
        },
        "split_counts": {split: len(values) for split, values in ids.items()},
        "duplicate_ids": {split: {"count": len(values), "examples": values[:20]} for split, values in duplicate_ids.items()},
        "manifest_coverage": {
            "split_union": len(union_ids),
            "chain_manifest_rows": len(chain_rows),
            "training_manifest_rows": len(training_rows),
            "missing_chain_manifest": missing_manifest[:20],
            "extra_chain_manifest": extra_manifest[:20],
            "missing_training_manifest": missing_training_manifest[:20],
            "extra_training_manifest": extra_training_manifest[:20],
            "split_field_mismatches": sorted(set(split_mismatches))[:20],
        },
        "identity_audits": identity_audits,
        "resolved_path_audits": resolved_path_audits,
        "training_protocol": {
            "test_split_loaded": bool(training.get("eval_test_each_epoch", True)),
            "selection_metric": training.get("selection_metric"),
            "strict_eval_gradient_isolation": bool(training.get("strict_eval_gradient_isolation", True)),
            "evaluate_function": evaluate_facts,
        },
        "si30_evidence": si_evidence,
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "README.md").write_text(_render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
