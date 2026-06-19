#!/usr/bin/env python3
"""Train sequence-only ESM residue contact-site model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ppcbind.data import ESMProteinSiteDataset, collate_esm_site_features
from ppcbind.models import ESMSiteClassifier

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


PATH_KEYS = {"esm_root", "label_root", "manifest", "split_dir", "sequence_feature_root", "output_dir"}


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["pdb_id"].lower()] = row
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config")
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return _normalize_keys(data)


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k).replace("-", "_"): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(v) for v in value]
    return value


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def _apply_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    for section_name in ("data", "training"):
        for key, value in _section(config, section_name).items():
            if key in PATH_KEYS and value is not None:
                value = Path(value)
            setattr(args, key, value)
    return dict(_section(config, "model"))


def _limit_ids(ids: list[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0:
        return ids
    return ids[:limit]


def _parse_float_grid(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return default
    if ":" in text and "," not in text:
        start, stop, step = (float(x) for x in text.split(":"))
        if step <= 0:
            raise ValueError(f"Invalid positive step in float grid: {text!r}")
        out: list[float] = []
        current = start
        while current <= stop + step * 0.5:
            out.append(round(current, 10))
            current += step
        return out
    return [float(x) for x in text.split(",") if x.strip()]


def _frac_label(frac: float) -> str:
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1.0e-6:
        return f"{int(round(pct))}pct"
    return f"{pct:.2f}".replace(".", "p") + "pct"


def _class_weight_from_manifest(manifest_path: Path, train_ids: list[str], max_pos_weight: float) -> tuple[float, int, int]:
    rows = _read_manifest(manifest_path)
    pos, neg = 0, 0
    for pdb_id in train_ids:
        if pdb_id not in rows:
            continue
        row = rows[pdb_id]
        pos += int(row.get("n_positive") or 0)
        neg += int(row.get("n_negative") or 0)
    if pos <= 0:
        return 1.0, pos, neg
    return min(float(neg / pos), max_pos_weight), pos, neg


def _build_loader(
    esm_root: Path,
    label_root: Path,
    ids: list[str],
    sequence_feature_root: Path | None,
    require_sequence_features: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    max_residues: int,
    crop_mode: str,
    seed: int,
    shuffle: bool,
    preload: bool = False,
    strict_ids: bool = True,
    require_labels: bool = True,
    strict_label_metadata: bool = True,
    require_contact_graph: bool = False,
) -> DataLoader:
    dataset = ESMProteinSiteDataset(
        esm_root=esm_root,
        label_root=label_root,
        ids=ids,
        sequence_feature_root=sequence_feature_root,
        require_sequence_features=require_sequence_features,
        max_residues=max_residues,
        crop_mode=crop_mode,
        seed=seed,
        preload=preload,
        strict_ids=strict_ids,
        require_labels=require_labels,
        strict_label_metadata=strict_label_metadata,
        require_contact_graph=require_contact_graph,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory) and torch.cuda.is_available(),
        "collate_fn": collate_esm_site_features,
    }
    if num_workers > 0 and prefetch_factor is not None and int(prefetch_factor) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    keys = ("esm_embeddings", "seq_features", "protein_mask", "chain_ids", "chain_rel_pos", "protein_rel_pos")
    inputs = {key: batch[key].to(device, non_blocking=True) for key in keys}
    for key in ("contact_edge_index", "contact_edge_scores"):
        if key in batch:
            inputs[key] = batch[key].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return inputs, labels


def _mcc_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / math.sqrt(denom))


def _metrics_from_counts(total: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = total["tp"], total["fp"], total["tn"], total["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        **total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mcc": _mcc_from_counts(tp, fp, tn, fn),
    }


def _step_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    mask = labels != -100
    if not bool(mask.any()):
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pred = logits.argmax(dim=-1)[mask]
    target = labels[mask]
    return {
        "tp": int(((pred == 1) & (target == 1)).sum().item()),
        "fp": int(((pred == 1) & (target == 0)).sum().item()),
        "tn": int(((pred == 0) & (target == 0)).sum().item()),
        "fn": int(((pred == 0) & (target == 1)).sum().item()),
    }


def _merge_metrics(metrics: list[dict[str, int]]) -> dict[str, float]:
    total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for item in metrics:
        for key in total:
            total[key] += item[key]
    return _metrics_from_counts(total)


def _counts_at_threshold(scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, int]:
    pred = scores >= float(threshold)
    target = targets.bool()
    return {
        "tp": int((pred & target).sum().item()),
        "fp": int((pred & ~target).sum().item()),
        "tn": int((~pred & ~target).sum().item()),
        "fn": int((~pred & target).sum().item()),
    }


def _auc_metrics(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float | None]:
    if scores.numel() == 0 or targets.numel() == 0 or int(targets.min().item()) == int(targets.max().item()):
        return {"auroc": None, "pr_auc": None}
    if average_precision_score is None or roc_auc_score is None:
        return {"auroc": None, "pr_auc": None}
    y_true = targets.cpu().numpy()
    y_score = scores.cpu().numpy()
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def _threshold_metrics(scores: torch.Tensor, targets: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    if scores.numel() == 0:
        return {"threshold_sweep": [], "best_threshold": None, "f1_best_threshold": 0.0, "mcc_best_threshold": 0.0}
    sweep: list[dict[str, float]] = []
    best_f1: dict[str, float] | None = None
    best_mcc: dict[str, float] | None = None
    for threshold in thresholds:
        row = {"threshold": float(threshold), **_metrics_from_counts(_counts_at_threshold(scores, targets, threshold))}
        sweep.append(row)
        if best_f1 is None or row["f1"] > best_f1["f1"]:
            best_f1 = row
        if best_mcc is None or row["mcc"] > best_mcc["mcc"]:
            best_mcc = row
    assert best_f1 is not None and best_mcc is not None
    at_05 = min(sweep, key=lambda row: abs(row["threshold"] - 0.5))
    return {
        "threshold_sweep": sweep,
        "threshold_0_5": at_05["threshold"],
        "f1_at_0_5": at_05["f1"],
        "mcc_at_0_5": at_05["mcc"],
        "best_threshold": best_f1["threshold"],
        "f1_best_threshold": best_f1["f1"],
        "precision_best_threshold": best_f1["precision"],
        "recall_best_threshold": best_f1["recall"],
        "mcc_at_best_f1_threshold": best_f1["mcc"],
        "best_mcc_threshold": best_mcc["threshold"],
        "mcc_best_threshold": best_mcc["mcc"],
        "f1_at_best_mcc_threshold": best_mcc["f1"],
    }


def _topk_metrics(per_protein: list[tuple[str, torch.Tensor, torch.Tensor]], topk_fracs: list[float]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"topk_n_proteins": len(per_protein)}
    for frac in topk_fracs:
        label = _frac_label(frac)
        macro_precisions: list[float] = []
        total_hits = 0
        total_k = 0
        total_pos = 0
        total_res = 0
        for _, scores, targets in per_protein:
            n_res = int(scores.numel())
            if n_res <= 0:
                continue
            k = max(1, int(math.ceil(n_res * float(frac))))
            k = min(k, n_res)
            top_idx = torch.topk(scores, k=k, largest=True).indices
            hits = int(targets[top_idx].sum().item())
            macro_precisions.append(hits / k)
            total_hits += hits
            total_k += k
            total_pos += int(targets.sum().item())
            total_res += n_res
        micro_precision = total_hits / max(1, total_k)
        base_rate = total_pos / max(1, total_res)
        out[f"top_{label}_precision_macro"] = sum(macro_precisions) / max(1, len(macro_precisions))
        out[f"top_{label}_precision_micro"] = micro_precision
        out[f"top_{label}_recall_micro"] = total_hits / max(1, total_pos)
        out[f"top_{label}_enrichment_micro"] = micro_precision / max(1.0e-12, base_rate)
    return out


def _score_metrics(
    score_chunks: list[torch.Tensor],
    target_chunks: list[torch.Tensor],
    per_protein: list[tuple[str, torch.Tensor, torch.Tensor]],
    thresholds: list[float],
    topk_fracs: list[float],
) -> dict[str, Any]:
    if not score_chunks:
        return {}
    scores = torch.cat(score_chunks).float()
    targets = torch.cat(target_chunks).long()
    return {
        **_auc_metrics(scores, targets),
        **_threshold_metrics(scores, targets, thresholds),
        **_topk_metrics(per_protein, topk_fracs),
    }


def _progress(iterable: Any, enabled: bool, total: int | None, desc: str) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False, mininterval=5.0)


def _set_lr(optimizer: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(scale)


def _safe_clip(parameters: list[torch.nn.Parameter], max_norm: float, value_clip: float | None) -> tuple[torch.Tensor, int]:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.zeros(()), 0
    n_bad = 0
    for param in params:
        grad = param.grad.detach()
        bad = ~torch.isfinite(grad)
        if bool(bad.any()):
            n_bad += int(bad.sum().item())
            grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if value_clip is not None and value_clip > 0:
            grad.clamp_(min=-float(value_clip), max=float(value_clip))
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if not bool(torch.isfinite(norm)):
        for param in params:
            param.grad.detach().zero_()
    return norm, n_bad


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    show_progress: bool,
    desc: str,
    thresholds: list[float],
    topk_fracs: list[float],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    metrics: list[dict[str, int]] = []
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    per_protein: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    skipped_nonfinite = 0
    with torch.no_grad():
        progress_iter = _progress(loader, show_progress, len(loader), desc)
        for step, batch in enumerate(progress_iter, 1):
            inputs, labels = _move_batch(batch, device)
            logits = model(**inputs)["logits"]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                skipped_nonfinite += 1
                continue
            losses.append(float(loss.item()))
            metrics.append(_step_metrics(logits, labels))
            valid = labels != -100
            probs = torch.softmax(logits.float(), dim=-1)[..., 1]
            if bool(valid.any()):
                score_chunks.append(probs[valid].detach().cpu())
                target_chunks.append(labels[valid].detach().cpu().long())
            for item_idx, pdb_id in enumerate(batch["pdb_id"]):
                item_valid = valid[item_idx]
                if bool(item_valid.any()):
                    per_protein.append(
                        (
                            str(pdb_id),
                            probs[item_idx][item_valid].detach().cpu(),
                            labels[item_idx][item_valid].detach().cpu().long(),
                        )
                    )
            if hasattr(progress_iter, "set_postfix"):
                progress_iter.set_postfix(loss=f"{losses[-1]:.4f}", skipped=skipped_nonfinite)
            if max_batches is not None and step >= max_batches:
                break
    merged = _merge_metrics(metrics)
    merged.update(_score_metrics(score_chunks, target_chunks, per_protein, thresholds, topk_fracs))
    merged["loss"] = sum(losses) / max(1, len(losses))
    merged["skipped_nonfinite"] = skipped_nonfinite
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--esm-root", default="features/esm2_t33_650M_UR50D/pt", type=Path)
    parser.add_argument("--label-root", default="features/contact_labels", type=Path)
    parser.add_argument("--manifest", default="features/contact_labels/manifest.csv", type=Path)
    parser.add_argument("--split-dir", default="features/contact_labels/splits_mmseq30_tmk_no_len_limit", type=Path)
    parser.add_argument("--sequence-feature-root", default=None, type=Path)
    parser.add_argument("--require-sequence-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-label-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-contact-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-residues", type=int, default=0)
    parser.add_argument("--train-crop-mode", choices=["none", "first", "random"], default="none")
    parser.add_argument("--eval-crop-mode", choices=["none", "first", "random"], default="none")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-scale", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--threshold-grid", default="0.01:0.99:0.01")
    parser.add_argument("--topk-fracs", default="0.05,0.10")
    parser.add_argument("--selection-metric", default="f1")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-value-clip", type=float, default=100.0)
    parser.add_argument("--adam-eps", type=float, default=1e-7)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    model_config = _apply_config(args, config)
    if args.output_dir is None:
        raise ValueError("training.output_dir must be set in YAML config")

    thresholds = _parse_float_grid(args.threshold_grid, [i / 100.0 for i in range(1, 100)])
    topk_fracs = _parse_float_grid(args.topk_fracs, [0.05, 0.10])

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = _limit_ids(_read_ids(args.split_dir / "train_ids.txt"), args.max_train_samples)
    val_ids = _limit_ids(_read_ids(args.split_dir / "val_ids.txt"), args.max_val_samples)
    test_ids = _read_ids(args.split_dir / "test_ids.txt")
    pos_weight, train_pos, train_neg = _class_weight_from_manifest(args.manifest, train_ids, args.max_pos_weight)
    require_contact_graph = bool(model_config.get("use_contact_graph")) if args.require_contact_graph is None else bool(args.require_contact_graph)

    train_loader = _build_loader(
        args.esm_root,
        args.label_root,
        train_ids,
        args.sequence_feature_root,
        args.require_sequence_features,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.prefetch_factor,
        args.max_residues,
        args.train_crop_mode,
        args.seed,
        True,
        preload=args.preload,
        strict_ids=args.strict_ids,
        require_labels=args.require_labels,
        strict_label_metadata=args.strict_label_metadata,
        require_contact_graph=require_contact_graph,
    )
    val_loader = _build_loader(
        args.esm_root,
        args.label_root,
        val_ids,
        args.sequence_feature_root,
        args.require_sequence_features,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.prefetch_factor,
        args.max_residues,
        args.eval_crop_mode,
        args.seed + 1,
        False,
        preload=args.preload,
        strict_ids=args.strict_ids,
        require_labels=args.require_labels,
        strict_label_metadata=args.strict_label_metadata,
        require_contact_graph=require_contact_graph,
    )

    model = ESMSiteClassifier(model_config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device),
        ignore_index=-100,
        label_smoothing=max(0.0, min(1.0, float(args.label_smoothing))),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and args.warmup_ratio > 0:
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    warmup_steps = min(warmup_steps, total_steps)
    min_lr_scale = max(0.0, min(1.0, float(args.min_lr_scale)))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if warmup_steps > 0:
        _set_lr(optimizer, base_lrs, min_lr_scale)

    run_config = vars(args).copy()
    run_config.update(
        {
            "yaml_config": config,
            "model_config": model_config,
            "device_resolved": str(device),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "train_positive_residues": train_pos,
            "train_negative_residues": train_neg,
            "pos_weight": pos_weight,
            "total_train_steps": total_steps,
            "warmup_steps_resolved": warmup_steps,
            "min_lr_scale_resolved": min_lr_scale,
            "thresholds_resolved": thresholds,
            "topk_fracs_resolved": topk_fracs,
            "input_policy": "ESM embeddings plus sequence-derived residue/global features only; optional ESM-predicted contact graph; no PDB coordinates or complete structure features",
            "strict_ids": args.strict_ids,
            "require_labels": args.require_labels,
            "strict_label_metadata": args.strict_label_metadata,
            "require_contact_graph_resolved": require_contact_graph,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
    log_path = args.output_dir / "metrics.jsonl"
    best_metric = -1.0
    best_metric_name = str(args.selection_metric)
    global_step = 0
    print(json.dumps({"event": "start", **run_config}, default=str), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        metrics: list[dict[str, int]] = []
        skipped_nonfinite = 0
        t0 = time.time()
        progress_iter = _progress(train_loader, args.progress, len(train_loader), f"train {epoch}/{args.epochs}")
        for step, batch in enumerate(progress_iter, 1):
            global_step += 1
            if warmup_steps > 0 and global_step <= warmup_steps:
                scale = min_lr_scale + (1.0 - min_lr_scale) * (global_step / warmup_steps)
                _set_lr(optimizer, base_lrs, scale)
            elif warmup_steps > 0 and global_step == warmup_steps + 1:
                _set_lr(optimizer, base_lrs, 1.0)
            inputs, labels = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(**inputs)["logits"]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                skipped_nonfinite += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm, n_bad_grad = _safe_clip(list(model.parameters()), args.grad_clip_norm, args.grad_value_clip)
            if n_bad_grad > 0:
                print(json.dumps({"event": "sanitized_grad", "epoch": epoch, "step": step, "n_bad_grad": n_bad_grad}), flush=True)
            if not bool(torch.isfinite(grad_norm)):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            metrics.append(_step_metrics(logits.detach(), labels))
            if step % 25 == 0:
                merged = _merge_metrics(metrics[-25:])
                window_loss = sum(losses[-25:]) / len(losses[-25:])
                if hasattr(progress_iter, "set_postfix"):
                    progress_iter.set_postfix(loss=f"{window_loss:.4f}", f1=f"{merged['f1']:.3f}")
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss": window_loss,
                            **merged,
                        }
                    ),
                    flush=True,
                )

        train_metrics = _merge_metrics(metrics)
        train_metrics["loss"] = sum(losses) / max(1, len(losses))
        train_metrics["skipped_nonfinite"] = skipped_nonfinite
        val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.eval_max_batches,
            show_progress=args.progress,
            desc=f"val {epoch}/{args.epochs}",
            thresholds=thresholds,
            topk_fracs=topk_fracs,
        )
        record = {
            "event": "epoch",
            "epoch": epoch,
            "seconds": time.time() - t0,
            "train": train_metrics,
            "val": val_metrics,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": run_config,
            "train": train_metrics,
            "val": val_metrics,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        current_metric = val_metrics.get(best_metric_name)
        if not isinstance(current_metric, (int, float)) or not math.isfinite(float(current_metric)):
            current_metric = val_metrics["f1"]
        if float(current_metric) > best_metric:
            best_metric = float(current_metric)
            torch.save(ckpt, args.output_dir / "best.pt")

    print(
        json.dumps(
            {
                "event": "done",
                "best_metric": best_metric,
                "selection_metric": best_metric_name,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
