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
from torch.utils.data import BatchSampler, DataLoader

from ppcbind.data import ESMProteinSiteDataset, collate_esm_site_features
from ppcbind.models import ESMSiteClassifier

# Limit OpenMP threads to reduce futex contention when running multiple training
# jobs with DataLoader workers. Empirically optimal value from sweep testing.
torch.set_num_threads(4)

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


PATH_KEYS = {
    "esm_root",
    "label_root",
    "manifest",
    "split_dir",
    "chain_filter_manifest",
    "sequence_feature_root",
    "primary_embedding_root",
    "prottrans_embedding_root",
    "contact_graph_root",
    "aux_contact_graph_root",
    "output_dir",
    "init_checkpoint",
}


def _read_ids(path: Path, lowercase: bool = True) -> list[str]:
    return [
        line.strip().lower() if lowercase else line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row["pdb_id"].strip()
            rows[key] = row
            rows.setdefault(key.lower(), row)
    return rows


def _split_ids_path(split_dir: Path, split: str, chain_filtered: bool) -> Path:
    if chain_filtered:
        chain_path = split_dir / f"{split}_chain_ids.txt"
        if chain_path.exists():
            return chain_path
    return split_dir / f"{split}_ids.txt"


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


def _coerce_config_value(args: argparse.Namespace, key: str, value: Any) -> Any:
    if key in PATH_KEYS and value is not None:
        return Path(value)
    if not hasattr(args, key) or value is None:
        return value
    current = getattr(args, key)
    if isinstance(current, bool):
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    for section_name in ("data", "training"):
        for key, value in _section(config, section_name).items():
            setattr(args, key, _coerce_config_value(args, key, value))
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


def _lengths_from_manifest(manifest_path: Path) -> dict[str, int]:
    rows = _read_manifest(manifest_path)
    lengths: dict[str, int] = {}
    for pdb_id, row in rows.items():
        value = row.get("n_residues")
        if value is None or value == "":
            continue
        key = str(pdb_id).strip()
        length = int(value)
        lengths[key] = length
        lengths.setdefault(key.lower(), length)
    return lengths


def _esm_length(path: Path) -> int:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        value = data.get("embeddings")
        if value is not None:
            return int(value.shape[0])
        length = int(data.get("feature_n_residues") or len(data.get("chain_ids", ())))
        if length > 0:
            return length
        raise ValueError(f"Cannot infer residue count from compact payload: {path}")
    return int(data.shape[0])


class LengthAwareBatchSampler(BatchSampler):
    """Pack examples with similar lengths to reduce padding waste.

    The token budget is measured as ``batch_size * max_length_in_batch`` because
    that is the tensor shape produced by ``collate_esm_site_features``.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        max_batch_tokens: int,
        shuffle: bool,
        seed: int,
        bucket_size: int,
    ) -> None:
        self.lengths = [max(1, int(length)) for length in lengths]
        self.batch_size = max(1, int(batch_size))
        self.max_batch_tokens = max(0, int(max_batch_tokens))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(0, int(bucket_size))
        self._epoch = 0

    def _ordered_indices(self, rng: random.Random | None = None) -> list[int]:
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            assert rng is not None
            rng.shuffle(indices)
            if self.bucket_size > 1:
                ordered: list[int] = []
                for start in range(0, len(indices), self.bucket_size):
                    bucket = indices[start : start + self.bucket_size]
                    bucket.sort(key=lambda idx: self.lengths[idx])
                    ordered.extend(bucket)
                return ordered
            return indices
        indices.sort(key=lambda idx: self.lengths[idx])
        return indices

    def _pack(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        max_len = 0
        for idx in indices:
            length = self.lengths[idx]
            candidate_max = max(max_len, length)
            candidate_size = len(batch) + 1
            over_items = len(batch) >= self.batch_size
            over_tokens = (
                self.max_batch_tokens > 0
                and bool(batch)
                and candidate_size * candidate_max > self.max_batch_tokens
            )
            if over_items or over_tokens:
                batches.append(batch)
                batch = [idx]
                max_len = length
            else:
                batch.append(idx)
                max_len = candidate_max
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        yield from self._pack(self._ordered_indices(rng if self.shuffle else None))

    def __len__(self) -> int:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: self.lengths[idx])
        return len(self._pack(indices))


def _build_loader(
    esm_root: Path,
    label_root: Path,
    ids: list[str],
    sequence_feature_root: Path | None,
    primary_embedding_root: Path | None,
    contact_graph_root: Path | None,
    aux_contact_graph_root: Path | None,
    prottrans_embedding_root: Path | None,
    require_sequence_features: bool,
    require_primary_embeddings: bool,
    require_prottrans_embeddings: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool,
    payload_cache_size: int,
    max_residues: int,
    crop_mode: str,
    seed: int,
    shuffle: bool,
    batching: str = "random",
    max_batch_tokens: int = 0,
    length_bucket_size: int = 0,
    lengths_by_id: dict[str, int] | None = None,
    preload: bool = False,
    strict_ids: bool = True,
    require_labels: bool = True,
    strict_label_metadata: bool = True,
    strict_sequence_feature_metadata: bool = True,
    require_contact_graph: bool = False,
    require_aux_contact_graph: bool = False,
    chain_filter_manifest: Path | None = None,
) -> DataLoader:
    dataset = ESMProteinSiteDataset(
        esm_root=esm_root,
        label_root=label_root,
        ids=ids,
        sequence_feature_root=sequence_feature_root,
        primary_embedding_root=primary_embedding_root,
        prottrans_embedding_root=prottrans_embedding_root,
        contact_graph_root=contact_graph_root,
        aux_contact_graph_root=aux_contact_graph_root,
        require_sequence_features=require_sequence_features,
        require_primary_embeddings=require_primary_embeddings,
        require_prottrans_embeddings=require_prottrans_embeddings,
        max_residues=max_residues,
        crop_mode=crop_mode,
        seed=seed,
        preload=preload,
        strict_ids=strict_ids,
        require_labels=require_labels,
        strict_label_metadata=strict_label_metadata,
        strict_sequence_feature_metadata=strict_sequence_feature_metadata,
        require_contact_graph=require_contact_graph,
        require_aux_contact_graph=require_aux_contact_graph,
        chain_filter_manifest=chain_filter_manifest,
        payload_cache_size=payload_cache_size,
    )
    use_length_batches = str(batching or "random").lower() != "random" or int(max_batch_tokens or 0) > 0
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory) and torch.cuda.is_available(),
        "collate_fn": collate_esm_site_features,
    }
    if use_length_batches:
        lengths: list[int] = []
        sample_ids = getattr(dataset, "sample_ids", None)
        fallback_ids: list[str] = []
        for idx, path in enumerate(dataset.esm_paths):
            pdb_id = str(sample_ids[idx]) if sample_ids is not None else path.parent.name.lower()
            if sample_ids is None and path.stem.endswith("_esm2"):
                pdb_id = path.stem[: -len("_esm2")].lower()
            elif sample_ids is None and path.stem.endswith("_protein"):
                pdb_id = path.stem[: -len("_protein")].lower()
            length = None
            if lengths_by_id is not None:
                length = lengths_by_id.get(pdb_id)
                if length is None:
                    length = lengths_by_id.get(pdb_id.lower())
            if length is None:
                fallback_ids.append(pdb_id)
                length = _esm_length(path)
            lengths.append(int(length))
        if fallback_ids:
            preview = ", ".join(fallback_ids[:5])
            print(
                f"[LengthAwareBatchSampler] manifest length missing for {len(fallback_ids)} "
                f"examples; loaded ESM tensors for fallback lengths. first={preview}",
                flush=True,
            )
        loader_kwargs["batch_sampler"] = LengthAwareBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            shuffle=shuffle,
            seed=seed,
            bucket_size=length_bucket_size,
        )
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    if num_workers > 0 and prefetch_factor is not None and int(prefetch_factor) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(dataset, **loader_kwargs)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    keys = ("esm_embeddings", "seq_features", "protein_mask", "chain_ids", "chain_rel_pos", "protein_rel_pos")
    inputs = {key: batch[key].to(device, non_blocking=True) for key in keys}
    if "prottrans_embeddings" in batch:
        inputs["prottrans_embeddings"] = batch["prottrans_embeddings"].to(device, non_blocking=True)
    for key in ("contact_edge_index", "contact_edge_scores", "aux_contact_edge_index", "aux_contact_edge_scores"):
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


def _metrics_at_threshold(metrics: dict[str, Any], threshold: float | None) -> dict[str, Any]:
    if threshold is None:
        return {}
    sweep = metrics.get("threshold_sweep") or []
    if not sweep:
        return {}
    row = min(sweep, key=lambda item: abs(float(item["threshold"]) - float(threshold)))
    return {
        "threshold": row.get("threshold"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "mcc": row.get("mcc"),
        "accuracy": row.get("accuracy"),
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


def _write_explaining_md(
    output_dir: Path,
    run_config: dict[str, Any],
    best_metric: float | None = None,
    status: str = "finished",
) -> None:
    metadata = run_config.get("yaml_config", {}).get("metadata", {})
    model_config = run_config.get("model_config", {})
    lines = [
        "# Experiment Explanation",
        "",
        f"- Run directory: `{output_dir}`",
        f"- Experiment time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Machine / source: `{run_config.get('device_resolved')}`",
        f"- Parent experiment: {metadata.get('parent_experiment', 'current strict MLC + PS + contact baseline')}",
        f"- Data split / protocol: `{run_config.get('split_dir')}`",
        f"- Purpose: {metadata.get('experiment_role', metadata.get('stage', 'own-model training run'))}",
        f"- Problem to solve: {metadata.get('problem_to_solve', 'Improve residue-level effect-site prediction under strict split.')}",
        f"- Main change: {metadata.get('main_change', json.dumps(model_config, default=str))}",
        f"- Expected outcome: {metadata.get('expected_outcome', 'Improve PR-AUC/F1/MCC or top-k ranking over parent baseline.')}",
        f"- Result brief: status={status}; selection_metric={run_config.get('selection_metric')}; best_metric={best_metric}",
        "- Key metrics: see `metrics.jsonl`, `best.pt`, and downstream strict evaluation summaries.",
        f"- Decision: {metadata.get('decision', 'pending strict evaluation')}",
        f"- Notes: {metadata.get('notes', '')}",
        "",
    ]
    (output_dir / "explaning.md").write_text("\n".join(lines))


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


class ResidueSiteLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        label_smoothing: float,
        loss_type: str,
        dice_weight: float,
        focal_gamma: float,
        focal_alpha: float | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type or "ce").lower()
        if self.loss_type not in {"ce", "ce_dice", "focal", "focal_dice"}:
            raise ValueError(f"Unsupported loss_type={loss_type!r}; use ce, ce_dice, focal, or focal_dice")
        self.dice_weight = max(0.0, float(dice_weight))
        self.focal_gamma = max(0.0, float(focal_gamma))
        self.focal_alpha = None if focal_alpha is None or focal_alpha < 0 else max(0.0, min(1.0, float(focal_alpha)))
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device),
            ignore_index=-100,
            label_smoothing=max(0.0, min(1.0, float(label_smoothing))),
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ce":
            return self.ce(logits, labels)
        valid = labels != -100
        if not bool(valid.any()):
            return logits.sum() * 0.0
        valid_logits = logits[valid]
        valid_labels = labels[valid].long()
        if self.loss_type.startswith("focal"):
            log_probs = torch.log_softmax(valid_logits.float(), dim=-1)
            probs = torch.exp(log_probs)
            target_log_probs = log_probs.gather(1, valid_labels.view(-1, 1)).squeeze(1)
            target_probs = probs.gather(1, valid_labels.view(-1, 1)).squeeze(1)
            focal = -((1.0 - target_probs).clamp_min(1.0e-6) ** self.focal_gamma) * target_log_probs
            if self.focal_alpha is not None:
                alpha = torch.where(
                    valid_labels == 1,
                    torch.full_like(focal, self.focal_alpha),
                    torch.full_like(focal, 1.0 - self.focal_alpha),
                )
                focal = focal * alpha
            base = focal.mean()
        else:
            base = self.ce(logits, labels)
        if "dice" not in self.loss_type or self.dice_weight <= 0.0:
            return base
        scores = torch.softmax(valid_logits.float(), dim=-1)[:, 1]
        targets = (valid_labels == 1).float()
        intersection = (scores * targets).sum()
        denom = scores.sum() + targets.sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
        return base + self.dice_weight * dice.to(dtype=base.dtype)


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
    parser.add_argument("--chain-filter-manifest", default=None, type=Path)
    parser.add_argument("--sequence-feature-root", default=None, type=Path)
    parser.add_argument("--primary-embedding-root", default=None, type=Path)
    parser.add_argument("--prottrans-embedding-root", default=None, type=Path)
    parser.add_argument("--contact-graph-root", default=None, type=Path)
    parser.add_argument("--aux-contact-graph-root", default=None, type=Path)
    parser.add_argument("--require-sequence-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-primary-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-prottrans-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-label-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-sequence-feature-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-contact-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-aux-contact-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--init-checkpoint", default=None, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--payload-cache-size", type=int, default=0)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batching", choices=["random", "length_sorted", "token_budget"], default="random")
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--length-bucket-size", type=int, default=0)
    parser.add_argument("--max-residues", type=int, default=0)
    parser.add_argument("--train-crop-mode", choices=["none", "first", "random"], default="none")
    parser.add_argument("--eval-crop-mode", choices=["none", "first", "random"], default="none")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--eval-test-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-non-prottrans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-scale", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=["ce", "ce_dice", "focal", "focal_dice"], default="ce")
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=-1.0)
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

    chain_filtered = args.chain_filter_manifest is not None
    train_ids = _limit_ids(
        _read_ids(_split_ids_path(args.split_dir, "train", chain_filtered), lowercase=not chain_filtered),
        args.max_train_samples,
    )
    val_ids = _limit_ids(
        _read_ids(_split_ids_path(args.split_dir, "val", chain_filtered), lowercase=not chain_filtered),
        args.max_val_samples,
    )
    test_ids = _read_ids(_split_ids_path(args.split_dir, "test", chain_filtered), lowercase=not chain_filtered)
    lengths_by_id = _lengths_from_manifest(args.manifest)
    pos_weight, train_pos, train_neg = _class_weight_from_manifest(args.manifest, train_ids, args.max_pos_weight)
    require_contact_graph = bool(model_config.get("use_contact_graph")) if args.require_contact_graph is None else bool(args.require_contact_graph)

    train_loader = _build_loader(
        args.esm_root,
        args.label_root,
        train_ids,
        args.sequence_feature_root,
        args.primary_embedding_root,
        args.contact_graph_root,
        args.aux_contact_graph_root,
        args.prottrans_embedding_root,
        args.require_sequence_features,
        args.require_primary_embeddings,
        args.require_prottrans_embeddings,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.prefetch_factor,
        args.persistent_workers,
        args.payload_cache_size,
        args.max_residues,
        args.train_crop_mode,
        args.seed,
        True,
        batching=args.batching,
        max_batch_tokens=args.max_batch_tokens,
        length_bucket_size=args.length_bucket_size,
        lengths_by_id=lengths_by_id,
        preload=args.preload,
        strict_ids=args.strict_ids,
        require_labels=args.require_labels,
        strict_label_metadata=args.strict_label_metadata,
        strict_sequence_feature_metadata=args.strict_sequence_feature_metadata,
        require_contact_graph=require_contact_graph,
        require_aux_contact_graph=args.require_aux_contact_graph,
        chain_filter_manifest=args.chain_filter_manifest,
    )
    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
    val_loader = _build_loader(
        args.esm_root,
        args.label_root,
        val_ids,
        args.sequence_feature_root,
        args.primary_embedding_root,
        args.contact_graph_root,
        args.aux_contact_graph_root,
        args.prottrans_embedding_root,
        args.require_sequence_features,
        args.require_primary_embeddings,
        args.require_prottrans_embeddings,
        eval_batch_size,
        args.num_workers,
        args.pin_memory,
        args.prefetch_factor,
        args.persistent_workers,
        args.payload_cache_size,
        args.max_residues,
        args.eval_crop_mode,
        args.seed + 1,
        False,
        batching=args.batching,
        max_batch_tokens=args.max_batch_tokens,
        length_bucket_size=args.length_bucket_size,
        lengths_by_id=lengths_by_id,
        preload=args.preload,
        strict_ids=args.strict_ids,
        require_labels=args.require_labels,
        strict_label_metadata=args.strict_label_metadata,
        strict_sequence_feature_metadata=args.strict_sequence_feature_metadata,
        require_contact_graph=require_contact_graph,
        require_aux_contact_graph=args.require_aux_contact_graph,
        chain_filter_manifest=args.chain_filter_manifest,
    )
    test_loader = (
        _build_loader(
            args.esm_root,
            args.label_root,
            test_ids,
            args.sequence_feature_root,
            args.primary_embedding_root,
            args.contact_graph_root,
            args.aux_contact_graph_root,
            args.prottrans_embedding_root,
            args.require_sequence_features,
            args.require_primary_embeddings,
            args.require_prottrans_embeddings,
            eval_batch_size,
            args.num_workers,
            args.pin_memory,
            args.prefetch_factor,
            args.persistent_workers,
            args.payload_cache_size,
            args.max_residues,
            args.eval_crop_mode,
            args.seed + 2,
            False,
            batching=args.batching,
            max_batch_tokens=args.max_batch_tokens,
            length_bucket_size=args.length_bucket_size,
            lengths_by_id=lengths_by_id,
            preload=args.preload,
            strict_ids=args.strict_ids,
            require_labels=args.require_labels,
            strict_label_metadata=args.strict_label_metadata,
            strict_sequence_feature_metadata=args.strict_sequence_feature_metadata,
            require_contact_graph=require_contact_graph,
            require_aux_contact_graph=args.require_aux_contact_graph,
            chain_filter_manifest=args.chain_filter_manifest,
        )
        if args.eval_test_each_epoch
        else None
    )

    model = ESMSiteClassifier(model_config).to(device)
    init_report: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
            elif "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint
        else:
            state = checkpoint
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported init checkpoint format: {args.init_checkpoint}")
        load_result = model.load_state_dict(state, strict=False)
        init_report = {
            "path": str(args.init_checkpoint),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }
        print(
            "Initialized model from "
            f"{args.init_checkpoint} with {len(load_result.missing_keys)} missing "
            f"and {len(load_result.unexpected_keys)} unexpected keys",
            flush=True,
        )
    if args.freeze_non_prottrans:
        for name, param in model.named_parameters():
            if not name.startswith("prottrans_"):
                param.requires_grad_(False)
        n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        n_frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
        print(
            json.dumps(
                {
                    "event": "freeze_non_prottrans",
                    "trainable_parameters": n_trainable,
                    "frozen_parameters": n_frozen,
                }
            ),
            flush=True,
        )
    criterion = ResidueSiteLoss(
        pos_weight=pos_weight,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        dice_weight=args.dice_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        device=device,
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
            "n_train_batches": len(train_loader),
            "n_val_batches": len(val_loader),
            "n_test_batches": len(test_loader) if test_loader is not None else 0,
            "batching": args.batching,
            "max_batch_tokens": args.max_batch_tokens,
            "length_bucket_size": args.length_bucket_size,
            "warmup_steps_resolved": warmup_steps,
            "min_lr_scale_resolved": min_lr_scale,
            "thresholds_resolved": thresholds,
            "topk_fracs_resolved": topk_fracs,
            "input_policy": "ESM embeddings plus sequence-derived residue/global features only; optional ESM-predicted contact graph; no PDB coordinates or complete structure features",
            "chain_filtered": chain_filtered,
            "chain_filter_manifest": str(args.chain_filter_manifest) if args.chain_filter_manifest else None,
            "strict_ids": args.strict_ids,
            "require_labels": args.require_labels,
            "strict_label_metadata": args.strict_label_metadata,
            "strict_sequence_feature_metadata": args.strict_sequence_feature_metadata,
            "primary_embedding_root": str(args.primary_embedding_root) if args.primary_embedding_root else None,
            "require_primary_embeddings": args.require_primary_embeddings,
            "persistent_workers": args.persistent_workers,
            "payload_cache_size": args.payload_cache_size,
            "require_contact_graph_resolved": require_contact_graph,
            "require_aux_contact_graph": args.require_aux_contact_graph,
            "prottrans_embedding_root": str(args.prottrans_embedding_root) if args.prottrans_embedding_root else None,
            "require_prottrans_embeddings": args.require_prottrans_embeddings,
            "eval_test_each_epoch": args.eval_test_each_epoch,
            "init_checkpoint_report": init_report,
            "freeze_non_prottrans": args.freeze_non_prottrans,
            "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
            "frozen_parameters": sum(param.numel() for param in model.parameters() if not param.requires_grad),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
    _write_explaining_md(args.output_dir, run_config, best_metric=None, status="started")
    log_path = args.output_dir / "metrics.jsonl"
    best_metric = -1.0
    best_metric_name = str(args.selection_metric)
    global_step = 0
    print(json.dumps({"event": "start", **run_config}, default=str), flush=True)

    if args.eval_only or args.epochs <= 0:
        val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.eval_max_batches,
            show_progress=args.progress,
            desc="val eval-only",
            thresholds=thresholds,
            topk_fracs=topk_fracs,
        )
        test_metrics = None
        test_at_val_threshold = None
        if test_loader is not None:
            test_metrics = _evaluate(
                model,
                test_loader,
                criterion,
                device,
                max_batches=args.eval_max_batches,
                show_progress=args.progress,
                desc="test eval-only",
                thresholds=thresholds,
                topk_fracs=topk_fracs,
            )
            test_at_val_threshold = _metrics_at_threshold(test_metrics, val_metrics.get("best_threshold"))
        record = {
            "event": "eval_only",
            "epoch": 0,
            "val": val_metrics,
        }
        if test_metrics is not None:
            record["test"] = test_metrics
            record["test_at_val_threshold"] = test_at_val_threshold
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 0,
                "config": run_config,
                "val": val_metrics,
                "test": test_metrics,
            },
            args.output_dir / "last.pt",
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 0,
                "config": run_config,
                "val": val_metrics,
                "test": test_metrics,
            },
            args.output_dir / "best.pt",
        )
        metric_value = val_metrics.get(best_metric_name)
        best_metric = float(metric_value) if isinstance(metric_value, (int, float)) else None
        _write_explaining_md(args.output_dir, run_config, best_metric=best_metric, status="eval_only")
        return 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        metrics: list[dict[str, int]] = []
        skipped_nonfinite = 0
        t0 = time.time()
        train_loop_start = time.perf_counter()
        last_step_end = train_loop_start
        data_wait_seconds = 0.0
        compute_seconds = 0.0
        train_residues = 0
        progress_iter = _progress(train_loader, args.progress, len(train_loader), f"train {epoch}/{args.epochs}")
        for step, batch in enumerate(progress_iter, 1):
            batch_ready = time.perf_counter()
            data_wait_seconds += batch_ready - last_step_end
            global_step += 1
            if warmup_steps > 0 and global_step <= warmup_steps:
                scale = min_lr_scale + (1.0 - min_lr_scale) * (global_step / warmup_steps)
                _set_lr(optimizer, base_lrs, scale)
            elif warmup_steps > 0 and global_step == warmup_steps + 1:
                _set_lr(optimizer, base_lrs, 1.0)
            inputs, labels = _move_batch(batch, device)
            train_residues += int((labels != -100).sum().item())
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(**inputs)["logits"]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                skipped_nonfinite += 1
                step_end = time.perf_counter()
                compute_seconds += step_end - batch_ready
                last_step_end = step_end
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm, n_bad_grad = _safe_clip(list(model.parameters()), args.grad_clip_norm, args.grad_value_clip)
            if n_bad_grad > 0:
                print(json.dumps({"event": "sanitized_grad", "epoch": epoch, "step": step, "n_bad_grad": n_bad_grad}), flush=True)
            if not bool(torch.isfinite(grad_norm)):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                step_end = time.perf_counter()
                compute_seconds += step_end - batch_ready
                last_step_end = step_end
                continue
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            metrics.append(_step_metrics(logits.detach(), labels))
            step_end = time.perf_counter()
            compute_seconds += step_end - batch_ready
            last_step_end = step_end
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
        train_loop_seconds = time.perf_counter() - train_loop_start
        train_metrics["train_loop_seconds"] = train_loop_seconds
        train_metrics["data_wait_seconds"] = data_wait_seconds
        train_metrics["compute_seconds"] = compute_seconds
        train_metrics["data_wait_fraction"] = data_wait_seconds / max(1.0e-9, data_wait_seconds + compute_seconds)
        train_metrics["residues_per_second"] = train_residues / max(1.0e-9, train_loop_seconds)
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
        test_metrics = None
        test_at_val_threshold = None
        if test_loader is not None:
            test_metrics = _evaluate(
                model,
                test_loader,
                criterion,
                device,
                max_batches=args.eval_max_batches,
                show_progress=args.progress,
                desc=f"test {epoch}/{args.epochs}",
                thresholds=thresholds,
                topk_fracs=topk_fracs,
            )
            test_at_val_threshold = _metrics_at_threshold(test_metrics, val_metrics.get("best_threshold"))
        record = {
            "event": "epoch",
            "epoch": epoch,
            "seconds": time.time() - t0,
            "train": train_metrics,
            "val": val_metrics,
        }
        if test_metrics is not None:
            record["test"] = test_metrics
            record["test_at_val_threshold"] = test_at_val_threshold
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
    _write_explaining_md(args.output_dir, run_config, best_metric=best_metric, status="finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
