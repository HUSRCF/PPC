#!/usr/bin/env python3
"""Train PPC residue-level protein-protein contact-site model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ppcbind.data import ProteinFeatureDataset, collate_protein_features
from ppcbind.models.protein_site_gvp import ProteinSiteGVP

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["pdb_id"].lower()] = row
    return rows


def _make_or_load_splits(
    manifest_path: Path,
    split_dir: Path,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    train_path = split_dir / "train_ids.txt"
    val_path = split_dir / "val_ids.txt"
    test_path = split_dir / "test_ids.txt"
    if train_path.exists() and val_path.exists() and test_path.exists():
        return _read_ids(train_path), _read_ids(val_path), _read_ids(test_path)

    rows = _read_manifest(manifest_path)
    eligible = [
        pdb_id
        for pdb_id, row in rows.items()
        if row.get("status") in {"OK", "SKIP"} and int(row.get("n_positive") or 0) > 0
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_total = len(eligible)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    train_ids = eligible[:n_train]
    val_ids = eligible[n_train : n_train + n_val]
    test_ids = eligible[n_train + n_val :]
    _write_ids(train_path, train_ids)
    _write_ids(val_path, val_ids)
    _write_ids(test_path, test_ids)
    return train_ids, val_ids, test_ids


def _limit_ids(ids: list[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0:
        return ids
    return ids[:limit]


def _class_weight_from_manifest(
    manifest_path: Path,
    train_ids: list[str],
    max_pos_weight: float,
) -> tuple[float, int, int]:
    rows = _read_manifest(manifest_path)
    pos = 0
    neg = 0
    for pdb_id in train_ids:
        row = rows[pdb_id]
        pos += int(row["n_positive"])
        neg += int(row["n_negative"])
    if pos <= 0:
        return 1.0, pos, neg
    return min(float(neg / pos), max_pos_weight), pos, neg


def _move_batch(batch: dict[str, Any], device: torch.device, use_esm: bool) -> dict[str, Any]:
    keys = [
        "protein_physchem",
        "protein_spatial_scalar",
        "protein_spatial_vector",
        "protein_backbone_vector",
        "protein_coords",
        "protein_mask",
        "chain_ids",
    ]
    out = {key: batch[key].to(device, non_blocking=True) for key in keys}
    if use_esm:
        out["esm_embeddings"] = batch["esm_embeddings"].to(device, non_blocking=True)
    return out


def _step_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    mask = labels != -100
    if not bool(mask.any()):
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pred = logits.argmax(dim=-1)
    pred = pred[mask]
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
    tp, fp, tn, fn = total["tp"], total["fp"], total["tn"], total["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    out = dict(total)
    out.update({"precision": precision, "recall": recall, "f1": f1, "accuracy": acc})
    return out


def _make_progress(iterable: Any, *, enabled: bool, total: int | None, desc: str) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        mininterval=5.0,
        smoothing=0.05,
    )


def _safe_clip_grad_norm_(
    parameters,
    max_norm: float,
    value_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip gradients with explicit finite checks and fp64 norm accumulation."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros((), device=device), {"n_nonfinite_grad_values": 0, "n_tensors_sanitized": 0}

    device = params[0].grad.device
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    n_nonfinite = 0
    n_tensors_sanitized = 0
    max_abs_grad = 0.0

    for param in params:
        grad = param.grad.detach()
        bad = ~torch.isfinite(grad)
        if bool(bad.any()):
            n_bad = int(bad.sum().item())
            n_nonfinite += n_bad
            n_tensors_sanitized += 1
            grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if value_clip is not None and value_clip > 0:
            grad.clamp_(min=-float(value_clip), max=float(value_clip))
        grad64 = grad.double()
        total_sq = total_sq + torch.sum(grad64 * grad64)
        if grad.numel() > 0:
            max_abs_grad = max(max_abs_grad, float(grad.float().abs().max().item()))

    total_norm = torch.sqrt(total_sq).to(dtype=torch.float32)
    if bool(torch.isfinite(total_norm)):
        clip_coef = float(max_norm) / (float(total_norm.item()) + 1e-6)
        if clip_coef < 1.0:
            for param in params:
                param.grad.detach().mul_(clip_coef)
    else:
        for param in params:
            param.grad.detach().zero_()

    return total_norm, {
        "n_nonfinite_grad_values": n_nonfinite,
        "n_tensors_sanitized": n_tensors_sanitized,
        "max_abs_grad": max_abs_grad,
    }


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(scale)


def _iter_tensors(value: Any, prefix: str = "tensor"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}.{idx}")


def _tensor_probe_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    n_total = value.numel()
    n_nonfinite = int((~finite).sum().item())
    stats: dict[str, Any] = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "n_total": int(n_total),
        "n_nonfinite": n_nonfinite,
    }
    if n_total > 0 and bool(finite.any().item()) and value.is_floating_point():
        finite_values = value[finite].float()
        stats.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "abs_max": float(finite_values.abs().max().item()),
            }
        )
    return stats


class NanProbe:
    """Lightweight module and parameter probes for non-finite tensors."""

    def __init__(self, max_reports: int = 20) -> None:
        self.max_reports = int(max_reports)
        self.n_reports = 0
        self.handles: list[Any] = []
        self.context: dict[str, Any] = {}
        self.param_to_module: dict[str, str] = {}
        self.last_module_stats: dict[str, dict[str, Any]] = {}

    def _can_report(self) -> bool:
        return self.n_reports < self.max_reports

    def _emit(self, event: dict[str, Any]) -> None:
        if not self._can_report():
            return
        self.n_reports += 1
        print(json.dumps(event, default=str), flush=True)

    def _find_bad(self, value: Any, prefix: str) -> list[dict[str, Any]]:
        bad: list[dict[str, Any]] = []
        for name, tensor in _iter_tensors(value, prefix):
            if tensor.numel() == 0:
                continue
            if not bool(torch.isfinite(tensor).all().item()):
                bad.append(_tensor_probe_stats(name, tensor))
        return bad

    def _compact_stats(self, value: Any, prefix: str, limit: int = 4) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        for name, tensor in _iter_tensors(value, prefix):
            if len(stats) >= limit:
                break
            if tensor.numel() == 0 or not tensor.is_floating_point():
                continue
            stats.append(_tensor_probe_stats(name, tensor))
        return stats

    def set_context(self, epoch: int, step: int, batch: dict[str, Any]) -> None:
        self.context = {"epoch": epoch, "step": step, "pdb_id": batch.get("pdb_id")}

    def attach(self, model: nn.Module, module_backward_hooks: bool = False) -> None:
        for module_name, module in model.named_modules():
            if not module_name:
                continue

            def forward_hook(mod, inputs, output, name=module_name):
                if any(param.requires_grad for param in mod.parameters(recurse=False)):
                    self.last_module_stats[name] = {
                        "module_type": mod.__class__.__name__,
                        "input": self._compact_stats(inputs, "input"),
                        "output": self._compact_stats(output, "output"),
                    }
                bad_output = self._find_bad(output, "output")
                if bad_output:
                    self._emit(
                        {
                            "event": "nan_probe_forward",
                            "module": name,
                            "module_type": mod.__class__.__name__,
                            "bad_output": bad_output,
                            "bad_input": self._find_bad(inputs, "input"),
                        }
                    )

            def backward_hook(mod, grad_input, grad_output, name=module_name):
                bad_grad_output = self._find_bad(grad_output, "grad_output")
                bad_grad_input = self._find_bad(grad_input, "grad_input")
                if bad_grad_output or bad_grad_input:
                    self._emit(
                        {
                            "event": "nan_probe_backward",
                            "module": name,
                            "module_type": mod.__class__.__name__,
                            "bad_grad_output": bad_grad_output,
                            "bad_grad_input": bad_grad_input,
                        }
                    )

            self.handles.append(module.register_forward_hook(forward_hook))
            if module_backward_hooks:
                self.handles.append(module.register_full_backward_hook(backward_hook))

        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            module_name = param_name.rsplit(".", 1)[0] if "." in param_name else ""
            self.param_to_module[param_name] = module_name

            def param_hook(grad, name=param_name):
                if not bool(torch.isfinite(grad).all().item()):
                    module = self.param_to_module.get(name, "")
                    self._emit(
                        {
                            "event": "nan_probe_param_hook",
                            **self.context,
                            "parameter": name,
                            "module": module,
                            "grad": _tensor_probe_stats(name, grad),
                            "last_forward": self.last_module_stats.get(module),
                        }
                    )
                return grad

            self.handles.append(param.register_hook(param_hook))

    def check_inputs(self, batch: dict[str, Any], inputs: dict[str, Any], labels: torch.Tensor, epoch: int, step: int) -> None:
        bad_inputs = self._find_bad(inputs, "inputs")
        bad_labels = self._find_bad(labels, "labels")
        if bad_inputs or bad_labels:
            self._emit(
                {
                    "event": "nan_probe_batch",
                    "epoch": epoch,
                    "step": step,
                    "pdb_id": batch.get("pdb_id"),
                    "bad_inputs": bad_inputs,
                    "bad_labels": bad_labels,
                }
            )

    def collect_parameter_grads(self, model: nn.Module) -> dict[str, Any]:
        bad: list[dict[str, Any]] = []
        top_finite: list[dict[str, Any]] = []
        total_sq: torch.Tensor | None = None
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            finite = torch.isfinite(grad)
            if not bool(finite.all().item()):
                bad.append(_tensor_probe_stats(name, grad))
            if grad.numel() > 0:
                safe_grad = torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
                norm = torch.linalg.vector_norm(safe_grad, ord=2)
                if total_sq is None:
                    total_sq = torch.zeros((), device=norm.device, dtype=torch.float32)
                total_sq = total_sq + norm.to(device=total_sq.device, dtype=torch.float32).pow(2)
                top_finite.append(
                    {
                        "name": name,
                        "shape": list(grad.shape),
                        "norm": float(norm.item()),
                        "abs_max": float(safe_grad.abs().max().item()),
                    }
                )
        top_finite.sort(key=lambda item: item["norm"], reverse=True)
        safe_total_norm = None
        if total_sq is not None:
            safe_total_norm = float(torch.sqrt(total_sq).item())
        return {
            "bad": bad[:40],
            "n_bad_parameters": len(bad),
            "top_finite": top_finite[:20],
            "safe_total_norm": safe_total_norm,
        }

    def emit_parameter_grads(
        self,
        report: dict[str, Any],
        epoch: int,
        step: int,
        batch: dict[str, Any],
        phase: str,
    ) -> None:
        self._emit(
            {
                "event": "nan_probe_parameter_grads",
                "epoch": epoch,
                "step": step,
                "pdb_id": batch.get("pdb_id"),
                "phase": phase,
                **report,
            }
        )

    def report_parameter_grads(self, model: nn.Module, epoch: int, step: int, batch: dict[str, Any], phase: str) -> None:
        self.emit_parameter_grads(self.collect_parameter_grads(model), epoch, step, batch, phase)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _build_loader(
    features_root: Path,
    label_root: Path,
    esm_root: Path | None,
    ids: list[str],
    batch_size: int,
    num_workers: int,
    max_residues: int,
    crop_mode: str,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ProteinFeatureDataset(
        features_root=features_root,
        esm_root=esm_root,
        label_root=label_root,
        ids=ids,
        max_residues=max_residues,
        crop_mode=crop_mode,
        positive_crop_prob=0.8,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_protein_features,
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_esm: bool,
    max_batches: int | None = None,
    show_progress: bool = False,
    desc: str = "val",
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metrics: list[dict[str, int]] = []
    skipped_nonfinite = 0
    with torch.no_grad():
        progress_iter = _make_progress(loader, enabled=show_progress, total=len(loader), desc=desc)
        for step, batch in enumerate(progress_iter, 1):
            inputs = _move_batch(batch, device, use_esm)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(**inputs)
            logits = out["logits"]
            if not bool(torch.isfinite(logits).all()):
                skipped_nonfinite += 1
                if hasattr(progress_iter, "set_postfix"):
                    progress_iter.set_postfix(skipped=skipped_nonfinite)
                continue
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(loss)):
                skipped_nonfinite += 1
                if hasattr(progress_iter, "set_postfix"):
                    progress_iter.set_postfix(skipped=skipped_nonfinite)
                continue
            losses.append(float(loss.item()))
            metrics.append(_step_metrics(logits, labels))
            if hasattr(progress_iter, "set_postfix"):
                progress_iter.set_postfix(loss=f"{losses[-1]:.4f}", skipped=skipped_nonfinite)
            if max_batches is not None and step >= max_batches:
                break
    merged = _merge_metrics(metrics)
    merged["loss"] = sum(losses) / max(1, len(losses))
    merged["skipped_nonfinite"] = skipped_nonfinite
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", default="features/protein_v4/pt", type=Path)
    parser.add_argument("--label-root", default="features/contact_labels", type=Path)
    parser.add_argument("--esm-root", default=None, type=Path)
    parser.add_argument("--manifest", default="features/contact_labels/manifest.csv", type=Path)
    parser.add_argument("--split-dir", default="features/contact_labels/splits_seed42", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-scale", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--use-esm", action="store_true")
    parser.add_argument("--d-esm", type=int, default=1280)
    parser.add_argument("--use-rmsnorm", action="store_true")
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument("--freeze-norms", action="store_true")
    parser.add_argument("--safe-grad-clip", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-value-clip", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-foreach", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--nan-probe", action="store_true", help="Log first non-finite module activations/gradients.")
    parser.add_argument("--nan-probe-max-reports", type=int, default=50)
    parser.add_argument("--nan-probe-module-backward-hooks", action="store_true")
    parser.add_argument("--nan-probe-anomaly", action="store_true", help="Enable torch autograd anomaly detection.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress bars.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = _make_or_load_splits(args.manifest, args.split_dir, args.seed)
    train_ids = _limit_ids(train_ids, args.max_train_samples)
    val_ids = _limit_ids(val_ids, args.max_val_samples)
    pos_weight, train_pos, train_neg = _class_weight_from_manifest(args.manifest, train_ids, args.max_pos_weight)

    train_loader = _build_loader(
        args.features_root,
        args.label_root,
        args.esm_root if args.use_esm else None,
        train_ids,
        args.batch_size,
        args.num_workers,
        args.max_residues,
        "positive_window",
        args.seed,
        True,
    )
    val_loader = _build_loader(
        args.features_root,
        args.label_root,
        args.esm_root if args.use_esm else None,
        val_ids,
        args.batch_size,
        args.num_workers,
        args.max_residues,
        "first",
        args.seed + 1,
        False,
    )

    model = ProteinSiteGVP(
        {
            "use_esm": args.use_esm,
            "d_esm": args.d_esm,
            "use_rmsnorm": args.use_rmsnorm,
            "rmsnorm_eps": args.rmsnorm_eps,
            "freeze_norms": args.freeze_norms,
        }
    ).to(device)
    probe = NanProbe(args.nan_probe_max_reports) if args.nan_probe else None
    if probe is not None:
        probe.attach(model, module_backward_hooks=args.nan_probe_module_backward_hooks)
    if args.nan_probe_anomaly:
        torch.autograd.set_detect_anomaly(True, check_nan=True)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device),
        ignore_index=-100,
    )
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
        foreach=args.adam_foreach,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_train_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and args.warmup_ratio > 0:
        warmup_steps = max(1, int(total_train_steps * args.warmup_ratio))
    warmup_steps = min(warmup_steps, total_train_steps)
    min_lr_scale = max(0.0, min(1.0, float(args.min_lr_scale)))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    global_step = 0
    if warmup_steps > 0:
        _set_optimizer_lr(optimizer, base_lrs, min_lr_scale)

    run_config = vars(args).copy()
    run_config.update(
        {
            "device_resolved": str(device),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "train_positive_residues": train_pos,
            "train_negative_residues": train_neg,
            "pos_weight": pos_weight,
            "tqdm_available": tqdm is not None,
            "total_train_steps": total_train_steps,
            "warmup_steps_resolved": warmup_steps,
            "min_lr_scale_resolved": min_lr_scale,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
    log_path = args.output_dir / "metrics.jsonl"
    best_f1 = -1.0

    print(json.dumps({"event": "start", **run_config}, default=str), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_metrics: list[dict[str, int]] = []
        skipped_nonfinite = 0
        t0 = time.time()
        progress_iter = _make_progress(
            train_loader,
            enabled=args.progress,
            total=len(train_loader),
            desc=f"train {epoch}/{args.epochs}",
        )
        for step, batch in enumerate(progress_iter, 1):
            global_step += 1
            if warmup_steps > 0 and global_step <= warmup_steps:
                lr_scale = min_lr_scale + (1.0 - min_lr_scale) * (global_step / warmup_steps)
                _set_optimizer_lr(optimizer, base_lrs, lr_scale)
            elif warmup_steps > 0 and global_step == warmup_steps + 1:
                _set_optimizer_lr(optimizer, base_lrs, 1.0)
            current_lr = float(optimizer.param_groups[0]["lr"])
            inputs = _move_batch(batch, device, args.use_esm)
            labels = batch["labels"].to(device, non_blocking=True)
            if probe is not None:
                probe.set_context(epoch, step, batch)
                probe.check_inputs(batch, inputs, labels, epoch, step)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(**inputs)
                logits = out["logits"]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if hasattr(progress_iter, "set_postfix"):
                    progress_iter.set_postfix(skipped=skipped_nonfinite)
                print(
                    json.dumps(
                        {
                            "event": "skip_nonfinite",
                            "epoch": epoch,
                            "step": step,
                            "pdb_id": batch["pdb_id"],
                            "loss": float(loss.item()) if torch.isfinite(loss) else "nan",
                        }
                    ),
                    flush=True,
                )
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            pre_clip_grad_report = probe.collect_parameter_grads(model) if probe is not None else None
            if args.safe_grad_clip:
                grad_norm, grad_clip_stats = _safe_clip_grad_norm_(
                    trainable_parameters,
                    args.grad_clip_norm,
                    value_clip=args.grad_value_clip,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip_norm)
                grad_clip_stats = {"n_nonfinite_grad_values": 0, "n_tensors_sanitized": 0, "max_abs_grad": None}
            if not bool(torch.isfinite(grad_norm)):
                if probe is not None and pre_clip_grad_report is not None:
                    probe.emit_parameter_grads(pre_clip_grad_report, epoch, step, batch, "pre_clip")
                    probe.report_parameter_grads(model, epoch, step, batch, "post_clip")
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if hasattr(progress_iter, "set_postfix"):
                    progress_iter.set_postfix(skipped=skipped_nonfinite)
                print(
                    json.dumps(
                        {
                            "event": "skip_nonfinite_grad",
                            "epoch": epoch,
                            "step": step,
                            "pdb_id": batch["pdb_id"],
                            "loss": float(loss.item()),
                            "grad_norm": "nan",
                        }
                    ),
                    flush=True,
                )
                continue
            if args.safe_grad_clip and (
                grad_clip_stats["n_nonfinite_grad_values"] > 0 or grad_clip_stats["n_tensors_sanitized"] > 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "safe_grad_sanitized",
                            "epoch": epoch,
                            "step": step,
                            "pdb_id": batch["pdb_id"],
                            "loss": float(loss.item()),
                            "grad_norm": float(grad_norm.item()),
                            **grad_clip_stats,
                        }
                    ),
                    flush=True,
                )
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            epoch_metrics.append(_step_metrics(logits.detach(), labels))
            if step % 25 == 0:
                merged = _merge_metrics(epoch_metrics[-25:])
                window_loss = sum(epoch_losses[-25:]) / len(epoch_losses[-25:])
                if hasattr(progress_iter, "set_postfix"):
                    progress_iter.set_postfix(
                        loss=f"{window_loss:.4f}",
                        f1=f"{merged['f1']:.3f}",
                        acc=f"{merged['accuracy']:.3f}",
                        lr=f"{current_lr:.2e}",
                        skipped=skipped_nonfinite,
                    )
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "lr": current_lr,
                            "loss": window_loss,
                            **merged,
                        }
                    ),
                    flush=True,
                )

        train_metrics = _merge_metrics(epoch_metrics)
        train_metrics["loss"] = sum(epoch_losses) / max(1, len(epoch_losses))
        train_metrics["skipped_nonfinite"] = skipped_nonfinite
        val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.use_esm,
            max_batches=args.eval_max_batches,
            show_progress=args.progress,
            desc=f"val {epoch}/{args.epochs}",
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
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, args.output_dir / "best.pt")

    print(json.dumps({"event": "done", "best_val_f1": best_f1, "output_dir": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
