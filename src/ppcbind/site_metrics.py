"""Residue-site validation diagnostics and explicit epoch-loss aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


class LossAccumulator:
    """Chain-weighted epoch losses, not an unweighted mean of batch means."""

    keys = ("classification_loss", "mass_loss_unweighted", "mass_loss_weighted",
            "aux_ratio_loss_unweighted", "aux_ratio_loss_weighted", "total_loss")

    def __init__(self):
        self.sums = np.zeros(len(self.keys), dtype=np.float64)
        self.n_chains = 0
        self.n_residues = 0
        self.n_batches = 0
        self.loss_denominator = 0
        self.reduction = None

    def update(self, criterion, labels):
        valid = labels != -100
        n_chains = int(valid.any(dim=1).sum().item())
        n_residues = int(valid.sum().item())
        values = torch.stack([criterion.last_components[k] for k in self.keys]).double().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite loss components")
        reduction = criterion.reduction
        if self.reduction is not None and self.reduction != reduction:
            raise ValueError("Loss reduction changed inside an epoch")
        self.reduction = reduction
        weight = n_chains if reduction == "chain_mean" else 1
        self.sums += values * weight
        self.loss_denominator += weight
        self.n_chains += n_chains
        self.n_residues += n_residues
        self.n_batches += 1

    def metrics(self):
        keys = self.keys
        result = dict(zip(keys, (self.sums / max(1, self.loss_denominator)).tolist()))
        result.update(loss_component_numerators=dict(zip(keys, self.sums.tolist())),
                      loss_chain_denominator=self.n_chains if self.reduction == "chain_mean" else None,
                      loss_component_denominator=self.loss_denominator, valid_chains=self.n_chains,
                      valid_residues=self.n_residues, successful_batches=self.n_batches,
                      loss_aggregation="valid-chain-weighted" if self.reduction == "chain_mean" else "batch-mean (legacy residue_mean)")
        result["loss"] = result["total_loss"]
        return result


def _f1_mcc(tp, fp, tn, fn):
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
                   where=(2 * tp + fp + fn) > 0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return f1, mcc


def extended_metrics(per_protein, thresholds):
    """Macro confusion metrics use a single shared validation gate per metric."""
    grid = np.asarray(sorted(set([0.5, *map(float, thresholds)])), dtype=np.float64)
    f1s, mccs, briers, nlls, true_rates, pred_rates, lengths = [], [], [], [], [], [], []
    for _, scores, targets in per_protein:
        p = scores.double().numpy()
        y = targets.double().numpy()
        if not len(y):
            continue
        order = np.argsort(p, kind="stable")
        prefix = np.concatenate(([0.0], np.cumsum(y[order])))
        cut = np.searchsorted(p[order], grid, side="left")
        tp = prefix[-1] - prefix[cut]
        fp = len(y) - cut - tp
        fn = prefix[cut]
        tn = cut - fn
        f1, mcc = _f1_mcc(tp, fp, tn, fn)
        f1s.append(f1)
        mccs.append(mcc)
        briers.append(float(np.mean((p - y) ** 2)))
        safe_p = np.clip(p, 1e-7, 1 - 1e-7)
        nlls.append(float(np.mean(-y * np.log(safe_p) - (1 - y) * np.log1p(-safe_p))))
        true_rates.append(float(y.mean()))
        pred_rates.append(float(p.mean()))
        lengths.append(len(y))
    if not lengths:
        return {}
    macro_f1, macro_mcc = np.mean(f1s, axis=0), np.mean(mccs, axis=0)
    at05 = int(np.flatnonzero(grid == 0.5)[0])
    best_f1, best_mcc = int(np.argmax(macro_f1)), int(np.argmax(macro_mcc))
    result = {
        "chain_f1_at_0_5": float(macro_f1[at05]),
        "chain_mcc_at_0_5": float(macro_mcc[at05]),
        "chain_f1_best_threshold": float(macro_f1[best_f1]),
        "chain_best_f1_threshold": float(grid[best_f1]),
        "chain_mcc_at_best_f1_threshold": float(macro_mcc[best_f1]),
        "chain_mcc_best_threshold": float(macro_mcc[best_mcc]),
        "chain_best_mcc_threshold": float(grid[best_mcc]),
        "chain_f1_at_best_mcc_threshold": float(macro_f1[best_mcc]),
        "chain_threshold_sweep": [dict(threshold=float(t), f1=float(f), mcc=float(m))
                                  for t, f, m in zip(grid, macro_f1, macro_mcc)],
        "chain_brier_macro": float(np.mean(briers)),
        "neg_chain_brier_macro": -float(np.mean(briers)),
        "pooled_brier": float(np.average(briers, weights=lengths)),
        "chain_nll_macro": float(np.mean(nlls)),
        "pooled_nll": float(np.average(nlls, weights=lengths)),
        "nll_probability_clip": 1e-7,
        "chain_ratio_mean_observed": float(np.mean(true_rates)),
        "chain_ratio_mean_predicted": float(np.mean(pred_rates)),
        "probability_metric_n_chains": len(lengths),
        "probability_metric_n_residues": int(sum(lengths)),
    }
    nonconstant = len(lengths) > 1 and np.std(true_rates) > 0 and np.std(pred_rates) > 0
    result["chain_ratio_pearson"] = float(pearsonr(true_rates, pred_rates).statistic) if nonconstant else None
    result["chain_ratio_spearman"] = float(spearmanr(true_rates, pred_rates).statistic) if nonconstant else None
    return result


def update_pareto(output_dir: Path, epoch: int, metrics: dict):
    """Record nondominated validation candidates; this is not final model selection."""
    path = output_dir / "pareto_candidates.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    row = {"epoch": epoch, "checkpoint": f"epochs/epoch_{epoch:03d}.pt",
           "macro_ap": metrics["chain_ap_macro"],
           "macro_mcc_at_0_5": metrics["chain_mcc_at_0_5"],
           "ratio_mae": metrics["chain_ratio_mae_raw"], "brier": metrics["chain_brier_macro"]}
    if not all(math.isfinite(row[k]) for k in ("macro_ap", "macro_mcc_at_0_5", "ratio_mae", "brier")):
        raise ValueError("Non-finite Pareto candidate")
    rows.append(row)
    def dominates(a, b):
        av = [a["macro_ap"], a["macro_mcc_at_0_5"], -a["ratio_mae"], -a["brier"]]
        bv = [b["macro_ap"], b["macro_mcc_at_0_5"], -b["ratio_mae"], -b["brier"]]
        return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))
    rows = [r for r in rows if not any(dominates(other, r) for other in rows)]
    path.write_text(json.dumps(rows, indent=2) + "\n")
