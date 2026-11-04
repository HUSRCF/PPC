import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score, log_loss, brier_score_loss, auc, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/training"), str(ROOT / "src")]
import train_esm_site as trainer
from ppcbind.site_metrics import LossAccumulator, extended_metrics, update_pareto


def criterion(weight=3.015931504372524, smooth=0.05, mass_weight=0.75, kind="ratio_mse"):
    return trainer.ResidueSiteLoss(weight, smooth, "ce", 0, 2, None, "chain_mean", mass_weight,
                                   torch.device("cpu"), kind)


class MatrixTests(unittest.TestCase):
    def test_legacy_mse_value_and_gradient(self):
        y = torch.tensor([[0, 1, 0, -100], [1, 1, -100, -100], [-100] * 4])
        torch.manual_seed(49)
        for weight in (1, 3.015931504372524):
            for smooth in (0, .05):
                for mass_weight in (.75, 3):
                    x = torch.randn(3, 4, 2, requires_grad=True)
                    valid = y != -100
                    weights = torch.tensor([1., weight])
                    tokens = torch.nn.functional.cross_entropy(x.reshape(-1, 2), y.reshape(-1),
                        weight=weights, ignore_index=-100, label_smoothing=smooth, reduction="none").reshape_as(y)
                    denominator = (weights[y.masked_fill(~valid, 0)] * valid).sum(1)
                    known = denominator > 0
                    ce = ((tokens.sum(1) / denominator.clamp_min(1e-12)) * known).sum() / known.sum()
                    p = (x.softmax(-1)[..., 1] * valid).sum(1) / valid.sum(1).clamp_min(1)
                    r = ((y == 1) & valid).sum(1).float() / valid.sum(1).clamp_min(1)
                    old = ce + mass_weight * torch.nn.functional.mse_loss(p[known], r[known])
                    old_grad = torch.autograd.grad(old, x)[0]
                    new = criterion(weight, smooth, mass_weight)(x, y)
                    new_grad = torch.autograd.grad(new, x)[0]
                    self.assertTrue(torch.equal(old, new))
                    self.assertTrue(torch.equal(old_grad, new_grad))

    def test_rate_kl_limits_mask_and_gradients(self):
        for target in (0, 1):
            for score in (-100., 0., 100.):
                x = torch.tensor([[[0., score], [0., score], [5., -8.]]], requires_grad=True)
                y = torch.tensor([[target, target, -100]])
                crit = criterion(weight=1, smooth=0, mass_weight=1, kind="bernoulli_rate_kl")
                value = crit(x, y)
                logp = torch.log_softmax(x[0, :2], -1)[:, target].mean()
                self.assertAlmostEqual(crit.last_components["mass_loss_unweighted"].item(), -logp.item(), places=5)
                value.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.equal(x.grad[0, 2], torch.zeros(2)))
                if (target == 0 and score > 0) or (target == 1 and score < 0):
                    self.assertGreater(x.grad[0, :2].abs().max().item(), 0)
        x = torch.randn(2, 3, 2, requires_grad=True)
        empty = criterion(kind="bernoulli_rate_kl")(x, torch.full((2, 3), -100))
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_length_normalization_and_mixed_kl(self):
        x = torch.tensor([[[0., -.5], [0., 1.], [1., 0.]]], requires_grad=True)
        y = torch.tensor([[1, 0, 1]])
        crit = criterion(weight=1, smooth=0, mass_weight=1, kind="bernoulli_rate_kl")
        value = crit(x, y)
        r = 2 / 3
        p = x.softmax(-1)[..., 1].mean().item()
        expected = r * math.log(r / p) + (1 - r) * math.log((1 - r) / (1 - p))
        self.assertAlmostEqual(crit.last_components["mass_loss_unweighted"].item(), expected, places=6)
        duplicated = crit(x.repeat(1, 2, 1), y.repeat(1, 2))
        self.assertAlmostEqual(value.item(), duplicated.item(), places=6)

    def test_literal_half_and_extended_metrics(self):
        p = torch.tensor([.2, .55, .8, .9])
        y = torch.tensor([0, 1, 0, 1])
        m = trainer._threshold_metrics(p, y, [.72])
        self.assertEqual(m["threshold_0_5"], .5)
        self.assertAlmostEqual(m["f1_at_0_5"], f1_score(y, p >= .5))
        rows = [("a", p, y), ("b", p[:2], y[:2]), ("c", p[:1], y[:1])]
        m = extended_metrics(rows, [.3, .72])
        self.assertAlmostEqual(m["chain_f1_at_0_5"], np.mean([f1_score(t, s >= .5, zero_division=0) for _, s, t in rows]))
        self.assertAlmostEqual(m["chain_mcc_at_0_5"], np.mean([matthews_corrcoef(t, s >= .5) for _, s, t in rows]))
        self.assertAlmostEqual(m["pooled_brier"], brier_score_loss(torch.cat([r[2] for r in rows]), torch.cat([r[1] for r in rows])))
        expected_nll = np.mean([log_loss(t.double().numpy(), s.double().numpy(), labels=[0, 1]) for _, s, t in rows])
        self.assertAlmostEqual(m["chain_nll_macro"], expected_nll)
        ranking = trainer._auc_metrics(p, y)
        self.assertAlmostEqual(ranking["pooled_ap"], average_precision_score(y, p))
        precision, recall, _ = precision_recall_curve(y, p)
        self.assertAlmostEqual(ranking["trapezoid_auprc"], auc(recall, precision))
        for row in m["chain_threshold_sweep"]:
            gate = row["threshold"]
            self.assertAlmostEqual(row["f1"], np.mean([f1_score(t, s >= gate, zero_division=0) for _, s, t in rows]))
            self.assertAlmostEqual(row["mcc"], np.mean([matthews_corrcoef(t, s >= gate) for _, s, t in rows]))

    def test_batch_partition_invariance(self):
        torch.manual_seed(49)
        x = torch.randn(3, 5, 2)
        y = torch.tensor([[1, 0, 0, -100, -100], [0, 1, 1, 0, 1], [0, -100, -100, -100, -100]])
        for kind in ("ratio_mse", "bernoulli_rate_kl"):
            full, chunks = LossAccumulator(), LossAccumulator()
            crit = criterion(kind=kind)
            crit(x, y)
            full.update(crit, y)
            for sl in (slice(0, 2), slice(2, 3)):
                crit(x[sl], y[sl])
                chunks.update(crit, y[sl])
            self.assertAlmostEqual(full.metrics()["loss"], chunks.metrics()["loss"], places=6)
            self.assertEqual(chunks.metrics()["valid_chains"], 3)
            self.assertEqual(chunks.metrics()["valid_residues"], 9)

    def test_pareto(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for epoch, ap, mae in ((1, .7, .1), (2, .8, .09), (3, .75, .05)):
                update_pareto(p, epoch, dict(chain_ap_macro=ap, chain_mcc_at_0_5=.5,
                                             chain_ratio_mae_raw=mae, chain_brier_macro=.2))
            self.assertEqual([r["epoch"] for r in json.loads((p / "pareto_candidates.json").read_text())], [2, 3])

    def test_legacy_head_source_is_preserved_by_default(self):
        x = torch.tensor([[[0., 1.], [1., 0.]]], requires_grad=True)
        y = torch.tensor([[0, 0]])
        head = torch.tensor([.3], requires_grad=True)
        c = criterion(weight=1, smooth=0)
        got = c(x, y, head)
        expected = torch.nn.functional.cross_entropy(x.reshape(-1, 2), y.reshape(-1)) + .75 * head.sigmoid().square().mean()
        self.assertTrue(torch.equal(got, expected))
        actual_grad = torch.autograd.grad(got, (x, head), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (x, head))
        for a, b in zip(actual_grad, expected_grad):
            self.assertTrue(torch.equal(a, b))

    def test_legacy_residue_mean_loss_accounting(self):
        c = trainer.ResidueSiteLoss(2., .05, "ce", 0, 2, None, "residue_mean", .75, torch.device("cpu"))
        meter = LossAccumulator()
        losses = []
        for batch_size in (3, 1):
            x = torch.randn(batch_size, 4, 2)
            y = torch.tensor([[0, 1, 0, 1]] * batch_size)
            losses.append(c(x, y).item())
            meter.update(c, y)
        self.assertAlmostEqual(meter.metrics()["loss"], sum(losses) / 2)
        self.assertIsNone(meter.metrics()["loss_chain_denominator"])
        self.assertEqual(meter.metrics()["loss_component_denominator"], 2)

    def test_all_unknown_legacy_head_stays_unused(self):
        x = torch.randn(1, 3, 2, requires_grad=True)
        head = torch.nn.Parameter(torch.tensor([.3]))
        before = head.detach().clone()
        optimizer = torch.optim.AdamW([head], lr=.1, weight_decay=.01)
        criterion()(x, torch.full((1, 3), -100), head).backward()
        self.assertIsNone(head.grad)
        optimizer.step()
        self.assertTrue(torch.equal(before, head))


if __name__ == "__main__":
    unittest.main()
