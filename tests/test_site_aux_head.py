import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/training"), str(ROOT / "src")]
import train_esm_site as trainer


def config(head=False, detach=True):
    cfg = dict(d_esm=6, d_seq=4, d_model=16, d_hidden=32, use_contact_graph=False,
               dropout=.25, classifier_dropout=.45, use_tcn_context=True, tcn_layers=2,
               tcn_kernel_size=7, tcn_dilations=[1, 2], tcn_block_type="gated_depthwise")
    cfg.update(d_esm=6, d_seq=4, d_model=16, d_hidden=32, use_contact_graph=False,
               use_chain_ratio_head=head, chain_ratio_head_hidden=8,
               chain_ratio_head_dropout=0., chain_ratio_head_detach=detach,
               chain_ratio_head_init_ratio=.3, chain_ratio_head_preserve_rng=True)
    return cfg


def criterion(aux=.75, mass=.75):
    return trainer.ResidueSiteLoss(1., 0., "ce", 0., 2., None, "chain_mean", mass,
                                   torch.device("cpu"), "ratio_mse", aux, "residue_mean")


def inputs():
    torch.manual_seed(17)
    return dict(esm_embeddings=torch.randn(2, 5, 6), seq_features=torch.randn(2, 5, 4),
                protein_mask=torch.tensor([[True] * 5, [True, True, True, False, False]]),
                chain_ids=torch.zeros(2, 5, dtype=torch.long),
                chain_rel_pos=torch.tensor([[0., .25, .5, .75, 1.], [0., .5, 1., 0., 0.]]),
                protein_rel_pos=torch.tensor([[0., .25, .5, .75, 1.], [0., .5, 1., 0., 0.]]))


def parameters(model):
    main = [p for n, p in model.named_parameters() if not n.startswith("chain_ratio_head.")]
    head = [p for n, p in model.named_parameters() if n.startswith("chain_ratio_head.")]
    return main, head


class AuxTests(unittest.TestCase):
    def test_residue_mass_preserved_and_aux_added(self):
        x = torch.tensor([[[0., -1.], [.2, .4], [3., -4.]], [[1., 0.], [0., 1.], [0., 1.]]], requires_grad=True)
        y = torch.tensor([[1, 0, -100], [0, 1, 1]])
        raw_head = torch.tensor([-2., 2.], requires_grad=True)
        base_crit = criterion(aux=0)
        expected = base_crit(x, y)
        expected_grad = torch.autograd.grad(expected, x)[0]
        aux_crit = criterion()
        actual = aux_crit(x, y, raw_head)
        r = torch.tensor([.5, 2 / 3])
        expected_aux = ((raw_head.sigmoid() - r) ** 2).mean()
        self.assertAlmostEqual(actual.item(), (expected + .75 * expected_aux).item(), places=7)
        self.assertTrue(torch.equal(expected_grad, torch.autograd.grad(actual, x, retain_graph=True)[0]))
        self.assertEqual(base_crit.last_components["mass_loss_unweighted"].item(),
                         aux_crit.last_components["mass_loss_unweighted"].item())
        self.assertGreater(torch.autograd.grad(actual, raw_head)[0].abs().sum().item(), 0)

    def test_missing_head_masks_and_empty_chains(self):
        x = torch.randn(3, 4, 2, requires_grad=True)
        y = torch.tensor([[0, 1, -100, -100], [1, 1, 1, 1], [-100] * 4])
        head = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "requires a ratio head"):
            criterion()(x, y)
        c = criterion()
        value = c(x, y, head)
        value.backward()
        self.assertEqual(head.grad[2].item(), 0)
        self.assertTrue(torch.equal(x.grad[0, 2:], torch.zeros_like(x.grad[0, 2:])))
        self.assertTrue(torch.isfinite(head.grad).all())
        empty_x = torch.randn(2, 3, 2, requires_grad=True)
        empty_head = torch.randn(2, requires_grad=True)
        empty = c(empty_x, torch.full((2, 3), -100), empty_head)
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertIsNone(empty_head.grad)

    def test_head_initialization_preserves_backbone_and_rng(self):
        torch.manual_seed(49)
        baseline = trainer.ESMSiteClassifier(config())
        rng = torch.get_rng_state()
        torch.manual_seed(49)
        added = trainer.ESMSiteClassifier(config(head=True))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, added.state_dict()[name]), name)

    def test_detached_and_joint_gradient_routing(self):
        x = inputs()
        for detached in (True, False):
            torch.manual_seed(49)
            model = trainer.ESMSiteClassifier(config(head=True, detach=detached))
            # The real output layer starts at zero; inspect routing after it becomes nonzero.
            with torch.no_grad():
                model.chain_ratio_head[-1].weight.fill_(.1)
            out = model(**x)
            loss = (out["chain_ratio_logits"].sigmoid() - torch.tensor([.2, .7])).square().mean()
            loss.backward()
            main, head = parameters(model)
            main_norm = sum(p.grad.abs().sum().item() for p in main if p.grad is not None)
            head_norm = sum(p.grad.abs().sum().item() for p in head if p.grad is not None)
            self.assertGreater(head_norm, 0)
            if detached:
                self.assertEqual(main_norm, 0)
            else:
                self.assertGreater(main_norm, 0)

    def test_detached_control_backbone_updates_match(self):
        x = inputs()
        y = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, -100, -100]])
        torch.manual_seed(49)
        baseline = trainer.ESMSiteClassifier(config())
        torch.manual_seed(49)
        detached = trainer.ESMSiteClassifier(config(head=True, detach=True))
        pairs = [parameters(baseline), parameters(detached)]
        opts = [torch.optim.AdamW([{"params": main}] + ([{"params": head}] if head else []), lr=.003)
                for main, head in pairs]
        criteria = [criterion(aux=0), criterion()]
        for step in range(4):
            rng = torch.get_rng_state()
            outputs = []
            for model, opt, c, (main, head) in zip((baseline, detached), opts, criteria, pairs):
                torch.set_rng_state(rng)
                model.train()
                opt.zero_grad(set_to_none=True)
                out = model(**x)
                outputs.append(out["logits"].detach())
                loss = c(out["logits"], y, out["chain_ratio_logits"])
                loss.backward()
                trainer._safe_clip(main, .05, 100.)
                trainer._safe_clip(head, .05, 100.)
                opt.step()
            self.assertTrue(torch.equal(*outputs), f"logits at step {step}")
            for name, value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value, detached.state_dict()[name]), f"{name} step {step}")

    def test_padding_does_not_change_head_prediction(self):
        torch.manual_seed(49)
        model = trainer.ESMSiteClassifier(config(head=True)).eval()
        batch = inputs()
        with torch.no_grad():
            model.chain_ratio_head[-1].weight.fill_(.1)
            a = model(**batch)["chain_ratio_logits"]
            batch["esm_embeddings"][1, 3:] = 10000.
            batch["seq_features"][1, 3:] = -10000.
            b = model(**batch)["chain_ratio_logits"]
        self.assertTrue(torch.equal(a, b))

    def test_legacy_head_rng_default_is_unchanged(self):
        cfg = config(head=True)
        del cfg["chain_ratio_head_preserve_rng"]
        torch.manual_seed(49)
        trainer.ESMSiteClassifier(config())
        no_head_rng = torch.get_rng_state()
        torch.manual_seed(49)
        trainer.ESMSiteClassifier(cfg)
        self.assertFalse(torch.equal(no_head_rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
