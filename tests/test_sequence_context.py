from pathlib import Path
import copy
import importlib.util
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppcbind.models.sequence_context import PostFusionSequenceContext
from ppcbind.models.esm_site import ESMSiteClassifier


def model_config(**extra):
    cfg = dict(d_esm=6, d_seq=4, d_model=16, d_hidden=32, dropout=0., classifier_dropout=0.,
               use_contact_graph=False, use_tcn_context=True, tcn_layers=1, tcn_kernel_size=3,
               tcn_dilations=[1], use_prottrans_embeddings=True, d_prottrans=8,
               prottrans_fusion_mode="zero_init_pre_contact_residual", prottrans_residual_dim=4,
               prottrans_residual_hidden=4, prottrans_residual_dropout=0.)
    return {**cfg, **extra}


def inputs():
    return dict(esm_embeddings=torch.randn(2, 5, 6), seq_features=torch.randn(2, 5, 4),
                prottrans_embeddings=torch.randn(2, 5, 8),
                protein_mask=torch.tensor([[True] * 5, [True, True, True, False, False]]),
                chain_ids=torch.zeros(2, 5, dtype=torch.long),
                chain_rel_pos=torch.linspace(0, 1, 5).repeat(2, 1),
                protein_rel_pos=torch.linspace(0, 1, 5).repeat(2, 1))


class SequenceContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(49)

    def test_packed_gru_padding_and_original_order(self):
        block = PostFusionSequenceContext(16, "bigru", 8, 0.).eval()
        h = torch.randn(3, 7, 16)
        lens = [4, 7, 2]
        mask = torch.arange(7)[None] < torch.tensor(lens)[:, None]
        out = block(h, mask)
        changed = h.clone()
        changed[~mask] = 100000.
        torch.testing.assert_close(out, block(changed, mask))
        for i, n in enumerate(lens):
            single = block(h[i:i + 1, :n], torch.ones(1, n, dtype=torch.bool))
            torch.testing.assert_close(out[i:i + 1, :n], single)
        self.assertTrue(torch.equal(out[~mask], torch.zeros_like(out[~mask])))

    def test_invalid_masks_and_chains_rejected(self):
        block = PostFusionSequenceContext(16, "bigru", 8)
        h = torch.randn(1, 3, 16)
        for mask in (torch.tensor([[1, 0, 1]]), torch.zeros(1, 3)):
            with self.assertRaises(ValueError):
                block(h, mask)
        with self.assertRaises(ValueError):
            block(h, torch.ones(1, 3), torch.tensor([[0, 1, 1]]))

    def test_parameter_match(self):
        gru = PostFusionSequenceContext(256, "bigru", 128)
        mlp = PostFusionSequenceContext(256, "mlp", 576)
        self.assertEqual(sum(p.numel() for p in gru.parameters()), 296960)
        self.assertEqual(sum(p.numel() for p in mlp.parameters()), 296256)

    def test_backbone_initialization_and_rng_preserved(self):
        control = ESMSiteClassifier(model_config())
        rng = torch.get_rng_state()
        for mode, hidden in (("bigru", 8), ("mlp", 36)):
            torch.manual_seed(49)
            model = ESMSiteClassifier(model_config(post_fusion_context_mode=mode, post_fusion_context_hidden=hidden))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for name, value in control.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_msa_position_reaches_tcn_only_when_pre(self):
        x = inputs()
        for position in ("pre_tcn", "post_tcn"):
            model = ESMSiteClassifier(model_config(msa_residual_position=position)).eval()
            with torch.no_grad():
                model.prottrans_residual_adapter[-1].weight.normal_(std=.1)
            seen = []
            hook = model.tcn_blocks[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
            model(**x)
            changed = {**x, "prottrans_embeddings": torch.randn_like(x["prottrans_embeddings"])}
            model(**changed)
            hook.remove()
            self.assertEqual(torch.equal(*seen), position == "post_tcn")

    def test_both_contexts_get_gradients(self):
        x = inputs()
        for mode, hidden in (("bigru", 8), ("mlp", 36)):
            model = ESMSiteClassifier(model_config(post_fusion_context_mode=mode, post_fusion_context_hidden=hidden))
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            for _ in range(2):
                opt.zero_grad()
                out = model(**x)["logits"]
                out[x["protein_mask"]].square().mean().backward()
                grads = [p.grad for p in model.post_fusion_context.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                opt.step()

    def test_config_matrix_keeps_features_and_loss_matched(self):
        spec = importlib.util.spec_from_file_location("context_configs", ROOT / "scripts/preparations/m18_union/prepare_msa_context_screen.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parent = dict(model=dict(d_seq=90, d_prottrans=768), data=dict(require_prottrans_embeddings=True),
                      training=dict(mass_loss_type="ratio_mse", ratio_loss_weight=.75), metadata={})
        original = copy.deepcopy(parent)
        for name in module.VARIANTS:
            cfg = module.build(parent, Path("/tmp/context_test"), name)
            self.assertEqual(cfg["data"], parent["data"])
            self.assertEqual(cfg["training"]["ratio_loss_weight"], .75)
            self.assertEqual(cfg["training"]["seed"], 49)
            self.assertEqual(cfg["training"]["epochs"], 50)
            self.assertTrue(cfg["training"]["validation_only"])
            self.assertFalse(cfg["training"]["eval_test_each_epoch"])
            self.assertIsNone(cfg["training"]["init_checkpoint"])
        self.assertEqual(parent, original)


if __name__ == "__main__":
    unittest.main()
