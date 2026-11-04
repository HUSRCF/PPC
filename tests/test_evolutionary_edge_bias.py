import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppcbind.models.geometry import EvolutionaryEdgeBias, GeometryGraphBlock
from ppcbind.models.esm_site import ESMSiteClassifier


def config(**changes):
    return dict(d_esm=6, d_seq=4, d_model=16, d_hidden=32, n_heads=4, geometry_heads=4,
                d_prottrans=8, prottrans_residual_dim=4, prottrans_residual_hidden=4,
                prottrans_fusion_mode="zero_init_pre_contact_residual", use_prottrans_embeddings=True,
                prottrans_residual_dropout=0., geometry_mode="edge_attention",
                dropout=0., classifier_dropout=0., **changes)


def payload():
    edges = torch.tensor([[1, 2, 0, 2, 0, 1, 5, 6, 4, 6, 4, 5],
                          [0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6]])
    return dict(
        esm_embeddings=torch.randn(2, 4, 6), seq_features=torch.randn(2, 4, 4),
        prottrans_embeddings=torch.randn(2, 4, 8),
        protein_mask=torch.tensor([[True] * 4, [True] * 3 + [False]]),
        geometry_vectors=torch.randn(2, 4, 8, 3),
        geometry_edge_index=edges, geometry_edge_features=torch.randn(12, 30),
        geometry_edge_vectors=torch.randn(12, 1, 3), geometry_edge_scores=torch.rand(12) * .5 + .5,
    )


class EvolutionaryEdgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(49)

    def test_zero_start_bounded_and_padding_invariance(self):
        layer = EvolutionaryEdgeBias(4, heads=4)
        x = payload()
        features = torch.randn(2, 4, 4)
        zero = layer(features, x["protein_mask"], x["geometry_edge_index"])
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        with torch.no_grad():
            layer.scale.fill_(10.)
        first = layer(features, x["protein_mask"], x["geometry_edge_index"])
        changed = features.clone()
        changed[~x["protein_mask"]] = 1.e4
        torch.testing.assert_close(first, layer(changed, x["protein_mask"], x["geometry_edge_index"]))
        self.assertLessEqual(first.abs().max().item(), 4.00001)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed = layer(features, x["protein_mask"], x["geometry_edge_index"])
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertEqual(mixed.dtype, torch.float32)

    def test_initial_backbone_rng_and_nonzero_geometry_output_unchanged(self):
        base = ESMSiteClassifier(config())
        rng = torch.get_rng_state()
        torch.manual_seed(49)
        evolved = ESMSiteClassifier(config(geometry_msa_edge_bias=True))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, evolved.state_dict()[name]), name)
        update = torch.randn_like(base.geometry_blocks[0].update[-1].weight) * .1
        with torch.no_grad():
            for model in (base, evolved):
                model.geometry_blocks[0].update[-1].weight.copy_(update)
        x = payload()
        base.eval()
        evolved.eval()
        self.assertTrue(torch.equal(base(**x)["logits"], evolved(**x)["logits"]))

    def test_evolutionary_features_change_attention_without_node_adapter(self):
        model = ESMSiteClassifier(config(geometry_msa_edge_bias=True)).eval()
        with torch.no_grad():
            model.geometry_blocks[0].evolution_bias.scale.fill_(.7)
            model.geometry_blocks[0].update[-1].weight.normal_(std=.2)
        x = payload()
        first = model(**x)["logits"]
        second = model(**{**x, "prottrans_embeddings": torch.randn_like(x["prottrans_embeddings"])})["logits"]
        self.assertFalse(torch.equal(first[x["protein_mask"]], second[x["protein_mask"]]))
        self.assertTrue(torch.equal(first[~x["protein_mask"]], torch.zeros_like(first[~x["protein_mask"]])))

    def test_three_steps_train_scale_and_projections_without_decay(self):
        model = ESMSiteClassifier(config(geometry_msa_edge_bias=True, geometry_layers=2))
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=0.)
        x = payload()
        for _ in range(3):
            optimizer.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(**x)["logits"]
                loss = logits[x["protein_mask"]].square().mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        for i, block in enumerate(model.geometry_blocks):
            for suffix in ("scale", "query.weight", "key.weight"):
                key = f"geometry_blocks.{i}.evolution_bias.{suffix}"
                self.assertFalse(torch.equal(initial[key], model.state_dict()[key]), key)
            self.assertGreater(block.evolution_bias.query.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.evolution_bias.key.weight.grad.abs().sum().item(), 0)

    def test_no_cross_sample_edges_and_missing_features_rejected(self):
        x = payload()
        model = ESMSiteClassifier(config(geometry_msa_edge_bias=True))
        with self.assertRaises(ValueError):
            model(**{**x, "prottrans_embeddings": None})
        for edges in (torch.tensor([[-1], [0]]), torch.tensor([[0], [4]])):
            bad = {**x, "geometry_edge_index": edges, "geometry_edge_features": x["geometry_edge_features"][:1],
                   "geometry_edge_vectors": x["geometry_edge_vectors"][:1], "geometry_edge_scores": x["geometry_edge_scores"][:1]}
            with self.assertRaises(ValueError):
                model(**bad)
        with self.assertRaises(ValueError):
            ESMSiteClassifier({**config(), "geometry_mode": "edge_mlp", "geometry_msa_edge_bias": True})
        with self.assertRaises(ValueError):
            ESMSiteClassifier({**config(), "geometry_msa_edge_bias": True, "prottrans_residual_dim": 0})

    def test_production_parameter_increment(self):
        block = GeometryGraphBlock(256, 30, "edge_attention", evolution_dim=64)
        self.assertEqual(sum(p.numel() for p in block.evolution_bias.parameters()), 8328)


if __name__ == "__main__":
    unittest.main()
