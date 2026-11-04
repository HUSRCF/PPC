import sys
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppcbind.models.geometry import GeometryGraphBlock, LinearAttentionContext
from ppcbind.data.geometry_features import crop_geometry
from ppcbind.data.esm_site_dataset import _collate_contact_edges
from ppcbind.data.esm_site_dataset import ESMProteinSiteDataset, collate_esm_site_features
from ppcbind.models.esm_site import ESMSiteClassifier


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(49)
        self.h = torch.randn(2, 4, 16)
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        self.v = torch.randn(2, 4, 8, 3)
        self.e = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
        self.a = torch.randn(6, 30)
        self.ev = torch.randn(6, 1, 3)
        self.s = torch.rand(6)

    def test_modes_update_and_padding(self):
        for mode in ("edge_mlp", "edge_attention", "gvp"):
            block = GeometryGraphBlock(16, 30, mode, 0., heads=4)
            opt = torch.optim.AdamW(block.parameters(), lr=.01)
            for _ in range(2):
                opt.zero_grad()
                h, v = block(self.h, self.v, self.mask, self.e, self.a, self.ev, self.s)
                h.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None))
                opt.step()
            self.assertGreater(block.update[-1].weight.abs().sum().item(), 0)
            self.assertTrue(torch.equal(h[1, 2:], torch.zeros_like(h[1, 2:])))
            self.assertTrue(torch.isfinite(v).all())
            if mode == "gvp":
                self.assertGreater(block.message.scalar.weight.grad.abs().sum().item(), 0)
                self.assertGreater(block.message.vector_out.weight.grad.abs().sum().item(), 0)
                self.assertGreater(block.message.gate.weight.grad.abs().sum().item(), 0)

    def test_gvp_scalar_invariance_vector_covariance(self):
        block = GeometryGraphBlock(16, 30, "gvp", 0., heads=4).eval()
        torch.nn.init.normal_(block.update[-1].weight, std=.05)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        h, v = block(self.h, self.v, self.mask, self.e, self.a, self.ev, self.s)
        hr, vr = block(self.h, self.v @ rotation, self.mask, self.e, self.a, self.ev @ rotation, self.s)
        torch.testing.assert_close(h, hr, atol=1.e-5, rtol=1.e-5)
        torch.testing.assert_close(v @ rotation, vr, atol=1.e-5, rtol=1.e-5)

    def test_invalid_edges_rejected(self):
        block = GeometryGraphBlock(16, 30, "edge_mlp", heads=4)
        for edge in (torch.tensor([[0], [5]]), torch.tensor([[4], [6]]), torch.tensor([[-1], [0]])):
            with self.assertRaises(ValueError):
                block(self.h, self.v, self.mask, edge, self.a[:1], self.ev[:1], self.s[:1])

    def test_no_edges(self):
        block = GeometryGraphBlock(16, 30, "gvp", heads=4)
        h, v = block(self.h, self.v, self.mask, self.e[:, :0], self.a[:0], self.ev[:0], self.s[:0])
        torch.testing.assert_close(h[self.mask], self.h[self.mask])
        self.assertEqual(v.shape, (2, 4, 16, 3))

    def test_isolated_node_has_no_batch_dependence(self):
        mask = self.mask.clone()
        mask[1] = torch.tensor([True, False, False, False])
        for mode in ("edge_mlp", "edge_attention", "gvp"):
            block = GeometryGraphBlock(16, 30, mode, 0., heads=4).eval()
            torch.nn.init.normal_(block.update[-1].weight, std=.1)
            torch.nn.init.normal_(block.update[-1].bias, std=.1)
            h, v = block(self.h, self.v, mask, self.e[:, :4], self.a[:4], self.ev[:4], self.s[:4])
            hs, vs = block(self.h[1:, :1], self.v[1:, :1], mask[1:, :1],
                           self.e[:, :0], self.a[:0], self.ev[:0], self.s[:0])
            torch.testing.assert_close(h[1:, :1], hs)
            torch.testing.assert_close(v[1:, :1], vs)

    def test_linear_attention_padding(self):
        block = LinearAttentionContext(16, heads=4, dropout=0.).eval()
        torch.nn.init.normal_(block.output.weight, std=.1)
        torch.nn.init.normal_(block.ff[-1].weight, std=.1)
        first = block(self.h, self.mask)
        changed = self.h.clone()
        changed[~self.mask] = 10000.
        second = block(changed, self.mask)
        torch.testing.assert_close(first, second)
        first.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None))

    def test_linear_attention_long_accumulation_float16(self):
        block = LinearAttentionContext(32, heads=1, dropout=0.).eval()
        with torch.no_grad():
            block.qkv.weight.zero_()
            block.qkv.bias.fill_(1.)
            block.output.weight.normal_(std=.02)
        h = torch.randn(1, 4096, 32, requires_grad=True)
        mask = torch.ones(1, 4096, dtype=torch.bool)
        reference = block(h, mask).detach()
        with torch.autocast("cpu", dtype=torch.float16):
            out = block(h, mask)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, reference, rtol=.005, atol=.005)
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None))

    def test_crop_keeps_edge_attributes_and_offsets(self):
        data = dict(geometry_vectors=self.v[0], geometry_edge_index=self.e[:, :4],
                    geometry_edge_features=self.a[:4], geometry_edge_vectors=self.ev[:4],
                    geometry_edge_scores=self.s[:4])
        cropped = crop_geometry(data, slice(1, 4), 4)
        torch.testing.assert_close(cropped["geometry_edge_index"], torch.tensor([[1, 2], [2, 1]]))
        torch.testing.assert_close(cropped["geometry_edge_features"], self.a[2:4])
        idx, scores = _collate_contact_edges([cropped, cropped], 3, "geometry_edge_index", "geometry_edge_scores")
        torch.testing.assert_close(idx[:, 2:], idx[:, :2] + 3)
        torch.testing.assert_close(scores[:2], scores[2:])

    def test_dataset_collate_and_full_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("metadata", "labels", "sequence", "geometry"):
                (root / name).mkdir()
            for uid, seq in (("a", "ACDE"), ("b", "AC")):
                n = len(seq)
                meta = dict(residue_names_1=list(seq), chain_ids=["A"] * n)
                torch.save(dict(**meta, embeddings=torch.randn(n, 6)), root / "metadata" / f"{uid}_esm2.pt")
                torch.save(dict(labels=torch.arange(n) % 2), root / "labels" / f"{uid}_labels.pt")
                torch.save(dict(**meta, seq_features=torch.randn(n, 4)), root / "sequence" / f"{uid}.pt")
                idx = torch.tensor([list(range(n - 1)), list(range(1, n))])
                torch.save(dict(sequence=seq, geometry_vectors=torch.randn(n, 8, 3),
                    geometry_edge_index=idx, geometry_edge_features=torch.randn(n - 1, 30),
                    geometry_edge_vectors=torch.randn(n - 1, 1, 3), geometry_edge_scores=torch.ones(n - 1)),
                    root / "geometry" / f"{uid}.pt")
            ds = ESMProteinSiteDataset(esm_root=root / "metadata", label_root=root / "labels", ids=["a", "b"],
                sequence_feature_root=root / "sequence", require_sequence_features=True,
                geometry_feature_root=root / "geometry", require_geometry_features=True,
                strict_label_metadata=False, max_residues=3, crop_mode="first")
            batch = collate_esm_site_features([ds[0], ds[1]])
            self.assertEqual(batch["geometry_vectors"].shape, (2, 3, 8, 3))
            self.assertEqual(batch["geometry_edge_features"].shape, (3, 30))
            for mode in ("edge_mlp", "edge_attention", "gvp"):
                model = ESMSiteClassifier(dict(d_esm=6, d_seq=4, d_model=16, d_hidden=32,
                    n_heads=4, geometry_heads=4, geometry_mode=mode, use_contact_graph=False))
                keys = ("esm_embeddings", "seq_features", "protein_mask", "chain_ids", "chain_rel_pos",
                        "protein_rel_pos", "geometry_vectors", "geometry_edge_index", "geometry_edge_features",
                        "geometry_edge_vectors", "geometry_edge_scores")
                out = model(**{k: batch[k] for k in keys})["logits"]
                self.assertEqual(out.shape, (2, 3, 2))
                self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
