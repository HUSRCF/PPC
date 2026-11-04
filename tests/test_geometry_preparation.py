import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/preparations/m18_union" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = module("geom_prep", "build_geometry_features.py")
configs = module("geom_configs", "prepare_geometry_screen.py")


class GeometryPreparationTests(unittest.TestCase):
    def test_edge_invariants_under_rigid_motion(self):
        rng = np.random.default_rng(49)
        ca = rng.normal(size=(12, 3)).cumsum(0).astype(np.float32)
        vec = rng.normal(size=(12, 8, 3)).astype(np.float32)
        frames = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(12)]).astype(np.float32)
        q = rng.uniform(.4, .9, 12).astype(np.float32)
        valid = np.ones(12, dtype=np.float32)
        rot = np.linalg.qr(rng.normal(size=(3, 3)))[0].astype(np.float32)
        a = prep.make_geometry(ca, q, vec, frames, valid)
        b = prep.make_geometry(ca @ rot + np.array([15., -7., 9.]), q, vec @ rot, frames @ rot, valid)
        torch.testing.assert_close(a["geometry_edge_index"], b["geometry_edge_index"])
        torch.testing.assert_close(a["geometry_edge_features"], b["geometry_edge_features"], atol=2.e-5, rtol=2.e-5)
        torch.testing.assert_close(a["geometry_edge_scores"], b["geometry_edge_scores"], atol=1.e-6, rtol=1.e-6)
        self.assertEqual(a["geometry_edge_features"].shape[1], 30)
        self.assertFalse(bool((a["geometry_edge_index"][0] == a["geometry_edge_index"][1]).any()))

    def test_small_chain_and_no_neighbors(self):
        for n in (1, 2):
            result = prep.make_geometry(np.zeros((n, 3), dtype=np.float32), np.ones(n, dtype=np.float32),
                np.zeros((n, 8, 3), dtype=np.float32), np.zeros((n, 3, 3), dtype=np.float32), np.zeros(n))
            self.assertEqual(result["geometry_edge_features"].shape, (n * (n - 1), 30))
            self.assertTrue(torch.isfinite(result["geometry_edge_features"]).all())

    def test_eight_configs_keep_contract_and_one_axis(self):
        self.assertEqual(len(configs.VARIANTS), 8)
        for name in configs.VARIANTS:
            cfg = configs.build(ROOT, Path("/tmp/geometry_test"), name)
            t, d, m = cfg["training"], cfg["data"], cfg["model"]
            self.assertEqual(t["seed"], 49)
            self.assertTrue(t["validation_only"])
            self.assertFalse(t["eval_test_each_epoch"])
            self.assertIsNone(t["init_checkpoint"])
            self.assertEqual(t["ratio_loss_weight"], .75)
            self.assertEqual(t["epochs"], 50)
            self.assertEqual(m["local_mlp_blocks"], 2)
            self.assertFalse(m.get("use_chain_ratio_head", False))
            if name.startswith("G") and name != "G0_of3_mean":
                self.assertFalse(m["use_contact_graph"])
                self.assertTrue(d["require_geometry_features"])
            if name.startswith("C"):
                self.assertFalse(m["use_global_context"])
                self.assertTrue(m["use_contact_graph"])


if __name__ == "__main__":
    unittest.main()
