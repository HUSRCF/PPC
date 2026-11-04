import importlib.util
import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_msa_features", ROOT / "scripts/preparations/m18_union/build_msa_features.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MSAFeaturesTests(unittest.TestCase):
    def test_insertions_and_exact_query(self):
        rows = MODULE.read_alignment(">q\nACD\n>a\nACtD\n>b\nA-D\n>dup\nACD\n", "ACD")
        self.assertEqual(rows, ["ACD", "A-D"])
        with self.assertRaises(ValueError):
            MODULE.read_alignment(">q\nACD\n", "ACE")
        with self.assertRaises(ValueError):
            MODULE.read_alignment(">q\nACD\n>b\nACDD\n", "ACD")

    def test_query_only_is_not_perfect_conservation(self):
        self.assertTrue(np.array_equal(MODULE.profile_features(["ACD"], "ACD"), np.zeros((3, 25))))

    def test_profile_probability_and_gap_contract(self):
        features = MODULE.profile_features(["ACD", "ADD", "A-D"], "ACD")
        self.assertEqual(features.shape, (3, 25))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(((features >= 0) & (features <= 1)).all())
        np.testing.assert_allclose(features[:, :20].sum(axis=1), 1., atol=1.e-6)
        self.assertGreater(features[1, 20], 0)
        self.assertGreater(features[1, 21], 0)
        self.assertEqual(features[0, 21], 0)
        self.assertTrue((features[:, 24] == 1).all())

    def test_windows_cover_long_proteins(self):
        for length in (1, 1000, 1001, 2049, 5000):
            cover = np.zeros(length)
            for start in MODULE.window_starts(length):
                cover[start:min(length, start + 1000)] += 1
            self.assertTrue((cover > 0).all())

    def test_subsampling_keeps_query_deterministically(self):
        rows = ["ACD", "ACE", "A-D", "DDD", "CCC"]
        self.assertEqual(MODULE.select_rows(rows, 1), [rows[0]])
        selected = MODULE.select_rows(rows, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0], rows[0])
        self.assertEqual(selected, MODULE.select_rows(rows, 3))

    def test_cache_identity_shape_finite_depth_and_checkpoint(self):
        row = {"msa_sha256": "msa"}
        base = {"sequence": "ACD", "sequence_sha256": hashlib.sha256(b"ACD").hexdigest(),
                "embeddings": torch.zeros(3, 768), "source_msa_sha256": "msa",
                "source_code_sha256": "code", "feature_mode": "encoder",
                "checkpoint_sha256": "checkpoint", "requested_depth": 64}
        MODULE.validate_payload(base, row, "ACD", "encoder", "code", "checkpoint", 64)
        for changes in ({"checkpoint_sha256": "other"}, {"requested_depth": 1},
                        {"embeddings": torch.full((3, 768), float("nan"))},
                        {"embeddings": torch.zeros(1, 768)}):
            with self.assertRaises(ValueError):
                MODULE.validate_payload({**base, **changes}, row, "ACD", "encoder", "code", "checkpoint", 64)


if __name__ == "__main__":
    unittest.main()
