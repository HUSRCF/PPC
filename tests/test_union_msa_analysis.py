import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
SPEC = importlib.util.spec_from_file_location("msa_analysis", ROOT / "scripts/analysis/analyze_union_msa_complementarity.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class AnalysisTests(unittest.TestCase):
    def test_weighted_ap_matches_explicit_component_resampling_with_ties(self):
        y = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
        p = np.array([.9, .5, .5, .2, .2, .1])
        comp = np.array([0, 0, 1, 1, 2, 2])
        for counts in (np.array([1, 1, 1]), np.array([0, 2, 1]), np.array([3, 0, 0])):
            take = np.repeat(np.arange(len(y)), counts[comp])
            expected = average_precision_score(y[take], p[take])
            self.assertAlmostEqual(M.weighted_ap(M.ap_curve(y, p, comp), counts), expected, places=14)

    def test_fixed_logit_average_not_probability_average(self):
        a, b = np.array([.9, .2, 0., 1.]), np.array([.4, .2, 1., 0.])
        result = M.sigmoid(.5 * M.logit(a) + .5 * M.logit(b))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1], .2)
        np.testing.assert_allclose(result[2:], .5, atol=1.e-9)
        self.assertNotAlmostEqual(result[0], .65)

    def test_alignment_rejects_nonval_duplicates_hash_and_indices(self):
        y = torch.tensor([1, 0, 1])
        h = hashlib.sha256(y.numpy().astype(np.int8).tobytes()).hexdigest()
        group = {"u": dict(length="3", seq_sha1="sequence", union_label_sha256=h)}
        record = dict(pdb_id="u", targets=y, scores=torch.tensor([.8, .3, .4]),
            model_residue_index_0based=torch.arange(3), sequence_sha1="sequence", union_label_sha256=h)
        indexed = M.index_predictions(dict(records=[record]), {"u"}, group)
        np.testing.assert_array_equal(indexed["u"][0], y)
        for rows in ([record, record], [{**record, "pdb_id": "train"}],
                     [{**record, "model_residue_index_0based": torch.tensor([0, 2, 1])}],
                     [{**record, "union_label_sha256": "wrong"}],
                     [{**record, "targets": torch.tensor([0, 0, 1])}]):
            with self.assertRaises(ValueError):
                M.index_predictions(dict(records=rows), {"u"}, group)

    def test_macro_bootstrap_reweights_chains_not_components(self):
        names = ["x"]
        curve = M.ap_curve(np.array([1, 0]), np.array([.8, .2]), np.array([0, 1]))
        sums = np.array([[.3, 2.]])
        chain_counts = np.array([1, 4])
        M.init_bootstrap([curve], names, sums, chain_counts)
        value = M.bootstrap_one(np.array([2, 1]))
        self.assertAlmostEqual(value[0, 0], (2 * .3 + 2.) / 6.)

    def test_topk_matches_current_ceil_policy(self):
        y = np.zeros(21, dtype=np.uint8)
        y[1] = 1
        p = np.linspace(.95, .05, 21)
        result = M.point_metrics(y, p, np.array([21]))
        self.assertEqual(result["L/20_ACC_macro"], .5)

    def test_constant_or_missing_correlations_are_json_safe(self):
        self.assertIsNone(M.safe_spearman([1, 2, 3], [0, 0, 0]))
        self.assertIsNone(M.safe_spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(M.safe_spearman([1, np.nan], [2, 3]))
        self.assertAlmostEqual(M.safe_spearman([1, 2, 3], [3, 2, 1]), -1.)


if __name__ == "__main__":
    unittest.main()
