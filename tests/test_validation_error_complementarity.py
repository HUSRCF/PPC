import hashlib
import importlib.util
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
SPEC = importlib.util.spec_from_file_location("validation_errors", ROOT / "scripts/analysis/analyze_validation_error_complementarity.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidationErrorTests(unittest.TestCase):
    def test_existing_blends_are_opt_in_and_only_use_original_sources(self):
        summary = dict(sources={"a": {}, "b": {}}, fixed_blends={"blend": ["a", "b"]})
        self.assertEqual(MODULE.selected_systems(summary), ["a", "b"])
        self.assertEqual(MODULE.selected_systems(summary, True), ["a", "b", "blend"])
        for definition in (["a", "a"], ["a", "missing"], ["a", "blend"], "ab", ["a", 1]):
            changed = copy.deepcopy(summary)
            changed["fixed_blends"]["blend"] = definition
            with self.assertRaises(ValueError):
                MODULE.selected_systems(changed, True)
        with self.assertRaises(ValueError):
            MODULE.selected_systems(dict(sources={"a": {}, "b": {}}, fixed_blends={"a": ["a", "b"]}), True)

    def test_fixed_blends_are_reconstructed_not_only_ap_checked(self):
        scores = {"a": np.array([.1, .3, .6]), "b": np.array([.4, .7, .9])}
        expected = MODULE.combine(scores, ["a", "b"])
        numerator = np.sqrt(scores["a"] * scores["b"])
        formula = numerator / (numerator + np.sqrt((1 - scores["a"]) * (1 - scores["b"])))
        np.testing.assert_allclose(expected, formula, rtol=1.e-14)
        MODULE.check_fixed_blend(expected, scores, ["a", "b"])
        with self.assertRaises(ValueError):
            MODULE.check_fixed_blend(expected + .001, scores, ["a", "b"])

    def test_cached_points_reject_wrong_order_labels_and_values(self):
        ids = np.asarray(["a", "b"])
        lengths = np.asarray([2, 3])
        y = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        groups = {uid: {"union_label_sha256": hashlib.sha256(part.astype(np.int8).tobytes()).hexdigest()}
                  for uid, part in zip(ids, [y[:2], y[2:]])}
        data = dict(seq_ids=ids, lengths=lengths, offsets=np.array([0, 2, 5]), labels=y,
                    scores=np.array([.1, .8, .9, .2, .7]))
        MODULE.check_arrays(data, ids, lengths, groups)
        for changed in ({**data, "seq_ids": ids[::-1]}, {**data, "labels": 1 - y},
                        {**data, "offsets": np.array([0, 3, 5])},
                        {**data, "scores": np.full(5, np.nan)}):
            with self.assertRaises(ValueError):
                MODULE.check_arrays(changed, ids, lengths, groups)

    def test_within_chain_rank_correlation_ignores_monotone_rescaling(self):
        p = np.array([.1, .3, .2, .8, .5])
        transformed = np.r_[p[:3] ** 3, np.sqrt(p[3:])]
        rows = MODULE.pairwise({"first": p, "second": transformed}, np.array([3, 2]))
        self.assertAlmostEqual(rows[0]["within_chain_Spearman_macro"], 1.)
        self.assertEqual(rows[0]["valid_chains"], 2)
        constant = MODULE.pairwise({"first": p, "second": np.full(5, .5)}, np.array([3, 2]))[0]
        self.assertIsNone(constant["pooled_score_Pearson"])
        self.assertIsNone(constant["within_chain_Spearman_macro"])

    def test_bootstrap_accepts_explicit_contrasts(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        p = np.array([.1, .8, .9, .2, .7, .3, .2, .8])
        lengths = np.array([2, 2, 2, 2])
        ap = np.array([average_precision_score(y[i:i + 2], p[i:i + 2]) for i in range(0, 8, 2)])
        result = MODULE.grouped_bootstrap(y, {"ref": p, "same": p.copy()}, {"ref": ap, "same": ap.copy()},
                                         np.array(["x", "x", "y", "y"]), lengths, 20, 1,
                                         contrasts={"same_minus_ref": ("ref", "same")})
        for metric in ("macro_AP", "pooled_AP"):
            row = result["contrasts"]["same_minus_ref"][metric]
            self.assertEqual(row["difference"], 0.)
            np.testing.assert_array_equal(row["ci95"], [0., 0.])

    def test_covariate_identity_and_finite_depth(self):
        row = dict(length="10", positive_ratio=".3", SI_component="c", MSA_effective_depth_mean="12")
        self.assertEqual(MODULE.checked_depth(row, 10, .3, "c"), 12.)
        for change in (dict(length="11"), dict(SI_component="other"), dict(positive_ratio=".4"),
                       dict(positive_ratio="nan"), dict(positive_ratio="inf"),
                       dict(MSA_effective_depth_mean="nan"), dict(MSA_effective_depth_mean="-1")):
            with self.assertRaises(ValueError):
                MODULE.checked_depth({**row, **change}, 10, .3, "c")


if __name__ == "__main__":
    unittest.main()
