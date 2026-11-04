import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
SPEC = importlib.util.spec_from_file_location("fixed_blends", ROOT / "scripts/analysis/analyze_union_fixed_blends.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixedBlendTests(unittest.TestCase):
    def test_equal_logit_not_probability_averaging(self):
        scores = {"a": np.array([.01, .2, .9]), "b": np.array([.8, .4, .6])}
        expected = 1 / (1 + np.exp(-.5 * (
            np.log(scores["a"] / (1 - scores["a"])) + np.log(scores["b"] / (1 - scores["b"])))))
        np.testing.assert_allclose(MODULE.combine(scores, ["a", "b"]), expected)
        self.assertFalse(np.allclose(expected, .5 * (scores["a"] + scores["b"])))

    def test_duplicate_and_missing_sources_rejected(self):
        scores = {"a": np.array([.3]), "b": np.array([.8])}
        for members in (["a"], ["a", "a"]):
            with self.assertRaises(ValueError):
                MODULE.combine(scores, members)
        with self.assertRaises(KeyError):
            MODULE.combine(scores, ["a", "missing"])


if __name__ == "__main__":
    unittest.main()
