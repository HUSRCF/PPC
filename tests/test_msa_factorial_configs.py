import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_msa_factorial", ROOT / "scripts/preparations/m18_union/prepare_msa_factorial.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FactorialTests(unittest.TestCase):
    def test_only_loss_axes_differ_within_pairs(self):
        for feature in ("none", "profile", "encoder"):
            for seed in (49, 50, 51):
                d = MODULE.build_config(ROOT, Path("/tmp/ppc_msa_test"), "D", feature, seed)
                e = MODULE.build_config(ROOT, Path("/tmp/ppc_msa_test"), "E", feature, seed)
                self.assertEqual(d["data"], e["data"])
                self.assertEqual(d["model"], e["model"])
                differences = {k for k in d["training"] if d["training"][k] != e["training"][k]}
                self.assertEqual(differences, {"output_dir", "mass_loss_type", "ratio_loss_weight"})
                self.assertEqual((d["training"]["mass_loss_type"], e["training"]["mass_loss_type"]), ("ratio_mse", "bernoulli_rate_kl"))
                self.assertTrue(d["training"]["validation_only"])
                self.assertFalse(d["training"]["eval_test_each_epoch"])
                self.assertIsNone(d["training"]["init_checkpoint"])
                self.assertEqual(d["training"]["epochs"], 50)
                self.assertFalse(d["model"]["use_chain_ratio_head"])

    def test_fresh_outputs_and_feature_dimensions(self):
        outputs = set()
        for loss in ("D", "E"):
            for feature in ("none", "profile", "encoder"):
                for seed in (49, 50, 51):
                    cfg = MODULE.build_config(ROOT, Path("/tmp/ppc_msa_test"), loss, feature, seed)
                    outputs.add(cfg["training"]["output_dir"])
                    if feature != "none":
                        self.assertEqual(cfg["model"]["d_prottrans"], {"profile": 25, "encoder": 768}[feature])
                        self.assertTrue(cfg["data"]["require_prottrans_embeddings"])
        self.assertEqual(len(outputs), 18)


if __name__ == "__main__":
    unittest.main()
