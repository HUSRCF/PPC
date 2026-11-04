import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/experiments/20261104_m18_union_of3"


class ConfigTests(unittest.TestCase):
    def test_all_replays_are_fresh_validation_only(self):
        for name in ("A", "B", "C", "D", "E", "R0", "R1", "R2"):
            cfg = json.loads((CONFIGS / f"{name}.json").read_text())
            t = cfg["training"]
            self.assertEqual((t["epochs"], t["seed"]), (50, 49))
            self.assertIsNone(t["init_checkpoint"])
            self.assertTrue(t["validation_only"] and t["refuse_existing_run"])
            self.assertFalse(t["eval_test_each_epoch"])
            self.assertEqual(t["ratio_loss_source"], "residue_mean")
            self.assertIn("ppc_repo_replay_20261104", t["output_dir"])
            self.assertNotIn("_seed49/", t["output_dir"])
            self.assertTrue(t["save_epoch_predictions"] and t["checkpoint_every_epoch"])

    def test_auxiliary_matrix_changes_only_intended_axes(self):
        configs = [json.loads((CONFIGS / f"R{i}.json").read_text()) for i in range(3)]
        for cfg in configs:
            self.assertEqual(cfg["data"], configs[0]["data"])
            t = cfg["training"]
            self.assertEqual((t["class_weight_mode"], t["label_smoothing"], t["ratio_loss_weight"]), ("none", 0., .75))
            self.assertTrue(t["strict_aux_control"] and t["separate_ratio_head_optimizer"])
            self.assertEqual(cfg["model"]["chain_ratio_head_dropout"], 0.)
            self.assertTrue(cfg["model"]["chain_ratio_head_preserve_rng"])
        self.assertEqual([c["model"]["use_chain_ratio_head"] for c in configs], [False, True, True])
        self.assertEqual([c["model"]["chain_ratio_head_detach"] for c in configs], [True, True, False])
        self.assertEqual([c["training"]["aux_ratio_loss_weight"] for c in configs], [0., .75, .75])

    def test_loss_matrix_weights(self):
        expected = {"A": ("auto", .05, "ratio_mse", .75), "B": ("auto", .05, "ratio_mse", 3.),
                    "C": ("none", .05, "ratio_mse", .75), "D": ("none", 0., "ratio_mse", .75),
                    "E": ("none", 0., "bernoulli_rate_kl", 1.)}
        for name, values in expected.items():
            t = json.loads((CONFIGS / f"{name}.json").read_text())["training"]
            self.assertEqual(tuple(t[k] for k in ("class_weight_mode", "label_smoothing", "mass_loss_type", "ratio_loss_weight")), values)


if __name__ == "__main__":
    unittest.main()
