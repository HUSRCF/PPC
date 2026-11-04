"""Optional CPU acceptance against the original trusted HPC3 artifacts."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/training")]
import train_esm_site as trainer
import evaluate_esm_site as standalone
from ppcbind.data import ESMProteinSiteDataset, collate_esm_site_features


@unittest.skipUnless(os.environ.get("PPC_M18_SNAPSHOT_ROOT"), "HPC3 reference assets not requested")
class SnapshotTests(unittest.TestCase):
    def test_eval_only_preserves_provenance_and_source_epoch(self):
        external = Path(os.environ["PPC_M18_SNAPSHOT_ROOT"])
        frozen = external / "ratio_head_matrix_20261104"
        cfg = json.loads((ROOT / "configs/experiments/20261104_m18_union_of3/R2.json").read_text())
        checkpoint_path = frozen / "runs/R2_seed49/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cfg["training"].update(output_dir=str(path / "run"), init_checkpoint=str(checkpoint_path),
                device="cpu", eval_only=True, save_eval_predictions=True, save_epoch_predictions=False,
                max_train_samples=2, max_val_samples=2, num_workers=0, persistent_workers=False,
                batch_size=2, eval_batch_size=2, amp=False, progress=False,
                prediction_provenance=str(frozen / "prediction_provenance.json"))
            config_path = path / "eval.json"
            config_path.write_text(json.dumps(cfg))
            with patch.object(sys, "argv", ["train_esm_site.py", "--config", str(config_path)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(trainer.main(), 0)
            predictions = torch.load(path / "run/predictions/val.pt", map_location="cpu", weights_only=False)
            self.assertEqual(predictions["epoch"], checkpoint["epoch"])
            self.assertIn("label_contract", predictions["provenance"])
            self.assertEqual(len(predictions["records"]), 2)
            self.assertTrue(all(len(r["sequence_sha1"]) == 40 for r in predictions["records"]))
            standalone_output = path / "standalone_metrics.json"
            with patch.object(sys, "argv", ["evaluate_esm_site.py", "--config", str(config_path),
                "--checkpoint", str(checkpoint_path), "--splits", "val", "--max-samples", "2",
                "--output-json", str(standalone_output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(standalone.main(), 0)
            independent = json.loads(standalone_output.read_text())["splits"]["val"]
            shared = json.loads((path / "run/metrics.jsonl").read_text())["val"]
            for key in ("chain_ap_macro", "pooled_ap", "trapezoid_auprc", "chain_ratio_mae_raw", "chain_ratio_head_mae", "loss"):
                self.assertAlmostEqual(independent[key], shared[key], places=7)

    def test_strict_checkpoints_and_cpu_forward_equivalence(self):
        external = Path(os.environ["PPC_M18_SNAPSHOT_ROOT"])
        frozen = external / "ratio_head_matrix_20261104"
        path = frozen / "ppc_source/src/ppcbind/models/esm_site.py"
        spec = importlib.util.spec_from_file_location("trusted_frozen_m18_model", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assets = external / "training_of3_union_20261102/inputs"
        ids = (assets / "split/val_ids.txt").read_text().split()[:2]
        dataset = ESMProteinSiteDataset(
            esm_root=assets / "metadata", label_root=assets / "labels", ids=ids,
            primary_embedding_root=assets / "embeddings", sequence_feature_root=assets / "structure",
            contact_graph_root=assets / "contact", require_primary_embeddings=True,
            require_sequence_features=True, require_contact_graph=True,
            strict_label_metadata=True, strict_sequence_feature_metadata=True)
        batch = collate_esm_site_features([dataset[i] for i in range(len(dataset))])
        inputs, _ = trainer._move_batch(batch, torch.device("cpu"))
        for variant in ("R0", "R2"):
            checkpoint = torch.load(frozen / "runs" / f"{variant}_seed49" / "best.pt",
                                    map_location="cpu", weights_only=False)
            cfg = checkpoint["config"]["model_config"]
            old = module.ESMSiteClassifier(cfg).eval()
            new = trainer.ESMSiteClassifier(cfg).eval()
            old.load_state_dict(checkpoint["model"], strict=True)
            new.load_state_dict(checkpoint["model"], strict=True)
            with torch.no_grad():
                a, b = old(**inputs), new(**inputs)
            self.assertTrue(torch.equal(a["logits"], b["logits"]), variant)
            if variant == "R2":
                self.assertTrue(torch.equal(a["chain_ratio_logits"], b["chain_ratio_logits"]))


if __name__ == "__main__":
    unittest.main()
