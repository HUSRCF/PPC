import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
spec = importlib.util.spec_from_file_location("runtime_replay", ROOT / "scripts/analysis/replay_union_validation_sources.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class RuntimeReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = self.root / "split"
        self.split.mkdir()
        (self.split / "val_ids.txt").write_text("a\n")
        self.groups = {"a": dict(length=3, seq_sha1=hashlib.sha1(b"AAA").hexdigest(),
            union_label_sha256=hashlib.sha256(np.array([0, 1, 0], dtype=np.int8).tobytes()).hexdigest())}
        self.p = np.array([.1, .9, .2])
        self.row = dict(pdb_id="a", targets=torch.tensor([0, 1, 0]), scores=torch.as_tensor(self.p),
            model_residue_index_0based=torch.arange(3), sequence_sha1=self.groups["a"]["seq_sha1"],
            union_label_sha256=self.groups["a"]["union_label_sha256"])
        self.prediction = dict(epoch=1, records=[self.row])
        self.config = dict(n_train=4868, n_val=1, n_test=0, eval_test_each_epoch=False,
                           split_dir=str(self.split), model_config={})
        for key, value in (("VAL_QUERIES", 1), ("VAL_RESIDUES", 3)):
            p = patch.object(M, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.points = self.root / "points"
        self.points.mkdir()
        self.metrics = self.root / "metrics.jsonl"
        self.metrics.write_text("{}\n")
        self.pred = self.root / "val_epoch_001.pt"
        torch.save(self.prediction, self.pred)
        (self.root / "epochs").mkdir()
        torch.save(dict(config=self.config, model={}, epoch=1), self.root / "epochs/epoch_001.pt")
        np.savez_compressed(self.points / "model_val.npz", seq_ids=np.array(["a"]), lengths=[3],
                            offsets=[0, 3], labels=[0, 1, 0], scores=self.p)
        summary = dict(test_read=False, validation_sequences=1, validation_residues=3,
            sources={"model": dict(epoch=1, prediction=str(self.pred), metrics=str(self.metrics),
                prediction_sha256=M.inference.sha(self.pred), metrics_sha256=M.inference.sha(self.metrics))},
            fixed_blends={})
        (self.points / "summary.json").write_text(json.dumps(summary))

    def test_only_validation_ids_and_complete_metadata(self):
        groups, digest = M.validation_groups(self.prediction, self.config)
        self.assertEqual(groups, self.groups)
        self.assertEqual(digest, M.inference.sha(self.split / "val_ids.txt"))
        bad = dict(self.config, n_test=1)
        with self.assertRaisesRegex(ValueError, "validation-only"):
            M.validation_groups(self.prediction, bad)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            M.validation_groups(dict(records=[self.row, self.row]), self.config)
        (self.split / "val_ids.txt").write_text("different\n")
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            M.validation_groups(self.prediction, self.config)

    def test_replay_never_opens_test_or_locked_evaluation(self):
        with patch.object(M.inference, "infer_member", return_value=(self.prediction, {})) as infer, \
             patch.object(M.inference, "run_locked", side_effect=AssertionError("test-capable path")):
            result = M.replay(self.points, self.root / "out", "cpu")
        self.assertEqual(infer.call_args.args[-2], "val")
        self.assertFalse(result["test_read"])
        self.assertFalse(result["test_inference_permitted"])
        self.assertFalse(result["model_selection"])
        self.assertEqual(result["reports"]["model"]["actual"]["pooled_AP"], 1.)
        with self.assertRaises(FileExistsError):
            M.replay(self.points, self.root / "out", "cpu")

    def test_bad_replay_fails_with_audit_and_without_relaxation(self):
        row = {**self.row, "scores": torch.as_tensor(self.p + .001)}
        payload = dict(epoch=1, records=[row])
        with patch.object(M.inference, "infer_member", return_value=(payload, {})):
            with self.assertRaisesRegex(ValueError, "differs"):
                M.replay(self.points, self.root / "bad", "cpu")
        failure = json.loads((self.root / "bad/failure.json").read_text())
        self.assertFalse(failure["test_read"])
        self.assertFalse(failure["test_inference_permitted"])
        self.assertEqual(failure["stage"], "source_reproduction")
        self.assertFalse((self.root / "bad/summary.json").exists())

    def test_inference_exception_retains_failure_stage(self):
        with patch.object(M.inference, "infer_member", side_effect=RuntimeError("injected forward failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                M.replay(self.points, self.root / "forward_bad", "cpu")
        failure = json.loads((self.root / "forward_bad/failure.json").read_text())
        self.assertEqual(failure["stage"], "validation_inference")
        self.assertEqual(failure["member"], "model")
        self.assertFalse(failure["test_read"])

    def test_manifest_mutation_rejected_instead_of_recording_new_hash(self):
        def mutate(*args):
            path = self.points / "summary.json"
            obj = json.loads(path.read_text())
            obj["sources"] = {}
            path.write_text(json.dumps(obj))
            return self.prediction, {}

        with patch.object(M.inference, "infer_member", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                M.replay(self.points, self.root / "mutated", "cpu")
        failure = json.loads((self.root / "mutated/failure.json").read_text())
        self.assertEqual(failure["stage"], "frozen_artifact_integrity")
        self.assertFalse((self.root / "mutated/summary.json").exists())

    def test_shared_path_first_digest_cannot_be_overwritten(self):
        path = self.points / "summary.json"
        obj = json.loads(path.read_text())
        obj["sources"]["second"] = dict(obj["sources"]["model"])
        path.write_text(json.dumps(obj))
        (self.points / "second_val.npz").write_bytes((self.points / "model_val.npz").read_bytes())

        def mutate(*args):
            with (self.split / "val_ids.txt").open("a") as handle:
                handle.write("\n")
            return self.prediction, {}

        with patch.object(M.inference, "infer_member", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed between members"):
                M.replay(self.points, self.root / "shared_mutation", "cpu")
        failure = json.loads((self.root / "shared_mutation/failure.json").read_text())
        self.assertEqual(failure["member"], "second")
        self.assertFalse((self.root / "shared_mutation/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
