import copy
import csv
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
spec = importlib.util.spec_from_file_location("locked_eval", ROOT / "scripts/analysis/locked_union_evaluation.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class LockedEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, value in (("VAL_QUERIES", 2), ("VAL_RESIDUES", 5), ("TEST_QUERIES", 1)):
            p = patch.object(M, name, value)
            p.start()
            self.addCleanup(p.stop)
        (self.root / "data").mkdir()
        self.points = self.root / "points"
        self.points.mkdir()
        self.y = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        self.p = np.array([.1, .9, .2, .8, .3])
        self.rows = []
        for uid, split, seq, y in (("a", "val", "AAA", self.y[:3]),
                                   ("b", "val", "AA", self.y[3:]),
                                   ("c", "test", "CC", np.array([1, 0]))):
            self.rows.append(dict(union_seq_id=uid, split=split, length=len(seq),
                seq_sha1=hashlib.sha1(seq.encode()).hexdigest(),
                union_label_sha256=hashlib.sha256(y.astype(np.int8).tobytes()).hexdigest()))
        with (self.root / "data/sequence_groups.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(self.rows)
        records = [self.record("a", self.y[:3], self.p[:3]), self.record("b", self.y[3:], self.p[3:])]
        self.pred = self.root / "val.pt"
        torch.save(dict(epoch=1, records=records), self.pred)
        self.metrics = self.root / "metrics.jsonl"
        val = dict(valid_chains=2, valid_residues=5, skipped_nonfinite=0, pooled_ap=1., chain_ap_macro=1.)
        self.metrics.write_text("".join(json.dumps(dict(event="epoch", epoch=i, val=val)) + "\n" for i in range(1, 51)))
        cfg = dict(d_esm=6, d_seq=4, d_model=16, d_hidden=32, n_heads=4, dropout=0., classifier_dropout=0.)
        self.model = M.strict_model(cfg, M.ESMSiteClassifier(cfg).state_dict())
        (self.root / "epochs").mkdir()
        self.checkpoint = self.root / "epochs/epoch_001.pt"
        torch.save(dict(epoch=1, model=self.model.state_dict(), val=val,
                        config=dict(n_train=4868, n_val=2, n_test=0, model_config=cfg)), self.checkpoint)
        np.savez_compressed(self.points / "model_val.npz", seq_ids=np.array(["a", "b"]),
            lengths=np.array([3, 2]), offsets=np.array([0, 3, 5]), labels=self.y, scores=self.p)
        summary = dict(validation_sequences=2, validation_residues=5, test_read=False,
            sources={"model": dict(epoch=1, prediction=str(self.pred), metrics=str(self.metrics),
                prediction_sha256=M.sha(self.pred), metrics_sha256=M.sha(self.metrics))},
            fixed_blends={}, metrics={"model": dict(average_precision=1., chain_ap_macro=1.)})
        (self.points / "summary.json").write_text(json.dumps(summary))
        self.completion = self.root / "completion.json"
        self.completion.write_text(json.dumps({"model": dict(state="COMPLETED", exit_code="0:0")}))

    def record(self, uid, y, p):
        row = next(r for r in self.rows if r["union_seq_id"] == uid)
        return dict(pdb_id=uid, targets=torch.as_tensor(y), scores=torch.as_tensor(p),
                    model_residue_index_0based=torch.arange(len(y)),
                    sequence_sha1=row["seq_sha1"], union_label_sha256=row["union_label_sha256"])

    def lock(self):
        value = M.make_lock(self.root, self.points, "model", self.completion)
        path = self.root / "lock.json"
        M.write_new(path, value)
        return path, M.sha(path), value

    def test_validation_lock_and_strict_checkpoint(self):
        path, digest, value = self.lock()
        self.assertEqual(M.verified_lock(path, digest)["system"], "model")
        self.assertEqual(value["test_ids"], ["c"])
        self.assertFalse(value["test_threshold_search"])
        with self.assertRaises(FileExistsError):
            M.write_new(path, value)
        bad = dict(self.model.state_dict())
        bad.pop(next(iter(bad)))
        with self.assertRaises(RuntimeError):
            M.strict_model(value["sources"]["model"]["model_config"], bad)
        bad = {key: tensor.clone() for key, tensor in self.model.state_dict().items()}
        next(tensor for tensor in bad.values() if tensor.is_floating_point()).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            M.strict_model(value["sources"]["model"]["model_config"], bad)

    def test_subtarget_fails_before_test_cohort_or_checkpoint_reads(self):
        np.savez_compressed(self.points / "model_val.npz", seq_ids=np.array(["a", "b"]),
            lengths=np.array([3, 2]), offsets=np.array([0, 3, 5]), labels=self.y, scores=1 - self.p)
        with patch.object(M, "groups_from", side_effect=AssertionError("shared/test metadata read")), patch.object(M.torch, "load", side_effect=AssertionError("checkpoint read")):
            with self.assertRaisesRegex(ValueError, "target/guardrail"):
                M.make_lock(self.root, self.points, "model", self.root / "missing_completion.json")

    def test_terminal_failure_rejected(self):
        self.completion.write_text(json.dumps({"model": dict(state="FAILED", exit_code="1:0")}))
        with self.assertRaisesRegex(ValueError, "terminal-state"):
            M.make_lock(self.root, self.points, "model", self.completion)

    def test_changed_lock_checkpoint_and_nonfinite_contract_rejected(self):
        path, digest, value = self.lock()
        with self.assertRaisesRegex(ValueError, "checksum"):
            M.verified_lock(path, "0" * 64)
        for changed in (dict(pooled_AP=float("nan"), macro_AP=1.),
                        dict(pooled_AP=1., macro_AP=float("inf")),
                        dict(pooled_AP=1.1, macro_AP=1.)):
            bad = copy.deepcopy(value)
            bad["validation"] = changed
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "contract"):
                M.verified_lock(path, M.sha(path))
        path.write_text(json.dumps(value))
        with self.checkpoint.open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            M.verified_lock(path, M.sha(path))

    def test_fixed_metrics_never_search_test_thresholds_and_use_ceil(self):
        y = np.r_[1, np.zeros(20)].astype(np.uint8)
        p = np.linspace(.99, .01, 21)
        with patch("build_validation_logit_ensemble.select_threshold", side_effect=AssertionError("test search")):
            result = M.fixed_gate_metrics(y, p, np.array([21]), .8, .6)
        self.assertEqual(result["selected_threshold"]["threshold"], .8)
        self.assertEqual(result["at_validation_MCC_gate"]["threshold"], .6)
        self.assertEqual(result["L/20_ACC_macro"], .5)
        self.assertFalse(result["test_threshold_search"])
        self.assertNotIn("best_mcc", result)

    def test_score_requires_exact_members_and_validated_inference_receipt(self):
        path, digest, value = self.lock()
        pred = self.root / "test.pt"
        provenance = dict(lock_sha256=digest, member="model", split="test",
                          checkpoint_sha256=value["sources"]["model"]["checkpoint_sha256"])
        torch.save(dict(epoch=1, records=[self.record("c", np.array([1, 0]), np.array([.7, .2]))],
                        provenance=provenance), pred)
        receipt = dict(lock_sha256=digest, member="model", split="test",
            checkpoint_sha256=value["sources"]["model"]["checkpoint_sha256"], prediction_sha256=M.sha(pred),
            no_optimizer_created=True, unchanged_model_state=True, validation_reproduced=True,
            state_before_sha256=value["sources"]["model"]["model_state_sha256"],
            state_after_sha256=value["sources"]["model"]["model_state_sha256"],
            eval_grad_mode_enabled=False, parameter_grads_present=0, optimizer_steps=0)
        rp = pred.with_suffix(".receipt.json")
        rp.write_text(json.dumps(receipt))
        result = M.score_test(path, digest, {"model": str(pred)}, self.root / "scored")
        self.assertEqual(result["average_precision"], 1.)
        self.assertFalse(result["test_threshold_search"])
        with patch.object(M.torch, "load", side_effect=AssertionError("repeated test read")):
            with self.assertRaises(FileExistsError):
                M.score_test(path, digest, {"model": str(pred)}, self.root / "scored")
            dangling = self.root / "dangling"
            dangling.symlink_to(self.root / "missing")
            with self.assertRaises(FileExistsError):
                M.score_test(path, digest, {"model": str(pred)}, dangling)
        reservation = self.root / "reserved"
        with patch.object(M, "groups_from", side_effect=AssertionError("test read")):
            with self.assertRaisesRegex(AssertionError, "test read"):
                M.score_test(path, digest, {"model": str(pred)}, reservation)
        self.assertTrue(reservation.is_dir())
        with self.assertRaisesRegex(ValueError, "Exactly"):
            M.score_test(path, digest, {"extra": str(pred)}, self.root / "extra")
        receipt["validation_reproduced"] = False
        rp.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "integrity"):
            M.score_test(path, digest, {"model": str(pred)}, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
