import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
spec = importlib.util.spec_from_file_location("locked_infer", ROOT / "scripts/analysis/run_locked_union_inference.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class TinyModel(torch.nn.Module):
    def __init__(self, mutate=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("counter", torch.zeros(()))
        self.mutate = mutate

    def forward(self, esm_embeddings, **kwargs):
        if self.mutate:
            self.counter.add_(1)
        score = esm_embeddings[..., 0] * self.scale
        return {"logits": torch.stack((torch.zeros_like(score), score), dim=-1)}


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.groups = {"a": dict(length=3, seq_sha1=hashlib.sha1(b"AAA").hexdigest(), union_label_sha256=
                       hashlib.sha256(np.array([0, 1, 0], dtype=np.int8).tobytes()).hexdigest())}
        self.batch = dict(pdb_id=["a"], labels=torch.tensor([[0, 1, 0, -100]]),
            esm_embeddings=torch.tensor([[[.1], [.9], [.2], [0.]]]),
            seq_features=torch.zeros(1, 4, 2), protein_mask=torch.tensor([[1, 1, 1, 0]]),
            chain_ids=torch.zeros(1, 4, dtype=torch.long),
            chain_rel_pos=torch.zeros(1, 4), protein_rel_pos=torch.zeros(1, 4))
        self.hashes = {uid: row["seq_sha1"] for uid, row in self.groups.items()}

    def test_no_optimizer_no_grad_and_unchanged_state(self):
        model = TinyModel()
        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer")):
            payload, proof = M.predict_batches(model, [self.batch], ["a"], self.groups, torch.device("cpu"), self.hashes)
        self.assertEqual(len(payload["records"]), 1)
        self.assertTrue(proof["unchanged_model_state"])
        self.assertEqual(proof["optimizer_steps"], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.training)

    def test_mutation_or_incomplete_residues_rejected(self):
        with self.assertRaisesRegex(ValueError, "mutated"):
            M.predict_batches(TinyModel(True), [self.batch], ["a"], self.groups, torch.device("cpu"), self.hashes)
        self.batch["labels"][0, 1] = -100
        with self.assertRaisesRegex(ValueError, "Truncated"):
            M.predict_batches(TinyModel(), [self.batch], ["a"], self.groups, torch.device("cpu"), self.hashes)

    def test_replay_score_and_rank_guards(self):
        y, p = np.array([0, 1, 0]), np.array([.1, .9, .2])
        self.assertEqual(M.check_replay(y, p, p, [3])["max_score_error"], 0.)
        with self.assertRaisesRegex(ValueError, "differs"):
            M.check_replay(y, p + .001, p, [3])
        with self.assertRaisesRegex(ValueError, "Invalid"):
            M.check_replay(y, np.array([np.nan, .9, .2]), p, [3])

    def test_loader_uses_frozen_roots_and_full_strict_queries(self):
        config = dict(esm_root="/frozen/meta", label_root="/frozen/labels",
                      batch_size=64, eval_batch_size=64, seed=49,
                      require_contact_graph_resolved=False, max_residues=100,
                      crop_mode="random", strict_label_metadata=False)
        with patch.object(M.training, "_build_loader", return_value="loader") as build:
            self.assertEqual(M.loader_for(config, ["a"], np.array([3]), "val"), "loader")
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["esm_root"], Path("/frozen/meta"))
        self.assertEqual(kwargs["max_residues"], 0)
        self.assertEqual(kwargs["crop_mode"], "none")
        self.assertTrue(kwargs["strict_label_metadata"])
        self.assertTrue(kwargs["strict_sequence_feature_metadata"])
        self.assertFalse(kwargs["shuffle"])
        config["esm_root"] = "relative/meta"
        with self.assertRaisesRegex(ValueError, "Relative"):
            M.loader_for(config, ["a"], [3], "val")

    def test_input_sequence_and_missing_feature_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "a_esm2.pt"
            features = root / "a.pt"
            dataset = SimpleNamespace(esm_paths=[metadata])
            config = dict(sequence_feature_root=root)
            torch.save(dict(sequence="AAA"), metadata)
            torch.save(dict(sequence="AAA", features=torch.ones(3, 2)), features)
            hashes, inventory = M.audit_inputs(config, dataset, ["a"], self.groups)
            self.assertEqual(hashes, self.hashes)
            self.assertEqual(len(inventory), 2)
            (root / "a").mkdir()
            higher = root / "a/a_seq.pt"
            torch.save(torch.ones(3, 2), higher)
            self.assertEqual(M.sha(features), inventory[str(features)])
            with self.assertRaisesRegex(ValueError, "no sequence metadata"):
                M.audit_inputs(config, dataset, ["a"], self.groups)
            higher.unlink()
            torch.save(torch.ones(3, 2), features)
            with self.assertRaisesRegex(ValueError, "no sequence metadata"):
                M.audit_inputs(config, dataset, ["a"], self.groups)
            torch.save(dict(sequence="CCC", features=torch.ones(3, 2)), features)
            torch.save(dict(sequence="CCC"), metadata)
            with self.assertRaisesRegex(ValueError, "Actual input sequence differs"):
                M.audit_inputs(config, dataset, ["a"], self.groups)

    def test_failed_validation_never_starts_test_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = dict(base=tmp, members=["model"], sources={"model": {"validation_npz": "cached"}},
                        mode="single", validation=dict(pooled_AP=1., macro_AP=1.))
            y, p = np.array([0, 1, 0]), np.array([.1, .9, .2])
            with patch.object(M, "verified_lock", return_value=lock), \
                 patch.object(M, "groups_from", return_value=self.groups), \
                 patch.object(M, "cohort", return_value=(["a"], np.array([3]))) as cohort, \
                 patch.object(M, "packed", return_value=(y, p)), \
                 patch.object(M, "infer_member", side_effect=ValueError("validation failed")) as infer:
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    M.run_locked("lock", "sha", Path(tmp) / "out", "cpu")
                self.assertEqual(infer.call_args.args[-2], "val")
                self.assertEqual([call.args[1] for call in cohort.call_args_list], ["val"])

    def test_checkpoint_path_roundtrip_and_strict_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            model = TinyModel()
            config = dict(esm_root=Path(tmp), model_config={})
            torch.save(dict(config=config, epoch=2, model=model.state_dict()), checkpoint)
            source = dict(checkpoint=checkpoint, checkpoint_sha256=M.sha(checkpoint),
                          model_config={}, training_config=json.loads(json.dumps(config, default=str)),
                          epoch=2, model_state_sha256=M.state_sha(model.state_dict()))
            with patch.object(M, "strict_model", return_value=model) as strict, \
                 patch.object(M, "loader_for", return_value=SimpleNamespace(dataset=None)), \
                 patch.object(M, "audit_inputs", return_value=(self.hashes, {})) as audit, \
                 patch.object(M, "predict_batches", return_value=({"records": []}, dict(
                     state_after_sha256=source["model_state_sha256"]))):
                payload, proof = M.infer_member(source, ["a"], [3], self.groups, "val", torch.device("cpu"))
            strict.assert_called_once()
            self.assertEqual(audit.call_count, 2)
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(proof["state_after_sha256"], source["model_state_sha256"])

    def test_all_members_replay_before_test_and_no_test_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            groups = {**self.groups, "b": dict(self.groups["a"])}
            y, p = np.array([0, 1, 0]), np.array([.1, .9, .2])
            calls = []
            members = ["first", "second"]
            lock = dict(base=tmp, members=members, mode="equal_logit", test_ids=["b"],
                sources={name: dict(validation_npz="cached", checkpoint_sha256="ckpt", name=name)
                         for name in members}, validation=dict(pooled_AP=1., macro_AP=1.))

            def infer(source, ids, lengths, groups, split, device):
                calls.append((source["name"], split))
                record = dict(pdb_id=ids[0], scores=torch.as_tensor(p), targets=torch.as_tensor(y),
                    model_residue_index_0based=torch.arange(3), sequence_sha1=self.hashes["a"],
                    union_label_sha256=groups[ids[0]]["union_label_sha256"])
                return dict(epoch=1, records=[record]), dict(optimizer_steps=0)

            def choose(groups, split, count):
                return (["a"] if split == "val" else ["b"]), np.array([3])

            with patch.object(M, "verified_lock", return_value=lock), \
                 patch.object(M, "groups_from", return_value=groups), \
                 patch.object(M, "cohort", side_effect=choose), \
                 patch.object(M, "packed", return_value=(y, p)), \
                 patch.object(M, "infer_member", side_effect=infer):
                result = M.run_locked("lock", "sha", Path(tmp) / "out", "cpu")
            self.assertEqual(calls, [(name, split) for split in ("val", "test") for name in members])
            self.assertFalse(result["test_scored"])
            self.assertEqual(set(json.loads(Path(result["predictions"]).read_text())), set(members))


if __name__ == "__main__":
    unittest.main()
