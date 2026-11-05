import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/analysis/benchmark_frozen_embedding_io.py"
SPEC = importlib.util.spec_from_file_location("embedding_io", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class EmbeddingIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "split").mkdir()
        (self.root / "embeddings").mkdir()
        self.uids = ["useq_" + format(i, "064x") for i in range(1, 4)]
        (self.root / "split/train_ids.txt").write_text("\n".join(self.uids) + "\n")
        with (self.root / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pdb_id", "split", "n_residues"])
            writer.writeheader()
            for uid, length in zip(self.uids, [2, 3, 4]):
                writer.writerow(dict(pdb_id=uid, split="train", n_residues=length))
                torch.save(dict(embeddings=torch.arange(length*3456, dtype=torch.float32).reshape(length, 3456)), self.root / "embeddings" / (uid + ".pt"))
            writer.writerow(dict(pdb_id="test_unread", split="test", n_residues=5))

    def test_benchmark_is_train_only_equal_and_cleans_staging(self):
        before = {p: probe.file_sha(p) for p in (self.root / "embeddings").iterdir()}
        result = probe.benchmark(self.root, 2, 2, self.root, self.root / "embeddings")
        self.assertEqual({x["uid"] for x in result["sources"]}, {self.uids[0], self.uids[2]})
        self.assertEqual(len(result["trials"]), 18)
        self.assertEqual(len(result["summary"]), 5)
        self.assertTrue(all(x["all_modes_bitwise_equal"] for x in result["sources"]))
        self.assertTrue(all(x["source_sha256"] == x["copied_sha256"] for x in result["sources"]))
        self.assertEqual(before, {p: probe.file_sha(p) for p in (self.root / "embeddings").iterdir()})
        self.assertFalse(list(self.root.glob("ppc_embedding_io_*")))

    def test_nonfinite_values_fail_without_source_changes(self):
        p = self.root / "embeddings" / (self.uids[0] + ".pt")
        torch.save(dict(embeddings=torch.full((2, 3456), float("nan"))), p)
        before = probe.file_sha(p)
        with self.assertRaises(ValueError):
            probe.benchmark(self.root, 2, 1, self.root, self.root / "embeddings")
        self.assertEqual(probe.file_sha(p), before)
        self.assertFalse(list(self.root.glob("ppc_embedding_io_*")))

    def test_split_conflict_or_duplicates_fail(self):
        p = self.root / "split/train_ids.txt"
        p.write_text(self.uids[0] + "\n" + self.uids[0] + "\n")
        with self.assertRaises(ValueError):
            probe.select_train_rows(self.root, 2)
        p.write_text(self.uids[0] + "\nuseq_" + "f" * 64 + "\n")
        with self.assertRaises(ValueError):
            probe.select_train_rows(self.root, 2)

    def test_signed_zero_is_not_bitwise_equal(self):
        self.assertTrue(torch.equal(torch.tensor([0.0]), torch.tensor([-0.0])))
        self.assertFalse(probe.bit_equal(torch.tensor([0.0]), torch.tensor([-0.0])))

    def test_symlink_escape_fails_before_loading(self):
        p = self.root / "embeddings" / (self.uids[0] + ".pt")
        p.unlink()
        target = self.root / "outside.pt"
        target.write_text("must not be loaded")
        p.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "source bank"):
            probe.benchmark(self.root, 2, 1, self.root, self.root / "embeddings")
        self.assertFalse(list(self.root.glob("ppc_embedding_io_*")))

    def test_traversal_id_rejected(self):
        (self.root / "split/train_ids.txt").write_text("../outside\n" + self.uids[1] + "\n")
        with self.assertRaises(ValueError):
            probe.select_train_rows(self.root, 2)


if __name__ == "__main__":
    unittest.main()
