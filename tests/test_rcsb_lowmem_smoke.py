import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "scripts/preparations/m18_union/run_rcsb_lowmem_smoke.py"
SPEC = importlib.util.spec_from_file_location("rcsb_lowmem_smoke", PATH)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / "code"
        self.code.mkdir()
        self.hashes = {}
        for name in smoke.SOURCE_HASHES:
            p = self.code / name
            p.write_text("frozen test helper")
            self.hashes[name] = smoke.digest(p)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        (self.cache / "ready.json").write_text("{}")
        self.manifest = self.root / "original.tsv"
        seq = "A" * 4579
        sha = hashlib.sha256(seq.encode()).hexdigest()
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["unique_id", "seq_sha256", "n_residues", "sequence", "split"], delimiter="\t")
            writer.writeheader()
            writer.writerow(dict(unique_id="useq_" + sha, seq_sha256=sha, n_residues=4579, sequence=seq, split="train"))
        self.raw = self.manifest.read_bytes()
        self.argv = [str(PATH), "--work-root", "run", "--code-root", str(self.code),
                     "--manifest", str(self.manifest), "--cache-root", str(self.cache),
                     "--manifest-sha256", smoke.digest(self.manifest),
                     "--cache-ready-sha256", smoke.digest(self.cache / "ready.json")]

    def invoke(self, child):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with patch.dict(os.environ, {"PPC_LOWMEM_RUNNER_SOURCE_SHA256": smoke.digest(PATH), "SLURM_JOB_ID": "test"}), \
                    patch.object(smoke, "SOURCE_HASHES", self.hashes), \
                    patch.object(smoke.sys, "argv", self.argv), \
                    patch.object(smoke.subprocess, "run", side_effect=child) as run:
                smoke.main()
                return run
        finally:
            os.chdir(previous)

    def test_snapshot_survives_original_mutation_and_paths_are_absolute(self):
        def child(command, env, check):
            self.assertTrue(check)
            if "validate" in command:
                proof = Path(command[command.index("--output") + 1])
                self.assertTrue(proof.is_absolute())
                proof.write_text("{}")
                self.manifest.write_text("changed original input during proof")
            else:
                snapshot = Path(command[command.index("--manifest") + 1])
                self.assertEqual(snapshot, self.root / "run/input_manifest.tsv")
                self.assertEqual(snapshot.read_bytes(), self.raw)
                self.assertTrue(Path(env["PPC_LOWMEM_RELP_ACCEPTANCE"]).is_absolute())
                queue = Path(command[command.index("--root") + 1])
                queue.mkdir()
                (queue / "summary.json").write_text(json.dumps(dict(total=1, counts=dict(ok=1))))
        run = self.invoke(child)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(json.loads((self.root / "run/run_receipt.json").read_text())["status"], "complete")

    def test_failed_proof_never_starts_fold(self):
        calls = []
        def child(command, env, check):
            calls.append(command)
            raise subprocess.CalledProcessError(1, command)
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(child)
        self.assertEqual(len(calls), 1)
        self.assertIn("validate", calls[0])
        self.assertFalse((self.root / "run/queue").exists())
        self.assertEqual(json.loads((self.root / "run/run_receipt.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
