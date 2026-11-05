"""Prove the frozen row-chunked operator before retrying one complete query."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE_HASHES = {
    "protenix_lowmem_relpos.py": "bff0cb35af17da83a3791a038af17e03cc848eba830aacfc9e9bda20987aee71",
    "protenix_full_esm.py": "6cf9adccdc89538eed7a8d7b3b6f6149849d192284d64186d849c210dbd0907b",
    "run_rcsb_protenix_queue.py": "29352b1dfa24123e77da20e6f5f4fc4c3fbb5876eaeea86f041fc3086a883327",
}
PROJECT = "/data/user/shuang886/Folding/protenix_stage0_pkg/v1_1/site"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work-root", "code-root", "manifest", "cache-root"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("manifest-sha256", "cache-ready-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if digest(__file__) != os.environ["PPC_LOWMEM_RUNNER_SOURCE_SHA256"]:
        raise ValueError("Smoke orchestrator changed")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Full-length proof requires an allocated GPU and host RAM")
    args.code_root = args.code_root.resolve(strict=True)
    for name, expected in SOURCE_HASHES.items():
        if digest(args.code_root / name) != expected:
            raise ValueError("Frozen inference source changed: " + name)
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        raise ValueError("Pinned query manifest changed")
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8")), delimiter="\t"))
    if (len(rows) != 1 or int(rows[0]["n_residues"]) != 4579
            or len(rows[0]["sequence"]) != 4579 or rows[0]["split"] != "train"
            or rows[0]["unique_id"] != "useq_" + hashlib.sha256(rows[0]["sequence"].encode("ascii")).hexdigest()
            or rows[0]["seq_sha256"] != rows[0]["unique_id"][5:]):
        raise ValueError("Expected the complete native-train maximum query")
    if digest(args.cache_root / "ready.json") != args.cache_ready_sha256:
        raise ValueError("Full-query cache readiness changed")
    args.work_root = args.work_root.resolve()
    args.work_root.mkdir(parents=True, exist_ok=False)
    manifest_snapshot = args.work_root / "input_manifest.tsv"
    manifest_snapshot.write_bytes(manifest_bytes)
    manifest_snapshot.chmod(0o444)
    proof = args.work_root / "operator_equivalence.json"
    environment = {**os.environ,
                   "PPC_LOWMEM_RELP_SOURCE_SHA256": SOURCE_HASHES["protenix_lowmem_relpos.py"],
                   "PPC_FULL_ESM_ROOT": str(args.cache_root.resolve(strict=True)),
                   "PPC_FULL_ESM_READY_SHA256": args.cache_ready_sha256}
    receipt = dict(job=os.environ["SLURM_JOB_ID"], started=time.time(),
                   source_sha256=digest(__file__), inference_sources=SOURCE_HASHES,
                   manifest_sha256=args.manifest_sha256, cache_ready_sha256=args.cache_ready_sha256,
                   manifest_snapshot=str(manifest_snapshot),
                   status="started", query_count=1, length=4579, query_truncated=False,
                   model_weights_changed=False, inference_parameters_changed=False,
                   test_labels_read=False, packages_modified=False)
    try:
        subprocess.run([sys.executable, "-u", str(args.code_root / "protenix_lowmem_relpos.py"),
                        "validate", "--output", str(proof), "--full-n", "4579"],
                       env=environment, check=True)
        receipt["operator_proof_sha256"] = digest(proof)
        environment.update(PPC_LOWMEM_RELP_ACCEPTANCE=str(proof),
                           PPC_LOWMEM_RELP_ACCEPTANCE_SHA256=receipt["operator_proof_sha256"])
        if digest(manifest_snapshot) != args.manifest_sha256:
            raise ValueError("Validated manifest snapshot changed during operator proof")
        subprocess.run([sys.executable, "-u", str(args.code_root / "run_rcsb_protenix_queue.py"),
                        "--manifest", str(manifest_snapshot),
                        "--root", str(args.work_root / "queue"), "--project", PROJECT,
                        "--protenix-cli", str(args.code_root / "protenix_lowmem_relpos.py"),
                        "--batch-size", "1", "--max-attempts", "2", "--min-length", "50"],
                       env=environment, check=True)
        summary = args.work_root / "queue/summary.json"
        result = json.loads(summary.read_bytes())
        receipt["queue_summary_sha256"] = digest(summary)
        if result["total"] != 1 or result["counts"] != {"ok": 1}:
            raise ValueError("Operator proof passed but full-query folding failed: " + repr(result))
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["ended"] = time.time()
        target = args.work_root / "run_receipt.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(target)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
