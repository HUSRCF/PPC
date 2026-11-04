"""Submit one frozen whole-RCSB pilot after full input acceptance; never run test."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from pack_rcsb_sequence_inputs import sha, file_identity


def dump_new(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def load_plan(path, digest):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Frozen launch plan changed")
    return json.loads(raw)


def verify_code(plan):
    root = Path(plan["code_root"]).resolve(strict=True)
    files = {str(p.relative_to(root)): p for p in root.rglob("*.py")}
    if set(files) != set(plan["source_files_sha256"]):
        raise ValueError("Frozen source file inventory changed")
    for name, digest in plan["source_files_sha256"].items():
        if not files[name].resolve().is_relative_to(root) or sha(files[name]) != digest:
            raise ValueError("Frozen source changed: " + name)


def config_contract(plan):
    path = Path(plan["config"]).resolve(strict=True)
    if sha(path) != plan["config_sha256"]:
        raise ValueError("Frozen training configuration changed")
    cfg = json.loads(path.read_text()); t, d, m = (cfg[k] for k in ("training", "data", "model"))
    required = dict(seed=49, epochs=3, device="cuda:0", validation_only=True,
                    eval_test_each_epoch=False, eval_only=False, init_checkpoint=None,
                    refuse_existing_run=True, max_train_samples=None, max_val_samples=0,
                    max_residues=0, train_crop_mode="none", eval_crop_mode="none", eval_max_batches=None)
    if any(k not in t or t[k] != v for k, v in required.items()):
        raise ValueError("Scratch/full-data/no-test training contract changed")
    if any(t[k] != v for k, v in dict(class_weight_mode="none", label_smoothing=0.0,
              loss_reduction="chain_mean", ratio_loss_weight=0.0, aux_ratio_loss_weight=0.0,
              selection_metric="pooled_ap").items()):
        raise ValueError("Plain CE pilot objective changed")
    if (plan["split_choice"] != "native_RCSB" or plan["PDBTM_test_independent_after_RCSB_training"] is not False
            or plan["partition"] != "acd_u" or plan["time_limit"] != "12:00:00"):
        raise ValueError("Wrong split, evaluation or scheduling contract")
    root = Path(plan["input_root"]).resolve(strict=True)
    for key, name in dict(esm_root="metadata", primary_embedding_root="embeddings", label_root="labels",
                          manifest="manifest.csv", split_dir="split").items():
        if Path(d[key]).resolve() != root/name:
            raise ValueError("Wrong full-data input path: " + key)
    if Path(t["output_dir"]).resolve() != Path(plan["model_run"]).resolve():
        raise ValueError("Wrong model output directory")
    if (d["require_primary_embeddings"] is not True or d["require_labels"] is not True
            or any(d[k] is not None for k in ("sequence_feature_root", "geometry_feature_root",
                   "contact_graph_root", "aux_contact_graph_root", "prottrans_embedding_root", "chain_filter_manifest"))
            or any(d[k] for k in ("require_sequence_features", "require_geometry_features", "require_contact_graph",
                                 "require_aux_contact_graph", "require_prottrans_embeddings"))
            or any(m[k] for k in ("use_seq_features", "use_prottrans_embeddings", "use_contact_graph",
                                 "use_aux_contact_graph", "use_contact_profile_features", "geometry_layers"))
            or m["geometry_mode"] != "none"):
        raise ValueError("Wrong ESM-C-only input condition")
    if (cfg["metadata"]["expected_train_queries"] != plan["expected_split_counts"]["train"]
            or cfg["metadata"]["expected_val_queries"] != plan["expected_split_counts"]["val"]):
        raise ValueError("Configuration cohort size differs from launch contract")
    return cfg


def await_inputs(plan, timeout):
    root = Path(plan["input_root"]); launch_path = Path(plan["packer_launch"])
    if sha(launch_path) != plan["packer_launch_sha256"]:
        raise ValueError("Packer launch identity changed")
    launch = json.loads(launch_path.read_text()); deadline = time.monotonic()+timeout
    if launch["host"] != plan["identity_host"]:
        raise ValueError("Producer and CPU preflight identity hosts differ")
    while not (root/"ready.json").exists():
        if (root/"failure.json").exists():
            raise RuntimeError("Full input packing failed; no GPU submission")
        if socket.gethostname() != launch["host"]:
            raise RuntimeError("Cannot observe packer PID from this host; do not infer termination or restart it")
        proc = Path("/proc")/str(launch["pid"])
        try:
            cmd = (proc/"cmdline").read_bytes().replace(b"\0", b" ").decode()
        except FileNotFoundError:
            if (root/"ready.json").exists(): break
            raise RuntimeError("Packer handle missing without full acceptance; no GPU submission")
        if str(root) not in cmd or "pack_rcsb_sequence_inputs.py" not in cmd:
            if (root/"ready.json").exists(): break
            raise RuntimeError("Packer PID no longer identifies the launched producer")
        if time.monotonic() >= deadline:
            raise TimeoutError("Observation window expired; producer is not cancelled or declared failed")
        time.sleep(min(30, max(0, deadline-time.monotonic())))


def matching_identity(actual, expected, plan):
    # Shared-FS device IDs are client-local (measured 73 on login vs 49 on a GPU node).
    if socket.gethostname() != plan["identity_host"]:
        return actual[1:] == expected[1:]
    return actual == expected


def verify_references(plan):
    root = Path(plan["input_root"])
    with (root/"feature_audit.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            path = Path(row["embedding_source"])
            expected = [int(row[k]) for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]
            link = root/"embeddings"/(row["unique_id"]+".pt")
            if (not matching_identity(file_identity(path.stat()), expected, plan)
                    or not link.is_symlink() or link.resolve(strict=True) != path):
                raise ValueError("Embedding reference changed since content audit")


def verify_inputs(plan):
    root = Path(plan["input_root"]).resolve(strict=True)
    if (root/"failure.json").exists(): raise ValueError("Input failure marker present")
    ready = json.loads((root/"ready.json").read_text())
    counts = plan["expected_split_counts"]
    if (ready["status"] != "complete" or ready["queries"] != sum(counts.values())
            or ready["split_counts"] != counts or not ready["independent_full_feature_content_audit"]
            or not ready["embedding_bank_immutability_required_through_training"]
            or not ready["embedding_inventory_rechecked_before_publication"]
            or ready["normalization_fitted"] or ready["embedding_values_modified"] or ready["test_evaluated"]
            or ready["union_summary_sha256"] != plan["union_summary_sha256"]
            or ready["source_code_sha256"] != plan["packer_source_sha256"]
            or Path(ready["original_union"]).resolve() != Path(plan["union_root"]).resolve()):
        raise ValueError("Incomplete or wrong full-data readiness receipt")
    mandatory = {"feature_audit.tsv", "manifest.csv", "split/train_ids.txt", "split/val_ids.txt", "split/test_ids.txt"}
    if set(ready["files_sha256"]) != mandatory:
        raise ValueError("Wrong readiness file inventory")
    for rel, digest in ready["files_sha256"].items():
        if sha(root/rel) != digest: raise ValueError("Changed accepted input: "+rel)
    ids = {}
    for split, count in counts.items():
        names = (root/"split"/(split+"_ids.txt")).read_text().splitlines()
        if len(names) != count or len(set(names)) != count: raise ValueError("Invalid split IDs")
        for uid in names:
            if uid in ids: raise ValueError("Cross-split exact ID")
            ids[uid] = split
    with (root/"manifest.csv").open() as f:
        records = list(csv.DictReader(f))
    if len(records) != len(ids) or {r["pdb_id"]: r["split"] for r in records} != ids:
        raise ValueError("Full manifest differs from exact split IDs")
    union = Path(plan["union_root"])
    if sha(union/"summary.json") != plan["union_summary_sha256"]:
        raise ValueError("Frozen union changed")
    source_summary = json.loads((union/"summary.json").read_text())
    if sha(union/"queries.tsv") != source_summary["files_sha256"]["queries.tsv"]:
        raise ValueError("Native query assignments changed")
    with (union/"queries.tsv").open() as f:
        source_rows = list(csv.DictReader(f, delimiter="\t"))
    canonical = {r["unique_id"]: (r["native_split"], int(r["length"]), int(r["union_positive"])) for r in source_rows}
    actual = {r["pdb_id"]: (r["split"], int(r["n_residues"]), int(r["n_positive"])) for r in records}
    if canonical != actual or any(int(r["n_negative"]) != int(r["n_residues"])-int(r["n_positive"]) for r in records):
        raise ValueError("Packed manifest differs from original native union assignments")
    smoke_path = Path(plan["smoke_acceptance"])
    if sha(smoke_path) != plan["smoke_acceptance_sha256"]:
        raise ValueError("GPU smoke acceptance changed")
    smoke = json.loads(smoke_path.read_text())
    if (smoke["test_queries"] != 0 or smoke["optimizer_steps"] != 1 or smoke["AMP_skips"] != 0
            or smoke["val_skips"] != 0 or not smoke["eval_gradient_isolation"]):
        raise ValueError("GPU functional smoke did not pass")
    smoke_launch_path = Path(plan["smoke_launch"])
    if sha(smoke_launch_path) != plan["smoke_launch_sha256"]:
        raise ValueError("GPU smoke launch identity changed")
    smoke_launch = json.loads(smoke_launch_path.read_text())
    if smoke_launch["job"] != smoke["job"] or smoke_launch["parent_config_sha256"] != plan["config_sha256"]:
        raise ValueError("GPU smoke does not match the frozen parent configuration")
    verify_references(plan)
    return sha(root/"ready.json")


def payload_inventory(plan, destination=None, check=None):
    root = Path(plan["input_root"])
    if check is not None:
        with check.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                identity = [int(row[k]) for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]
                if not matching_identity(file_identity((root/row["relative_path"]).stat()), identity, plan):
                    raise ValueError("Packed metadata/label changed after submission")
        return
    with destination.open("x", newline="") as out, (root/"feature_audit.tsv").open() as f:
        w = csv.writer(out, delimiter="\t"); w.writerow(["relative_path", "device", "inode", "bytes", "mtime_ns", "ctime_ns"])
        seen = set()
        with (root/"manifest.csv").open() as manifest:
            expected_ids = {r["pdb_id"] for r in csv.DictReader(manifest)}
        for row in csv.DictReader(f, delimiter="\t"):
            uid = row["unique_id"]
            if uid in seen or uid not in expected_ids: raise ValueError("Duplicate or unexpected feature audit ID")
            seen.add(uid)
            for kind, suffix, key in (("metadata", "_esm2.pt", "metadata_sha256"), ("labels", "_labels.pt", "label_payload_sha256")):
                rel = kind+"/"+uid+suffix; path = root/rel; before = file_identity(path.stat())
                if sha(path) != row[key] or file_identity(path.stat()) != before:
                    raise ValueError("Packed input payload changed: "+uid)
                w.writerow([rel, *before])
        if seen != expected_ids or len(seen) != sum(plan["expected_split_counts"].values()):
            raise ValueError("Incomplete feature audit inventory")


def submit(plan_path, digest, wait):
    plan = load_plan(plan_path, digest); state = Path(plan["state_root"]); state.mkdir(parents=True, exist_ok=True)
    if socket.gethostname() != plan["identity_host"]:
        raise ValueError("CPU preflight must run on the recorded identity host")
    lock = (state/"controller.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (state/"submission_attempt.json").exists(): raise ValueError("Submission already attempted;reconcile before retrying")
    await_inputs(plan, wait); verify_code(plan); config_contract(plan); ready_sha = verify_inputs(plan)
    run = Path(plan["model_run"])
    if run.exists() and any(run.iterdir()): raise ValueError("Formal model run already exists")
    payloads = state/"payload_identity.tsv"; payload_inventory(plan, destination=payloads)
    # Recheck the immutable references after the potentially long payload pass.
    if verify_inputs(plan) != ready_sha: raise ValueError("Ready receipt changed")
    seal = dict(plan_sha256=digest, ready_sha256=ready_sha, payload_identity_sha256=sha(payloads))
    seal_path = state/"seal.json"; dump_new(seal_path, seal); seal_sha = sha(seal_path)
    script = Path(plan["code_root"])/"scripts/preparations/m18_union/submit_rcsb_union_pilot.py"
    command = [plan["python"], "-u", str(script), "--plan", str(plan_path.resolve()), "--plan-sha256", digest,
               "--run", "--seal-sha256", seal_sha]
    wrap = shlex.join(["env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4", *command])
    sbatch = ["sbatch", "--parsable", "--job-name=ppc_rcsb_R0", "--partition=acd_u", "--gres=gpu:1",
              "--cpus-per-task=12", "--mem=128G", "--time=12:00:00", "--chdir="+plan["code_root"],
              "--output="+str(state/"train_%j.out"), "--error="+str(state/"train_%j.err"), "--wrap", wrap]
    dump_new(state/"submission_attempt.json", dict(command=sbatch, plan_sha256=digest, time=time.time()))
    job = subprocess.check_output(sbatch, text=True).strip()
    if not job.split(";")[0].isdigit(): raise ValueError("Unrecognized scheduler response;do not resubmit")
    dump_new(state/"submitted.json", dict(job=job, plan_sha256=digest, seal_sha256=seal_sha, time=time.time(),
                                         input_ready_sha256=ready_sha, test_enabled=False))
    print(json.dumps(dict(submitted=job, partition="acd_u", test_enabled=False)), flush=True)


def run_in_job(plan_path, digest, seal_sha):
    plan = load_plan(plan_path, digest); state = Path(plan["state_root"])
    lock = (state/"GPU_run.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not os.environ.get("SLURM_JOB_ID", "").isdigit(): raise ValueError("GPU training requires a scheduler allocation")
    seal_path = state/"seal.json"
    if sha(seal_path) != seal_sha: raise ValueError("Submission seal changed")
    seal = json.loads(seal_path.read_text())
    if seal["plan_sha256"] != digest or sha(state/"payload_identity.tsv") != seal["payload_identity_sha256"]:
        raise ValueError("Submission inventory changed")
    verify_code(plan); cfg = config_contract(plan)
    if verify_inputs(plan) != seal["ready_sha256"]: raise ValueError("Accepted input changed while queued")
    payload_inventory(plan, check=state/"payload_identity.tsv")
    import torch
    if not torch.cuda.is_available() or torch.__version__ != "2.7.1+cu128":
        raise ValueError("Required approved CUDA Torch is unavailable;no CPU fallback")
    print(json.dumps(dict(GPU=torch.cuda.get_device_name(0), torch=torch.__version__, test_enabled=False)), flush=True)
    dump_new(state/"GPU_started.json", dict(job=os.environ["SLURM_JOB_ID"], time=time.time(), plan_sha256=digest,
                                            input_ready_sha256=seal["ready_sha256"], test_enabled=False))
    env = {**os.environ, "PYTHONPATH": str(Path(plan["code_root"])/"src")}
    result = subprocess.run([plan["python"], "-u", str(Path(plan["code_root"])/"scripts/training/train_esm_site.py"),
                             "--config", plan["config"]], cwd=plan["code_root"], env=env)
    dump_new(state/"GPU_returned.json", dict(exit_code=result.returncode, time=time.time(), test_enabled=False))
    if result.returncode: raise SystemExit(result.returncode)
    verify_code(plan); config_contract(plan)
    if verify_inputs(plan) != seal["ready_sha256"]: raise ValueError("Inputs changed during training")
    payload_inventory(plan, check=state/"payload_identity.tsv")
    runtime = json.loads((Path(plan["model_run"])/"config.json").read_text())
    if (runtime["n_train"] != plan["expected_split_counts"]["train"] or runtime["n_val"] != plan["expected_split_counts"]["val"]
            or runtime["n_test"] != 0 or runtime["init_checkpoint"] is not None or runtime["eval_test_each_epoch"]):
        raise ValueError("Unexpected actual training contract")
    for key, value in dict(seed=49, epochs=3, validation_only=True, max_residues=0,
                           train_crop_mode="none", eval_crop_mode="none").items():
        if runtime[key] != value: raise ValueError("Unexpected runtime training option: "+key)
    records = [json.loads(line) for line in (Path(plan["model_run"])/"metrics.jsonl").read_text().splitlines() if line.strip()]
    epochs = [r for r in records if r.get("event")=="epoch"]
    if [r["epoch"] for r in epochs] != [1,2,3] or any("test" in r or r.get("event")=="test" for r in records):
        raise ValueError("Incomplete or test-exposed pilot")
    for r in epochs:
        v=r["val"]
        if (v["valid_chains"] != plan["expected_split_counts"]["val"]
                or v["valid_residues"] != plan["expected_val_residues"] or v["skipped_nonfinite"] != 0
                or r["train"]["valid_chains"] != plan["expected_split_counts"]["train"]
                or v["eval_grad_mode_enabled"] or v["eval_logits_require_grad"]
                or v["eval_parameter_grads_present"] or v["eval_parameter_version_changes"]):
            raise ValueError("Invalid full validation or gradient isolation")
    dump_new(state/"pilot_completed.json", dict(epochs=3, time=time.time(), test_evaluated=False,
             metrics_sha256=sha(Path(plan["model_run"])/"metrics.jsonl"), model_run=plan["model_run"]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--plan-sha256",required=True);p.add_argument("--wait-seconds",type=int,default=0)
    p.add_argument("--run",action="store_true");p.add_argument("--seal-sha256")
    args=p.parse_args()
    if args.run:
        if not args.seal_sha256: raise ValueError("GPU mode requires a submission seal")
        run_in_job(args.plan,args.plan_sha256,args.seal_sha256)
    else: submit(args.plan,args.plan_sha256,args.wait_seconds)


if __name__=="__main__":main()
