#!/data/user/shuang886/.conda/envs/fold/bin/python
"""Precompute full-query ESM2 caches, then use the unchanged Protenix CLI."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from run_rcsb_protenix_queue import read_manifest, write_atomic


SOURCE_HASHES = {
    "runner/batch_inference.py": "98e2a0149359b6507e5c385665ad0365348ea8a6e361e6519c1f214801d5635b",
    "protenix/data/inference/infer_dataloader.py": "ef87dd0c928720d54e0243a30b4be57d7f5325dad7e27dcf27f85b1db4051112",
    "protenix/data/esm/esm_featurizer.py": "2ac3a6fa513e84a736f14e5d2d57ad156014b5cb7e785e2ca17e6f9d896ea657",
    "protenix/data/esm/compute_esm.py": "ce3fc45709d088ecba45fb49c8dafd4882a74401a24f7df9a91f8115091a9363",
}
MODEL = "esm2-3b"
MAX_QUERY_LENGTH = 4579
QUEUE_SOURCE_SHA256 = "29352b1dfa24123e77da20e6f5f4fc4c3fbb5876eaeea86f041fc3086a883327"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    from importlib import import_module
    from importlib.metadata import distribution, version
    import torch
    package = distribution("protenix")
    if version("protenix") != "1.1.0" or torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Approved Protenix/Torch/CUDA runtime required;no CPU fallback or installation")
    if digest(Path(__file__).with_name("run_rcsb_protenix_queue.py")) != QUEUE_SOURCE_SHA256:
        raise ValueError("Queue helper changed")
    for rel, expected in SOURCE_HASHES.items():
        source = Path(package.locate_file(rel)).resolve(strict=True)
        if digest(source) != expected:
            raise ValueError("Installed Protenix source changed: " + rel)
        module = import_module(rel[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != source:
            raise ValueError("Imported Protenix source differs from distribution: " + rel)
    return torch


def cache_path(root, uid):
    label = uid + "_1"
    return Path(root) / MODEL / label / (label + ".pt")


def check_tensor(path, length, torch):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (length, 2560)
            or not value.is_floating_point() or value.requires_grad
            or not bool(torch.isfinite(value).all()) or not bool(value.ne(0).any())):
        raise ValueError("Invalid full-query ESM cache: " + str(path))
    return dict(dtype=str(value.dtype), sha256=digest(path), length=length)


def prepare(manifest, root):
    rows, manifest_sha = read_manifest(manifest)
    if any(int(r["n_residues"]) > MAX_QUERY_LENGTH for r in rows):
        raise ValueError("Query exceeds this historical-repair contract;no truncation")
    torch = runtime()
    checkpoint = Path(os.environ["PROTENIX_ROOT_DIR"]) / "checkpoint"
    weights = [checkpoint / name for name in ("esm2_t36_3B_UR50D.pt", "esm2_t36_3B_UR50D-contact-regression.pt")]
    if not all(p.is_file() for p in weights):
        raise ValueError("Required existing local ESM weights missing;no download")
    weights_sha = {p.name: digest(p) for p in weights}
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = dict(manifest_sha256=manifest_sha, source_sha256=digest(__file__), model=MODEL,
                        full_query=True, source_hashes=SOURCE_HASHES, max_length=MAX_QUERY_LENGTH,
                        queue_source_sha256=QUEUE_SOURCE_SHA256, checkpoint_files_sha256=weights_sha)
        path = root / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("Changed full-cache contract")
        write_atomic(path, json.dumps(contract, indent=2).encode())
        contract_sha = digest(path)
        old_ready = root / "ready.json"
        if old_ready.exists():
            old_ready.replace(root / ("ready.previous_" + digest(old_ready) + ".json"))
        from protenix.data.esm.compute_esm import compute_ESM_embeddings, load_esm_model
        model, alphabet = load_esm_model(MODEL, local_esm_dir=str(checkpoint))
        records = {}
        for row in rows:
            uid, seq = row["unique_id"], row["sequence"]
            target = cache_path(root, uid)
            try:
                if not target.parent.exists():
                    temp = root / "attempts" / (uid + "_" + str(time.time_ns()))
                    temp.mkdir(parents=True)
                    label = uid + "_1"
                    compute_ESM_embeddings(MODEL, model, alphabet, [label], [seq], str(temp),
                                           toks_per_batch=16384, truncation_seq_length=len(seq))
                    accepted = dict(unique_id=uid, sequence_sha256=row["seq_sha256"], contract_sha256=contract_sha,
                                    **check_tensor(temp / (label + ".pt"), len(seq), torch))
                    write_atomic(temp / "cache_receipt.json", json.dumps(accepted, indent=2).encode())
                    target.parent.parent.mkdir(parents=True, exist_ok=True)
                    temp.replace(target.parent)
                actual = dict(unique_id=uid, sequence_sha256=row["seq_sha256"], contract_sha256=contract_sha,
                              **check_tensor(target, len(seq), torch))
                if json.loads((target.parent / "cache_receipt.json").read_text()) != actual:
                    raise ValueError("Unverified or changed cached tensor provenance")
                records[uid] = actual
            except Exception as error:
                failure = dict(uid=uid, length=len(seq), error=repr(error), time=time.time(),
                               OOM="out of memory" in str(error).lower(), query_shortened=False)
                directory = root / "failures"; directory.mkdir(exist_ok=True)
                write_atomic(directory / (uid + "_" + str(time.time_ns()) + ".json"), json.dumps(failure, indent=2).encode())
                raise
            print(json.dumps(dict(event="full_ESM_accepted", uid=uid, length=len(seq))), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if digest(manifest) != manifest_sha or any(digest(p) != weights_sha[p.name] for p in weights):
            raise ValueError("Manifest or ESM weights changed during full-query extraction")
        runtime()
        ready = dict(status="complete", **contract, records=records, queries=len(rows),
                     cache_contract_sha256=contract_sha, truncation=False,
                     model_weights_changed=False, packages_changed=False)
        write_atomic(root / "ready.json", json.dumps(ready, indent=2).encode())
        return ready


def input_rows(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pred"])
    for key in ("input", "out_dir", "model_name", "use_default_params", "use_msa", "sample", "seeds",
                "dtype", "trimul_kernel", "triatt_kernel", "enable_fusion"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    expected = dict(model_name="protenix_mini_esm_v0.5.0", use_default_params="true", use_msa="false",
                    sample="1", seeds="101", dtype="bf16", trimul_kernel="torch", triatt_kernel="torch", enable_fusion="false")
    if any(getattr(args, k) != v for k, v in expected.items()):
        raise ValueError("Changed historical folding parameters")
    raw = Path(args.input).resolve(strict=True).read_bytes()
    values = json.loads(raw)
    rows = {}
    if not isinstance(values, list) or not values:
        raise ValueError("Empty or invalid folding input")
    for value in values:
        if (set(value) != {"name", "sequences"} or len(value["sequences"]) != 1
                or set(value["sequences"][0]) != {"proteinChain"}):
            raise ValueError("Only single-chain protein inputs allowed")
        protein = value["sequences"][0]["proteinChain"]
        seq = protein["sequence"]; uid = value["name"]
        if (set(protein) != {"sequence", "count", "id"} or protein.get("count") != 1 or protein.get("id") != ["A"] or uid in rows
                or not 50 <= len(seq) <= MAX_QUERY_LENGTH or set(seq) - set("ACDEFGHIKLMNPQRSTVWYX")
                or uid != "useq_" + hashlib.sha256(seq.encode("ascii")).hexdigest()):
            raise ValueError("Wrong query identity or length;no remapping or truncation")
        rows[uid] = seq
    return args, rows, hashlib.sha256(raw).hexdigest()


def fold(argv):
    args, rows, input_sha = input_rows(argv)
    root = Path(os.environ["PPC_FULL_ESM_ROOT"]).resolve(strict=True)
    ready_path = root / "ready.json"
    ready_bytes = ready_path.read_bytes()
    ready_sha = os.environ["PPC_FULL_ESM_READY_SHA256"]
    if hashlib.sha256(ready_bytes).hexdigest() != ready_sha:
        raise ValueError("Unpinned full-query cache readiness")
    ready = json.loads(ready_bytes)
    if (ready["status"] != "complete" or not ready["full_query"] or ready["truncation"]
            or ready["source_sha256"] != digest(__file__) or ready["source_hashes"] != SOURCE_HASHES
            or ready["queue_source_sha256"] != QUEUE_SOURCE_SHA256
            or digest(root / "contract.json") != ready["cache_contract_sha256"]):
        raise ValueError("Invalid full-query cache contract")
    torch = runtime()
    output = Path(args.out_dir).resolve()
    staged = output.with_name(output.name + ".full_ESM_unaccepted")
    if output.exists() or staged.exists():
        raise ValueError("Fresh unpublished output directory required")
    paths = []
    for uid, seq in rows.items():
        record = ready["records"][uid]
        if (record["unique_id"] != uid or record["sequence_sha256"] != uid[5:] or record["length"] != len(seq)
                or record["contract_sha256"] != ready["cache_contract_sha256"]):
            raise ValueError("Wrong cached query")
        source = cache_path(root, uid)
        actual = check_tensor(source, len(seq), torch)
        if actual != {k: record[k] for k in ("dtype", "sha256", "length")}:
            raise ValueError("Full-query cache changed")
        if json.loads((source.parent / "cache_receipt.json").read_text()) != record:
            raise ValueError("Cached tensor provenance changed")
        destination = cache_path(Path.cwd() / "esm_embeddings", uid)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("Fresh attempt required;do not reuse an unverified work cache")
        shutil.copyfile(source, destination)
        if digest(destination) != record["sha256"]:
            raise ValueError("Full-query cache copy changed")
        paths.append((destination, record["sha256"]))
    try:
        from runner.batch_inference import protenix_cli
        sdk_argv = list(argv)
        sdk_argv[sdk_argv.index("--out_dir") + 1] = str(staged)
        protenix_cli.main(args=sdk_argv, standalone_mode=False)
        # The stock SDK skips already present tensors, even when its converter has a 4094 cap.
        if (digest(args.input) != input_sha or digest(ready_path) != ready_sha
                or any(digest(p) != h for p, h in paths)):
            raise ValueError("Input,readiness or complete ESM cache changed during stock CLI inference")
        errors = Path.cwd() / "esm_embeddings/esm_error"
        if any((errors / (uid + "_1.txt")).exists() for uid in rows):
            raise ValueError("Stock SDK recorded ESM input errors")
        runtime()
        staged.replace(output)
    except (Exception, SystemExit) as error:
        errors = Path.cwd() / "esm_embeddings/esm_error"; errors.mkdir(parents=True, exist_ok=True)
        for uid in rows:
            path = errors / (uid + "_1.txt")
            if not path.exists():
                write_atomic(path, ("Full-query adapter rejected unpublished outputs: " + repr(error)).encode())
        raise
    print(json.dumps(dict(event="stock_cli_full_ESM_completed", queries=len(rows), packages_modified=False)), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["prepare"])
        parser.add_argument("--manifest", required=True, type=Path)
        parser.add_argument("--root", required=True, type=Path)
        args = parser.parse_args()
        prepare(args.manifest, args.root)
    else:
        fold(sys.argv[1:])


if __name__ == "__main__":
    main()
