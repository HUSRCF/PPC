"""Audit full RCSB union/ESM-C alignment and pack existing PPC loader inputs."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

REVISION = "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c"
CONTRACT = "observed-polymer-heavy-atom-5p5-assembly-v2"
REQUIRED_ARTIFACTS = {"queries.tsv", "labels.npy", "contexts.tsv", "source_maps.tsv", "entry_audit.tsv"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_union(root, expected):
    receipt = json.loads((root / "summary.json").read_text())
    if (receipt["status"] != "complete" or receipt["contract"] != CONTRACT
            or receipt["queries"] != expected or receipt["native_queries"] != expected
            or receipt["partner_length_or_feature_filter"] or receipt["final_split_changed"]):
        raise ValueError("Wrong or incomplete full-universe union receipt")
    if not REQUIRED_ARTIFACTS <= set(receipt["files_sha256"]):
        raise ValueError("Missing mandatory union artifact fingerprints")
    for name, digest in receipt["files_sha256"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Changed union artifact: " + name)
    with (root / "queries.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    if labels.dtype != np.uint8 or labels.shape != (receipt["residues"],) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid union label array")
    ids, components, cursor = set(), {}, 0
    for r in rows:
        uid, seq, digest = r["unique_id"], r["sequence"], r["seq_sha256"]
        n, offset = int(r["length"]), int(r["offset"])
        y = labels[offset:offset+n]
        if (uid in ids or uid != "useq_"+digest or hashlib.sha256(seq.encode()).hexdigest() != digest
                or hashlib.sha1(seq.encode()).hexdigest() != r["seq_sha1"] or len(seq) != n or n < 50
                or offset != cursor or len(y) != n or int(y.sum()) != int(r["union_positive"])
                or sha_bytes(y.tobytes()) != r["union_label_sha256"]
                or r["native_split"] not in {"train", "val", "test"}):
            raise ValueError("Invalid query/label alignment: " + uid)
        component = r["native_component"]
        if not component or components.setdefault(component, r["native_split"]) != r["native_split"]:
            raise ValueError("Cross-split native component: " + component)
        ids.add(uid); cursor += n
    if len(ids) != expected or cursor != len(labels):
        raise ValueError("Incomplete or duplicated CSR query coverage")
    pdb_splits, context_ids = {}, set()
    by_id = {r["unique_id"]: r for r in rows}
    with (root / "contexts.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            uid = r["unique_id"]
            split = by_id[uid]["native_split"]
            if pdb_splits.setdefault(r["pdb_id"], split) != split:
                raise ValueError("Cross-split source PDB: " + r["pdb_id"])
            context_ids.add(uid)
    if context_ids != ids:
        raise ValueError("Missing inverse original-context mapping")
    return rows, receipt


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def verify_embedding_inventory(output):
    with (output / "feature_audit.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            path = Path(row["embedding_source"])
            identity = [int(row[k]) for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]
            link = output / "embeddings" / (row["unique_id"]+".pt")
            if (file_identity(path.stat()) != identity or not link.is_symlink()
                    or link.resolve(strict=True) != path):
                raise ValueError("Embedding source/reference changed during packing: " + row["unique_id"])


def pack_one(job):
    import torch
    torch.set_num_threads(1)
    row, union_root, bank, output = job
    uid, seq, n = row["unique_id"], row["sequence"], int(row["length"])
    source = (Path(bank) / row["seq_sha256"][:2] / (uid + ".pt")).resolve(strict=True)
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    if file_identity(before) != file_identity(after):
        raise ValueError("Feature changed while reading: " + uid)
    data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    x = data["embeddings"]
    if (data["unique_id"] != uid or data["sequence"] != seq
            or data["sequence_sha256"] != row["seq_sha256"] or data["n_residues"] != n
            or data["resolved_revision"] != REVISION or data["repr_layers"] != [12, 24, 36]
            or data["hidden_normalization"] != "final_layernorm" or tuple(x.shape) != (n, 3456)
            or x.dtype != torch.float32 or x.requires_grad or not torch.isfinite(x).all()):
        raise ValueError("Invalid ESM-C content/alignment: " + uid)
    labels = np.load(Path(union_root) / "labels.npy", mmap_mode="r", allow_pickle=False)
    start = int(row["offset"])
    y = np.array(labels[start:start+n], dtype=np.uint8, copy=True)
    if sha_bytes(y.tobytes()) != row["union_label_sha256"]:
        raise ValueError("Union label changed: " + uid)
    # Canonical sequence indices are not author coordinates; inverse maps stay in the union bundle.
    metadata = dict(pdb_id=uid, feature_n_residues=n, sequence=seq,
                    residue_names_1=list(seq), residue_indices=list(range(1, n+1)),
                    insertion_codes=[""]*n, chain_ids=["A"]*n,
                    coordinate_axis="canonical_observed_sequence_index_not_author_numbering")
    root = Path(output)
    payloads = {"metadata": (uid+"_esm2.pt", metadata),
                "labels": (uid+"_labels.pt", dict(metadata, residue_names=list(seq), labels=torch.from_numpy(y.astype(np.int64)),
                           cutoff=5.5, heavy_atoms_only=True, label_contract="exact_sequence_ever_observed_union_rcsb_v1"))}
    hashes = []
    for kind, (name, payload) in payloads.items():
        path = root / kind / name
        temp = path.with_suffix(".tmp")
        with temp.open("xb") as f:
            torch.save(payload, f)
        temp.replace(path)
        hashes.append(sha(path))
    (root / "embeddings" / (uid+".pt")).symlink_to(source)
    return [uid, str(source), sha_bytes(raw), *file_identity(before), *hashes]


def bounded(pool, jobs, window):
    pending, iterator = deque(), iter(jobs)
    for _ in range(window):
        job = next(iterator, None)
        if job is None: break
        pending.append(pool.submit(pack_one, job))
    while pending:
        value = pending.popleft().result()
        job = next(iterator, None)
        if job is not None: pending.append(pool.submit(pack_one, job))
        yield value


def pack(args):
    union, bank, output = args.union.resolve(strict=True), args.esmc_bank.resolve(strict=True), args.output.resolve()
    if output == union or union in output.parents or output == bank or bank in output.parents:
        raise ValueError("Output must be independent of source bundles")
    initial = sha(union / "summary.json")
    rows, source_receipt = read_union(union, args.expected_sequences)
    if args.smoke_count:
        ordered = sorted(rows, key=lambda r: (int(r["length"]), r["unique_id"]))
        indices = sorted(set(np.linspace(0, len(rows)-1, min(args.smoke_count, len(rows)), dtype=int).tolist()))
        rows = [ordered[i] for i in indices]
    output.mkdir(parents=True, exist_ok=False)
    for name in ("metadata", "labels", "embeddings", "split"):
        (output / name).mkdir()
    jobs = [(r, str(union), str(bank), str(output)) for r in rows]
    started = time.monotonic(); completed = 0; pool = None
    try:
        with (output / "feature_audit.tsv").open("x", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["unique_id", "embedding_source", "embedding_sha256", "device", "inode", "bytes", "mtime_ns", "ctime_ns", "metadata_sha256", "label_payload_sha256"])
            if args.workers == 1:
                results = map(pack_one, jobs)
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
                results = bounded(pool, jobs, 2*args.workers)
            for result in results:
                writer.writerow(result); completed += 1
                if completed % 1000 == 0:
                    print(json.dumps(dict(completed=completed, queries=len(rows), seconds=time.monotonic()-started)), flush=True)
        if pool:
            pool.shutdown(wait=True); pool = None
        if initial != sha(union / "summary.json"):
            raise ValueError("Union summary changed during packing")
        for name, digest in source_receipt["files_sha256"].items():
            if sha(union / name) != digest: raise ValueError("Union artifact changed: "+name)
        verify_embedding_inventory(output)
        with (output / "manifest.csv").open("x", newline="") as f:
            writer = csv.writer(f); writer.writerow(["pdb_id", "split", "n_residues", "n_positive", "n_negative"])
            for r in rows:
                writer.writerow([r["unique_id"], r["native_split"], r["length"], r["union_positive"], int(r["length"])-int(r["union_positive"])])
        for split in ("train", "val", "test"):
            with (output / "split" / (split+"_ids.txt")).open("x") as f:
                f.write("".join(r["unique_id"]+"\n" for r in rows if r["native_split"]==split))
        receipt = dict(status="smoke_complete" if args.smoke_count else "complete", queries=completed,
                       split_counts=dict(Counter(r["native_split"] for r in rows)),
                       residues=sum(int(r["length"]) for r in rows), union_summary_sha256=initial,
                       source_code_sha256=sha(__file__), original_union=str(union), esmc_bank=str(bank),
                       coordinate_axis="canonical observed sequence;source author/insertion maps preserved in original union",
                       input_mode="ESM-C-only preparation;no structure or MSA fabricated",
                       independent_full_feature_content_audit=True, embedding_values_modified=False,
                       embedding_bank_immutability_required_through_training=True,
                       embedding_inventory_rechecked_before_publication=True,
                       inventory_check="Device/inode/size/mtime/ctime and resolved links;audited content SHA-256 recorded per source",
                       normalization_fitted=False, model_trained=False, test_evaluated=False,
                       files_sha256={str(p.relative_to(output)): sha(p) for pattern in ("*.tsv", "*.csv", "split/*.txt") for p in output.glob(pattern)},
                       seconds=time.monotonic()-started)
        name = "smoke_ready.json" if args.smoke_count else "ready.json"
        temp = output / (name+".tmp"); temp.write_text(json.dumps(receipt, indent=2)+"\n"); temp.replace(output/name)
    except Exception as error:
        if pool: pool.shutdown(wait=True, cancel_futures=True)
        (output/"failure.json").write_text(json.dumps(dict(error=repr(error),completed=completed,formal_ready=False), indent=2)+"\n")
        raise
    try: print(json.dumps(receipt), flush=True)
    except OSError: pass
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("union", "esmc-bank", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, default=251267)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke-count", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.smoke_count < 0: raise ValueError("Invalid workers/smoke count")
    pack(args)


if __name__ == "__main__": main()
