"""Read-only train-feature I/O comparison; never changes a running loader."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import statistics
import tempfile
import time

import torch


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_train_rows(pack, count):
    ids_bytes = (pack / "split/train_ids.txt").read_bytes()
    ids = ids_bytes.decode().splitlines()
    if (len(ids) != len(set(ids)) or count < 2
            or any(re.fullmatch(r"useq_[0-9a-f]{64}", uid) is None for uid in ids)):
        raise ValueError("Unique training IDs and at least two samples required")
    id_set = set(ids)
    manifest = (pack / "manifest.csv").read_bytes()
    rows = []
    for row in csv.DictReader(io.StringIO(manifest.decode())):
        if row["pdb_id"] in id_set:
            if row["split"] != "train":
                raise ValueError("Training-ID/manifest split conflict")
            rows.append((int(row["n_residues"]), row["pdb_id"]))
    if len(rows) != len(ids) or len({uid for _, uid in rows}) != len(ids) or count > len(rows):
        raise ValueError("Incomplete training universe or oversized sample")
    rows.sort()
    selected = [rows[(i * (len(rows) - 1)) // (count - 1)] for i in range(count)]
    return selected, dict(train_ids_sha256=hashlib.sha256(ids_bytes).hexdigest(),
                          manifest_sha256=hashlib.sha256(manifest).hexdigest())


def load_materialized(path, mmap):
    start = time.perf_counter()
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    loaded = time.perf_counter()
    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError("Expected accepted ESM-C embeddings payload")
    value = torch.as_tensor(data["embeddings"], dtype=torch.float32)
    materialized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    end = time.perf_counter()
    if value.dtype != data["embeddings"].dtype or not bit_equal(value, materialized):
        raise ValueError("Feature cast or nonfinite replacement would change source values")
    if materialized.ndim != 2 or materialized.shape[1] != 3456:
        raise ValueError("Wrong feature width")
    return materialized, dict(load_seconds=loaded-start,
                              materialize_seconds=end-loaded, total_seconds=end-start)


def bit_equal(left, right):
    if left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape:
        return False
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def benchmark(pack, count, repeats, stage_parent, source_bank):
    if repeats < 1:
        raise ValueError("Positive repeat count required")
    selected, identity = select_train_rows(pack, count)
    trials, sources = [], []
    modes = [("shared", True), ("shared", False), ("local", True), ("local", False)]
    with tempfile.TemporaryDirectory(prefix="ppc_embedding_io_", dir=stage_parent) as directory:
        stage = Path(directory)
        for index, (length, uid) in enumerate(selected):
            source = pack / "embeddings" / (uid + ".pt")
            resolved = source.resolve(strict=True)
            if not resolved.is_relative_to(source_bank.resolve(strict=True)):
                raise ValueError("Embedding link leaves the explicitly accepted source bank")
            before = resolved.stat()
            reference, initial = load_materialized(source, True)
            if reference.shape[0] != length:
                raise ValueError("Manifest/embedding length mismatch")
            trials.append(dict(uid=uid, mode="shared_initial_mmap", repeat=-1, **initial))
            start = time.perf_counter()
            local = stage / (uid + ".pt")
            shutil.copyfile(source, local)
            copy_seconds = time.perf_counter()-start
            source_hash = file_sha(source)
            copied_hash = file_sha(local)
            if copied_hash != source_hash:
                raise ValueError("Staged bytes do not equal source")
            for repeat in range(repeats):
                offset = (index + repeat) % len(modes)
                for location, mmap in modes[offset:] + modes[:offset]:
                    value, timing = load_materialized(source if location == "shared" else local, mmap)
                    if not bit_equal(value, reference):
                        raise ValueError("Reader/storage modes changed feature values")
                    trials.append(dict(uid=uid, repeat=repeat,
                                       mode=location+("_mmap" if mmap else "_eager"), **timing))
                    del value
            after = source.resolve(strict=True).stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Source identity changed during read-only probe")
            sources.append(dict(uid=uid, length=length, source=str(source.resolve()),
                                source_sha256=source_hash, copied_sha256=copied_hash, bytes=before.st_size,
                                warm_source_copy_seconds=copy_seconds, all_modes_bitwise_equal=True))
            del reference
    summary = {}
    for mode in sorted({x["mode"] for x in trials}):
        times = [x["total_seconds"] for x in trials if x["mode"] == mode]
        summary[mode] = dict(n=len(times), mean_seconds=statistics.mean(times),
                             median_seconds=statistics.median(times), total_seconds=sum(times))
    return dict(status="complete", identity=identity, sample_count=count, repeats=repeats,
                selection="training-only evenly spaced length ranks,including extremes",
                stage_copies_removed=True, source_values_or_dtype_changed=False,
                test_label_payloads_read=False, running_training_changed=False,
                index_scope="Manifest all-split rows include label-count columns;only train IDs,lengths,split used;no label tensor opened",
                limitations=["Single-process component probe,not full DataLoader throughput",
                             "OS cache not cleared;initial read is not guaranteed cold",
                             "Byte verification warms both files before counterbalanced repeats",
                             "Copy timing is warm-source only,not an estimate for full-bank cold staging",
                             "Includes load/materialization only,not metadata or IPC/collation"],
                torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                sources=sources, trials=trials, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stage-parent", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    pack = args.pack_root.resolve(strict=True)
    output = args.output.resolve()
    bank = args.source_bank.resolve(strict=True)
    stage = args.stage_parent.resolve(strict=True)
    if (output.is_relative_to(pack) or output.is_relative_to(bank) or output.exists()
            or stage.is_relative_to(pack) or stage.is_relative_to(bank)):
        raise ValueError("Fresh receipt outside the immutable input bank required")
    torch.set_num_threads(1)
    start = time.time()
    result = benchmark(pack, args.samples, args.repeats, stage, bank)
    result.update(started=start, ended=time.time(), source_sha256=file_sha(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "trials")}), flush=True)


if __name__ == "__main__":
    main()
