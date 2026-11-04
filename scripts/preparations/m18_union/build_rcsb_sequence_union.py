"""Build observed-sequence PPIS unions from original RCSB assembly contact tags."""

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np


CONTRACT = "observed-polymer-heavy-atom-5p5-assembly-v2"
AA = set("ACDEFGHIKLMNPQRSTVWYBXZ")
_RAW_LENGTHS, _NATIVE = {}, set()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize(raw_lengths, native):
    global _RAW_LENGTHS, _NATIVE
    _RAW_LENGTHS, _NATIVE = raw_lengths, native


def read_inputs(raw_path, native_path, chain_map, expected):
    raw, native, grouped = {}, {}, defaultdict(dict)
    with Path(raw_path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seq, digest, uid = row["sequence"], row["seq_sha256"], row["unique_id"]
            if (not seq or set(seq) - AA or len(seq) != int(row["n_residues"])
                    or hashlib.sha256(seq.encode("ascii")).hexdigest() != digest
                    or uid != "useq_" + digest or uid in raw):
                raise ValueError(f"Invalid/duplicate raw sequence: {uid}")
            raw[uid] = seq
    with Path(native_path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            uid = row["unique_id"]
            if (uid in native or uid not in raw or row["seq_sha256"] != uid[5:]
                    or int(row["length"]) != len(raw[uid]) or row["split"] not in {"train", "val", "test"}
                    or not row["component_id"]):
                raise ValueError(f"Invalid/duplicate native sequence: {uid}")
            native[uid] = row
    if len(native) != expected or set(native) != {uid for uid, seq in raw.items() if len(seq) >= 50}:
        raise ValueError("Native universe must equal all snapshot sequences>=50,without feature filtering")
    map_rows = 0
    with Path(chain_map).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pdb, uid, cid = row["pdb_id"], row["unique_id"], row["chain_instance_id"]
            if (not pdb.isascii() or not pdb.isalnum() or pdb != pdb.lower()
                    or uid not in raw or row["seq_sha256"] != uid[5:]
                    or Path(row["tag_path"]).name != pdb + ".json.gz"
                    or cid in grouped[pdb]):
                raise ValueError(f"Invalid/duplicate snapshot chain: {pdb}/{cid}")
            grouped[pdb][cid] = (uid, row["assembly_id"], row["source_label_asym_id"])
            map_rows += 1
    grouped = {pdb: rows for pdb, rows in grouped.items() if any(v[0] in native for v in rows.values())}
    represented = {uid for rows in grouped.values() for uid, _, _ in rows.values() if uid in native}
    if represented != set(native):
        raise ValueError("Native queries missing from original full snapshot chain map")
    return raw, native, grouped, map_rows


def summarize_entry(entry, expected, raw_lengths, native):
    if entry.get("contract") != CONTRACT or entry.get("model_policy") != "first_model":
        raise ValueError("Wrong observed-heavy-atom/assembly/model contract")
    instances = {x["chain_instance_id"]: x for x in entry["chains"]}
    if len(instances) != len(entry["chains"]):
        raise ValueError("Duplicate assembly chain-instance ID")
    sources, maps = {}, []
    for asym, source in entry["sources"].items():
        seq, mapping = source["sequence"], source["residue_map"]
        digest = hashlib.sha256(seq.encode("ascii")).hexdigest()
        if (digest != source["seq_sha256"] or set(seq) - AA or len(seq) != len(mapping)
                or [r["index"] for r in mapping] != list(range(len(seq)))
                or any(type(r["label_seq_id"]) is not int or not isinstance(r["insertion_code"], str)
                       for r in mapping)
                or any(a["label_seq_id"] >= b["label_seq_id"] for a, b in zip(mapping, mapping[1:]))):
            raise ValueError(f"Invalid observed source sequence/residue map: {asym}")
        uid = "useq_" + digest
        sources[asym] = (uid, len(seq), mapping)
    for cid, (uid, assembly, asym) in expected.items():
        instance = instances.get(cid)
        if (not instance or instance["assembly_id"] != assembly or instance["source_label_asym_id"] != asym
                or asym not in sources or sources[asym][0] != uid
                or raw_lengths.get(uid) != sources[asym][1]):
            raise ValueError(f"Snapshot chain/source mismatch: {cid}")
    positives = {cid: set() for cid in expected}
    used, seen_pairs = set(), set()
    short_sides, outside_native_sides = 0, 0
    for pair in entry["pairs"]:
        left, right = pair["left"], pair["right"]
        key = tuple(sorted((left, right)))
        if (left == right or key in seen_pairs or left not in expected or right not in expected
                or instances[left]["assembly_id"] != instances[right]["assembly_id"]):
            raise ValueError("Duplicate,self,cross-assembly or out-of-snapshot pair")
        seen_pairs.add(key)
        l_uid, r_uid = expected[left][0], expected[right][0]
        l_len, r_len = raw_lengths[l_uid], raw_lengths[r_uid]
        ls, rs, previous = set(), set(), None
        for contact in pair["residue_contacts"]:
            if (len(contact) != 2 or any(type(x) is not int for x in contact)
                    or not 0 <= contact[0] < l_len or not 0 <= contact[1] < r_len):
                raise ValueError("Out-of-range or noninteger observed residue index")
            point = tuple(contact)
            if previous is not None and point <= previous:
                raise ValueError("Contact pairs must be sorted and unique")
            previous = point
            ls.add(point[0]); rs.add(point[1])
        if not ls or pair["left_positive"] != sorted(ls) or pair["right_positive"] != sorted(rs):
            raise ValueError("Positive masks do not reproduce original residue contacts")
        used.update((left, right)); positives[left].update(ls); positives[right].update(rs)
        for uid, partner_uid, partner_len in ((l_uid, r_uid, r_len), (r_uid, l_uid, l_len)):
            if uid in native:
                short_sides += partner_len < 50
                outside_native_sides += partner_uid not in native
    if used != set(expected):
        raise ValueError("Raw contact-participating contexts differ from snapshot map")
    updates, contexts, assemblies = defaultdict(set), [], defaultdict(set)
    for cid, (uid, assembly, asym) in sorted(expected.items()):
        if uid not in native:
            continue
        updates[uid].update(positives[cid]); assemblies[uid].add(assembly)
        mask = np.zeros(raw_lengths[uid], dtype=np.uint8)
        mask[list(positives[cid])] = 1
        contexts.append((cid, assembly, asym, uid, len(positives[cid]), hashlib.sha256(mask.tobytes()).hexdigest()))
    for asym, (uid, length, mapping) in sorted(sources.items()):
        if uid in updates:
            maps.append((asym, uid, length,
                         hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         sum(bool(row["insertion_code"]) for row in mapping),
                         sum(b["label_seq_id"] != a["label_seq_id"] + 1 for a, b in zip(mapping, mapping[1:]))))
    return dict(updates={u: sorted(v) for u, v in updates.items()}, contexts=contexts, source_maps=maps,
                assemblies={u: len(v) for u, v in assemblies.items()}, short_partner_sides=short_sides,
                outside_native_partner_sides=outside_native_sides, raw_pairs=len(entry["pairs"]),
                coordinate_models=entry["coordinate_models"])


def process_file(job):
    pdb, path, expected = job
    path = Path(path)
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Raw tag changed while reading")
        entry = json.loads(gzip.decompress(raw))
        if entry["pdb_id"] != pdb:
            raise ValueError("Wrong tag entry")
        result = summarize_entry(entry, expected, _RAW_LENGTHS, _NATIVE)
    except Exception as error:
        raise ValueError(f"{pdb} ({path}): {error}") from error
    return dict(result, pdb=pdb, path=str(path), tag_sha256=hashlib.sha256(raw).hexdigest(),
                source_bytes=len(raw), source_mtime_ns=after.st_mtime_ns)


def bounded_results(pool, jobs, window):
    iterator = iter(jobs)
    pending = deque()
    for _ in range(window):
        job = next(iterator, None)
        if job is None:
            break
        pending.append(pool.submit(process_file, job))
    while pending:
        result = pending.popleft().result()
        job = next(iterator, None)
        if job is not None:
            pending.append(pool.submit(process_file, job))
        yield result


def build(args):
    args = argparse.Namespace(**vars(args))
    for name in ("raw_sequences", "native_split", "chain_map", "tags_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    inputs = {str(p): sha(p) for p in (args.raw_sequences, args.native_split, args.chain_map)}
    raw, native, groups, map_rows = read_inputs(args.raw_sequences, args.native_split, args.chain_map,
                                              args.expected_sequences)
    pdbs = sorted(groups)
    if args.limit_entries:
        pdbs = pdbs[:args.limit_entries]
    ids = sorted({u for p in pdbs for u, _, _ in groups[p].values() if u in native})
    offsets = {}; total = 0
    for uid in ids:
        offsets[uid] = total; total += len(raw[uid])
    output = args.output.resolve()
    if output == args.tags_root.resolve() or args.tags_root.resolve() in output.parents:
        raise ValueError("Output cannot be inside original tag directory")
    output.mkdir(parents=True, exist_ok=False)
    labels = np.lib.format.open_memmap(output / "labels.npy", mode="w+", dtype=np.uint8, shape=(total,))
    labels[:] = 0
    context_counts, pdb_counts, assembly_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    masks = defaultdict(set); completed = 0; short_sides = 0; outside_sides = 0
    started = time.monotonic()
    jobs = [(pdb, str(args.tags_root / pdb[1:3] / (pdb + ".json.gz")), groups[pdb]) for pdb in pdbs]
    try:
        with (output / "contexts.tsv").open("x", newline="") as cf, \
                (output / "source_maps.tsv").open("x", newline="") as sf, \
                (output / "entry_audit.tsv").open("x", newline="") as ef:
            cw, sw, ew = (csv.writer(f, delimiter="\t") for f in (cf, sf, ef))
            cw.writerow(["pdb_id", "chain_instance_id", "assembly_id", "source_label_asym_id", "unique_id", "positive_count", "mask_sha256"])
            sw.writerow(["pdb_id", "source_label_asym_id", "unique_id", "length", "residue_map_sha256", "insertion_codes", "internal_gaps", "tag_path", "tag_sha256"])
            ew.writerow(["pdb_id", "tag_path", "tag_sha256", "bytes", "mtime_ns", "raw_pairs", "coordinate_models"])
            initialize({uid: len(s) for uid, s in raw.items()}, set(native))
            pool = None
            if args.workers == 1:
                results = map(process_file, jobs)
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                           initializer=initialize, initargs=(_RAW_LENGTHS, _NATIVE))
                results = bounded_results(pool, jobs, 2 * args.workers)
            try:
                for result in results:
                    for uid, positive in result["updates"].items():
                        labels[offsets[uid] + np.asarray(positive, dtype=np.int64)] = 1
                        pdb_counts[uid] += 1; assembly_counts[uid] += result["assemblies"][uid]
                    for row in result["contexts"]:
                        uid = row[3]; context_counts[uid] += 1; masks[uid].add(row[5])
                        cw.writerow([result["pdb"], *row])
                    for row in result["source_maps"]:
                        sw.writerow([result["pdb"], *row, result["path"], result["tag_sha256"]])
                    ew.writerow([result[k] for k in ("pdb", "path", "tag_sha256", "source_bytes", "source_mtime_ns", "raw_pairs", "coordinate_models")])
                    completed += 1; short_sides += result["short_partner_sides"]; outside_sides += result["outside_native_partner_sides"]
                    if completed % 1000 == 0:
                        print(json.dumps(dict(entries=completed, expected_entries=len(jobs),
                                              seconds=time.monotonic()-started)), flush=True)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
        labels.flush()
        if set(pdb_counts) != set(ids) or (not args.limit_entries and set(ids) != set(native)):
            raise ValueError("Incomplete exact-sequence union coverage")
        all_positive = 0; positive_total = 0
        with (output / "queries.tsv").open("x", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["unique_id", "seq_sha256", "seq_sha1", "sequence", "length", "offset", "union_label_sha256", "union_positive", "contact_contexts", "pdb_contexts", "assembly_contexts", "distinct_masks", "native_split", "native_component"])
            for uid in ids:
                sequence = raw[uid]; offset = offsets[uid]; length = len(sequence)
                n = int(labels[offset:offset+length].sum())
                if n == 0:
                    raise ValueError(f"Contact-participating query has an empty union: {uid}")
                all_positive += n == length; positive_total += n
                writer.writerow([uid, uid[5:], hashlib.sha1(sequence.encode()).hexdigest(), sequence,
                                 length, offset, hashlib.sha256(labels[offset:offset+length].tobytes()).hexdigest(),
                                 n, context_counts[uid], pdb_counts[uid],
                                 assembly_counts[uid], len(masks[uid]), native[uid]["split"], native[uid]["component_id"]])
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise ValueError("Sequence/split/snapshot mapping input changed during build")
        summary = dict(status="smoke_complete" if args.limit_entries else "complete", contract=CONTRACT,
                       task="single-chain exact-sequence all-partner ever-observed binary union", native_queries=len(native),
                       queries=len(ids), residues=total, union_positive=positive_total, all_positive_queries=all_positive,
                       contact_contexts=sum(context_counts.values()), source_entries=completed,
                       raw_snapshot_chain_map_rows=map_rows, short_partner_directional_contributions=short_sides,
                       outside_native_partner_directional_contributions=outside_sides, partner_length_or_feature_filter=False,
                       contexts_scope="Contact-participating snapshot contexts;adding zero masks cannot change a binary OR",
                       residue_axis="Original observed sequence index,including insertion-coded residues;no gap filling",
                       layout="labels.npy uint8 CSR;queries.tsv gives offset,length,sequence and both sequence hashes",
                       normalization_fitted=False, model_trained=False, final_split_changed=False,
                       heldout_labels_materialized_as_dataset_creation=True, labels_used_for_model_selection=False,
                       inputs_sha256=inputs, source_code_sha256=sha(__file__),
                       files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                       seconds=time.monotonic()-started)
        name = "smoke_summary.json" if args.limit_entries else "summary.json"
        temp = output / (name + ".tmp")
        with temp.open("x") as handle:
            handle.write(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        temp.replace(output/name)
    except Exception as error:
        labels.flush()
        (output/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__, error=str(error),
                         completed_entries=completed, expected_entries=len(jobs), formal_ready=False), indent=2)+"\n")
        raise
    try:
        print(json.dumps(summary), flush=True)
    except OSError:
        pass
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-sequences", "native-split", "chain-map", "tags-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-entries", type=int, default=0, help="Smoke only;never publishes formal summary.json")
    args = parser.parse_args()
    if args.workers < 1 or args.limit_entries < 0 or args.expected_sequences < 1:
        raise ValueError("Invalid worker/limit/universe count")
    build(args)


if __name__ == "__main__":
    main()
