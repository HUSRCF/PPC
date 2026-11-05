"""Catalog existing RCSB monomer sources without declaring content acceptance."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_queries(path, digest, expected):
    if sha(path) != digest:
        raise ValueError("Native query fingerprint changed")
    with Path(path).open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    queries = {}
    for row in rows:
        uid, seq = row["unique_id"], row["sequence"]
        if (uid in queries or len(seq) != int(row["length"]) or len(seq) < 50
                or hashlib.sha256(seq.encode("ascii")).hexdigest() != row["seq_sha256"]
                or uid != "useq_"+row["seq_sha256"] or row["native_split"] not in {"train", "val", "test"}):
            raise ValueError("Invalid native query: "+uid)
        queries[uid] = row
    if sha(path) != digest:
        raise ValueError("Native query fingerprint changed during reading")
    if len(queries) != expected:
        raise ValueError("Wrong native universe size")
    return queries


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if not line.endswith("\n"):
                raise ValueError("Incomplete source ledger: "+str(path))
            yield json.loads(line)


def checked_cif(path, uid):
    p = Path(path)
    if (not p.is_absolute() or ".." in p.parts or any(c in str(p) for c in "\n\r\0")
            or p.name not in {uid+".cif", uid+"_sample_0.cif"}):
        raise ValueError("Invalid CIF path for "+uid)
    return p


def build(queries, missing_path, local_index, ledgers, retained, star_root, output, expected_covered):
    source_paths = [missing_path, local_index, *ledgers, *retained]
    initial = {str(p): sha(p) for p in source_paths}
    star_root = Path(star_root)
    if not star_root.is_absolute() or ".." in star_root.parts:
        raise ValueError("Star root must be an absolute canonical path")
    with Path(missing_path).open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    missing = set()
    for row in rows:
        uid = row["unique_id"]
        if uid in missing or uid not in queries or row["sequence"] != queries[uid]["sequence"]:
            raise ValueError("Wrong missing-request universe: "+uid)
        missing.add(uid)
    covered = set(queries)-missing
    if len(covered) != expected_covered:
        raise ValueError("Wrong claimed existing-feature scope")
    local = {}
    foreign = Counter()
    for row in jsonl(local_index):
        uid = row["unique_id"]
        if uid not in covered:
            foreign["local_index"] += 1
            continue
        p = checked_cif(row["source_cif"], uid)
        if uid in local or int(row["n_residues"]) != int(queries[uid]["length"]):
            raise ValueError("Duplicate or mismatched local CIF index: "+uid)
        local[uid] = str(p)
    candidates = defaultdict(set)
    retained_without_length = 0
    for path, require_ok in [*( (p, True) for p in ledgers), *( (p, False) for p in retained)]:
        for row in jsonl(path):
            if require_ok and row.get("status") != "ok":
                continue
            uid = row["unique_id"]
            if uid not in covered:
                foreign["star_index"] += 1
                continue
            if not require_ok and row.get("status", "ok") != "ok":
                raise ValueError("Non-success record in retained index")
            if "n_residues" not in row and not require_ok:
                retained_without_length += 1
            elif int(row["n_residues"]) != int(queries[uid]["length"]):
                raise ValueError("Star query length changed: "+uid)
            p = checked_cif(row["cif"], uid)
            if not p.is_relative_to(star_root):
                raise ValueError("CIF is outside the allowed Star tree")
            candidates[uid].add(str(p))
    unresolved = covered-set(local)-set(candidates)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    transfer = set()
    counts = Counter()
    with (output/"sources.tsv").open("x", newline="") as f:
        fields = ["unique_id", "length", "native_split", "local_cif", "star_candidates_json", "resolution"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader()
        for uid in sorted(covered):
            query = queries[uid]
            resolution = "local_index" if uid in local else "star_candidates" if uid in candidates else "unresolved"
            counts[resolution] += 1
            w.writerow(dict(unique_id=uid, length=query["length"], native_split=query["native_split"],
                            local_cif=local.get(uid, ""), star_candidates_json=json.dumps(sorted(candidates[uid])),
                            resolution=resolution))
            if uid not in local:
                transfer.update(str(Path(p).relative_to(star_root)) for p in candidates[uid])
    (output/"star_files.txt").write_text("".join(p+"\n" for p in sorted(transfer)))
    if {str(p): sha(p) for p in source_paths} != initial:
        raise ValueError("Source index changed while building catalog")
    summary = dict(status="candidate_catalog_complete" if not unresolved else "incomplete",
                   native_queries=len(queries), claimed_existing_features=len(covered), missing_structure_queries=len(missing),
                   counts=dict(counts), transfer_files=len(transfer), unresolved_queries=sorted(unresolved),
                   additional_candidates=sum(max(0, len(candidates[u])-1) for u in covered-set(local)),
                   unknown_index_records_ignored=dict(foreign), star_root=str(star_root),
                   source_files_sha256=initial, retained_records_without_length=retained_without_length,
                   source_code_sha256=sha(__file__), file_existence_checked=False, CIF_content_accepted=False,
                   raw_feature_CIF_equivalence_checked=False, labels_used_for_selection=False, source_files_modified=False,
                   next_gate="Exact query/finite coordinates/confidence and raw-descriptor coordinate equivalence before geometry use",
                   outputs_sha256={p.name:sha(p) for p in (output/"sources.tsv", output/"star_files.txt")})
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--queries-sha256", required=True)
    p.add_argument("--expected-queries", required=True, type=int)
    p.add_argument("--missing", required=True, type=Path)
    p.add_argument("--local-index", required=True, type=Path)
    p.add_argument("--ledger", action="append", type=Path, default=[])
    p.add_argument("--retained", action="append", type=Path, default=[])
    p.add_argument("--star-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-covered", required=True, type=int)
    args = p.parse_args()
    queries = read_queries(args.queries, args.queries_sha256, args.expected_queries)
    result = build(queries, args.missing, args.local_index, args.ledger, args.retained,
                   args.star_root, args.output, args.expected_covered)
    result["native_queries_sha256"] = args.queries_sha256
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "candidate_catalog_complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
