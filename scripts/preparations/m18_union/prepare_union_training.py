"""Build independent M18 training inputs from exact-sequence union and OF3 monomers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(source):
    return hashlib.sha256(json.dumps({p.name: sha(p) for p in sorted(Path(source).glob("*.py"))},
                                     sort_keys=True).encode()).hexdigest()


def preparation_id():
    return os.environ.get("PREP_RUN_ID") or os.environ.get("SLURM_JOB_ID")


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def load_fasta(path):
    result = {}
    uid = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            uid = line[1:].split()[0]
            if uid in result:
                raise ValueError("Duplicate FASTA id")
            result[uid] = ""
        elif line.strip():
            if uid is None:
                raise ValueError("Missing FASTA header")
            result[uid] += line.strip()
    return result


def source_rows(dataset):
    summary = json.loads((dataset / "summary.json").read_text())
    for name in ("sequence_groups.tsv", "unique_sequences.fasta", "train.npz", "val.npz", "test.npz"):
        if sha(dataset / name) != summary["artifacts_sha256"][name]:
            raise ValueError("Frozen union artifact changed: " + name)
    for key in ("cross_split_exact_sequences", "cross_split_si30_components", "cross_split_pdbs"):
        if summary["split_audit"][key] != 0:
            raise ValueError("Source split leakage: " + key)
    with (dataset / "sequence_groups.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    sequences = load_fasta(dataset / "unique_sequences.fasta")
    if len(rows) != 6667 or len(sequences) != 6667:
        raise ValueError("Wrong exact sequence universe")
    for row in rows:
        uid = row["union_seq_id"]
        seq = sequences[uid]
        if hashlib.sha1(seq.encode()).hexdigest() != row["seq_sha1"] or len(seq) != int(row["length"]):
            raise ValueError("Sequence hash/length mismatch: " + uid)
    return rows, sequences


def extract_one(job):
    import numpy as np
    import torch
    uid, sequence, source, selected, output, fingerprint = job
    torch.set_num_threads(1)
    sys.path.insert(0, source)
    from protein_features_v4 import extract_complete_features_v4

    path = Path(selected) / "pdb" / (uid + ".pdb")
    target = Path(output) / "raw" / (uid + ".pt")
    pdb_sha = sha(path)
    if target.exists():
        data = torch.load(target, map_location="cpu", weights_only=False)
        if (data["source_pdb_sha256"] != pdb_sha or data["sequence"] != sequence
                or data.get("extractor_sha256") != fingerprint):
            raise ValueError("Stale raw structure feature: " + uid)
        return {"uid": uid, "status": "reused"}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = extract_complete_features_v4(path, chain_id="A", mode="chain")
    from Bio.SeqUtils import seq1
    observed = "".join(seq1(name) for name in result["residue_names"])
    raw = result["spatial_scalar_raw_features"]
    if observed != sequence or raw.shape != (len(sequence), 89):
        raise ValueError("Extracted sequence/shape mismatch: " + uid)
    if not torch.isfinite(raw).all():
        raise ValueError("Nonfinite structure descriptors: " + uid)
    messages = sorted({str(w.message) for w in captured})
    failures = [msg for msg in messages if "failed" in msg.lower() or "失败" in msg]
    if failures:
        raise ValueError("Feature backend failure: " + repr(failures))
    if not float(raw[:, :6].abs().sum()) > 0:
        raise ValueError("Empty SASA features: " + uid)
    result.update(sequence=sequence, source_pdb_sha256=pdb_sha, source_pdb_path=str(path),
                  feature_warnings=messages, coordinate_policy="exact_query_chain_A_1based",
                  extractor_sha256=fingerprint)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp.{os.getpid()}")
    torch.save(result, tmp)
    tmp.replace(target)
    return {"uid": uid, "status": "ok", "warnings": messages}


def extract(args):
    if not shutil.which("mkdssp"):
        raise ValueError("mkdssp not on PATH; refusing coil-only fallback")
    rows, sequences = source_rows(args.dataset)
    selection = json.loads((args.selected / "selection_summary.json").read_text())
    if not selection["complete"] or selection["labels_used"]:
        raise ValueError("Selection not complete/label independent")
    with (args.selected / "selection.tsv").open() as handle:
        selected = {r["union_seq_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    if set(selected) != set(sequences):
        raise ValueError("Selection differs from union dataset")
    jobs = []
    fingerprint = source_fingerprint(args.complete_source)
    for row in rows[:args.max or None]:
        uid = row["union_seq_id"]
        path = args.selected / "pdb" / (uid + ".pdb")
        if sha(path) != selected[uid]["pdb_sha256"]:
            raise ValueError("Selected PDB hash changed: " + uid)
        jobs.append((uid, sequences[uid], str(args.complete_source), str(args.selected), str(args.output), fingerprint))
    start = time.monotonic()
    errors, results = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"uid": uid, "error": repr(exc)})
            if (len(results) + len(errors)) % 100 == 0:
                print(json.dumps({"done": len(results), "errors": len(errors), "total": len(jobs),
                                  "elapsed_s": time.monotonic() - start}), flush=True)
    summary = {"expected": len(jobs), "complete": len(results), "errors": errors,
               "prepare_job_id": preparation_id(), "extractor_sha256": fingerprint,
               "source_sha256": {p.name: sha(p) for p in args.complete_source.glob("*.py")},
               "selection_sha256": selection["selection_sha256"], "results": results,
               "elapsed_s": time.monotonic() - start}
    dump(args.output / "extraction_summary.json", summary)
    if errors:
        raise RuntimeError("Incomplete structure feature extraction")


def pack(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.complete_source))
    from protein_features_v4 import fit_normalization_stats, normalize_spatial_scalar_features

    rows, sequences = source_rows(args.dataset)
    root = args.output
    if (root / "ready.json").exists():
        raise ValueError("Refusing to overwrite ready training bundle")
    extraction = json.loads((root / "extraction_summary.json").read_text())
    fingerprint = source_fingerprint(args.complete_source)
    if (extraction["complete"] != len(rows) or extraction["expected"] != len(rows) or extraction["errors"]
            or extraction["extractor_sha256"] != fingerprint
            or extraction["prepare_job_id"] != preparation_id()
            or {r["uid"] for r in extraction["results"]} != set(sequences)):
        raise ValueError("Extraction receipt is incomplete or from another source/job")
    train_rows = [r for r in rows if r["split"] == "train"]
    raw = [torch.load(root / "raw" / (r["union_seq_id"] + ".pt"), map_location="cpu",
                      weights_only=False)["spatial_scalar_raw_features"].numpy() for r in train_rows]
    first = torch.load(root / "raw" / (rows[0]["union_seq_id"] + ".pt"), map_location="cpu", weights_only=False)
    names = first["spatial_scalar_feature_names"]
    stats = fit_normalization_stats(raw, names)
    stats.update(fit_split="train", n_sequences=len(train_rows), n_residues=sum(map(len, raw)),
                 train_ids_sha256=hashlib.sha256("\n".join(r["union_seq_id"] for r in train_rows).encode()).hexdigest())
    dump(root / "normalization_train_only.json", stats)
    del raw
    labels = {}
    for split in ("train", "val", "test"):
        with np.load(args.dataset / (split + ".npz"), allow_pickle=False) as data:
            for i, uid in enumerate(data["seq_ids"]):
                labels[str(uid)] = data["labels"][data["offsets"][i]:data["offsets"][i + 1]].copy()
    if set(labels) != set(sequences):
        raise ValueError("Packed labels differ from sequences")
    manifest, artifacts = [], []
    for index, row in enumerate(rows):
        uid, seq, rep = row["union_seq_id"], sequences[row["union_seq_id"]], row["representative_chain"]
        length = len(seq)
        y = torch.from_numpy(labels[uid].astype(np.int64))
        if len(y) != length or not set(y.tolist()) <= {0, 1}:
            raise ValueError("Unknown/nonbinary union label: " + uid)
        if hashlib.sha256(labels[uid].astype(np.int8).tobytes()).hexdigest() != row["union_label_sha256"]:
            raise ValueError("Union label changed: " + uid)
        emb_path, contact_path = args.esmc_root / (rep + ".pt"), args.contact_root / (rep + ".pt")
        emb = torch.load(emb_path, map_location="cpu", weights_only=False)
        contact = torch.load(contact_path, map_location="cpu", weights_only=False)
        for payload in (emb, contact):
            if payload["sequence"] != seq:
                raise ValueError("Cached sequence mismatch: " + rep)
        if emb["embeddings"].shape != (length, 3456) or not torch.isfinite(emb["embeddings"]).all():
            raise ValueError("Invalid ESM-C MLC: " + rep)
        if emb["repr_layers"] != [12, 24, 36]:
            raise ValueError("Unexpected ESM-C layers")
        edges, scores = contact["contact_edge_index"], contact["contact_edge_scores"]
        if edges.ndim != 2 or edges.shape[0] != 2 or scores.shape != (edges.shape[1],):
            raise ValueError("Contact graph shape mismatch")
        if edges.numel() and (edges.min() < 0 or edges.max() >= length):
            raise ValueError("Contact residue index outside exact sequence")
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite contact scores")
        raw_path = root / "raw" / (uid + ".pt")
        features = torch.load(raw_path, map_location="cpu", weights_only=False)
        if (features["sequence"] != seq or features["spatial_scalar_feature_names"] != names
                or features.get("extractor_sha256") != fingerprint):
            raise ValueError("Structure feature metadata mismatch")
        x = normalize_spatial_scalar_features(features["spatial_scalar_raw_features"].numpy(), stats)
        x = torch.cat([torch.from_numpy(x), torch.ones(length, 1)], dim=1)
        if x.shape != (length, 90) or not torch.isfinite(x).all():
            raise ValueError("Invalid normalized structure features")
        metadata = {"pdb_id": uid, "feature_n_residues": length, "sequence": seq,
                    "residue_names_1": list(seq), "residue_indices": list(range(1, length + 1)),
                    "insertion_codes": [""] * length, "chain_ids": ["A"] * length}
        payloads = {
            "metadata": (uid + "_esm2.pt", metadata),
            "labels": (uid + "_labels.pt", {**metadata, "residue_names": list(seq), "labels": y,
                                           "cutoff": 5.5, "heavy_atoms_only": True,
                                           "label_contract": "exact_sequence_ever_observed_union_v2"}),
            "structure": (uid + "_seq.pt", {**metadata, "seq_features": x,
                                            "feature_names_with_availability": names + ["pred_struct_available"]}),
            "contact": (uid + "_contact_graph.pt", {"n_residues": length,
                          "contact_edge_index": edges, "contact_edge_scores": scores,
                          "sequence": seq, "source": str(contact_path.resolve())}),
        }
        for kind, (filename, payload) in payloads.items():
            path = root / kind / uid / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
        link = root / "embeddings" / (uid + ".pt")
        link.parent.mkdir(exist_ok=True)
        if link.is_symlink():
            if link.resolve() != emb_path.resolve():
                raise ValueError("Existing embedding link mismatch")
        elif link.exists():
            raise ValueError("Expected symbolic embedding reference")
        else:
            link.symlink_to(emb_path.resolve())
        manifest.append({"pdb_id": uid, "split": row["split"], "n_residues": length,
                         "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
                         "representative_chain": rep})
        artifacts.append({"uid": uid, "embedding_source": str(emb_path.resolve()),
                          "embedding_sha256": sha(emb_path), "contact_source_sha256": sha(contact_path),
                          "raw_structure_sha256": sha(raw_path), "label_sha256": row["union_label_sha256"]})
        if (index + 1) % 250 == 0:
            print(json.dumps({"packaged": index + 1, "total": len(rows)}), flush=True)
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    for split in ("train", "val", "test"):
        path = root / "split" / (split + "_ids.txt")
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(r["pdb_id"] + "\n" for r in manifest if r["split"] == split))
    dump(root / "asset_hashes.json", artifacts)
    dump(root / "ready.json", {"n_sequences": len(rows), "split_sizes": {
        s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "prepare_job_id": preparation_id(), "run_id": "m18_union_of3_20261102",
        "extractor_sha256": fingerprint, "extraction_summary_sha256": sha(root / "extraction_summary.json"),
        "normalization_sha256": sha(root / "normalization_train_only.json"),
        "manifest_sha256": sha(root / "manifest.csv"), "asset_hashes_sha256": sha(root / "asset_hashes.json"),
        "feature_contract": "ESMC_MLC_12_24_36_3456D+OF3_complete89D+availability+ESM2_contact",
        "labels": "exact sequence union, no context reweighting", "bridge_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "pack"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--complete-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--esmc-root", type=Path)
    parser.add_argument("--contact-root", type=Path)
    args = parser.parse_args()
    {"extract": extract, "pack": pack}[args.command](args)
