"""Audit existing Protenix CIF/raw-feature alignment and cache absolute confidence."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import time

import gemmi
import numpy as np
import torch

from catalog_rcsb_structures import checked_cif, sha


# Frozen raw v4 schema, not the corrected 91D training interface.
RAW89_NAMES = tuple(
    [f"v2_sasa_{i}" for i in range(6)]
    + [f"v2_surface_{i}" for i in range(14)]
    + [f"v2_structural_geometry_{i}" for i in range(26)]
    + [f"v2_enhanced_{i}" for i in range(12)]
    + [f"local_density_raw_{i}" for i in range(4)]
    + ["distance_to_center_raw", "surface_normal_magnitude"]
    + [f"charge_density_raw_{i}" for i in range(4)]
    + [f"hydrophobicity_{i}" for i in range(4)]
    + ["phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin", "omega_cos"]
    + ["nearest_sidechain_dist", "packing_score", "void_ratio"]
    + ["env_aromatic_ratio", "env_hbond_donor_ratio", "env_hbond_acceptor_ratio", "env_gly_pro_ratio"]
    + ["self_metal_binder", "neighbor_metal_binder_6A", "neighbor_metal_binder_10A", "local_anisotropy"]
)


def read_stable(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    value = path.read_bytes()
    after = path.stat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in keys):
        raise ValueError("Source changed during content audit: "+str(path))
    return value, dict(path=str(path), sha256=hashlib.sha256(value).hexdigest(),
                       **{k: getattr(after, k) for k in keys})


def source_paths(row, summary, snapshot):
    uid = row["unique_id"]
    if row["resolution"] == "local_index":
        cif = checked_cif(row["local_cif"], uid)
    elif row["resolution"] == "star_candidates":
        paths = json.loads(row["star_candidates_json"])
        if len(paths) != 1:
            raise ValueError("Ambiguous source requires explicit resolution: "+uid)
        relative = checked_cif(paths[0], uid).relative_to(summary["star_root"])
        cif = Path(snapshot)/relative
    else:
        raise ValueError("Unresolved source: "+uid)
    name = uid+".summary_confidence.json" if cif.name == uid+".cif" else uid+"_summary_confidence_sample_0.json"
    return cif, cif.parent/name


def validate_content(uid, n, cif_path, confidence_path, raw_path):
    cif_bytes, cif_id = read_stable(cif_path)
    confidence_bytes, confidence_id = read_stable(confidence_path)
    raw_bytes, raw_id = read_stable(raw_path)
    structure = gemmi.make_structure_from_block(gemmi.cif.read_string(cif_bytes.decode()).sole_block())
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError("Expected one model and one protein chain")
    residues = list(structure[0][0])
    sequence = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in residues)
    if len(sequence) != n or "useq_"+hashlib.sha256(sequence.encode()).hexdigest() != uid:
        raise ValueError("CIF query length/hash mismatch")
    if any(not r.find_atom("CA", "*") for r in residues):
        raise ValueError("Missing C-alpha")
    if not all(np.isfinite([a.pos.x, a.pos.y, a.pos.z, a.b_iso]).all() for r in residues for a in r):
        raise ValueError("Nonfinite CIF coordinates/confidence")
    ca = np.asarray([[r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y,
                      r.find_atom("CA", "*").pos.z] for r in residues], dtype=np.float32)
    q = np.asarray([r.find_atom("CA", "*").b_iso for r in residues], dtype=np.float32)
    if np.any((q < 0) | (q > 100)):
        raise ValueError("C-alpha confidence is not on the 0--100 scale")
    confidence = json.loads(confidence_bytes)
    global_q = float(confidence["plddt"])
    if not np.isfinite(global_q) or not 0 <= global_q <= 100:
        raise ValueError("Invalid official global pLDDT")
    data = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
    if (data["unique_id"] != uid or data["sequence"] != sequence or data["sequence_sha256"] != uid[5:]
            or data["source"] != "protenix_mini_esm_v0.5.0"
            or data["normalization"] != "raw; train-only fitting deferred"):
        raise ValueError("Wrong raw feature identity/source/normalization")
    shapes = {"spatial_scalar_raw_features": (n, 89), "spatial_scalar_features": (n, 89),
              "physchem_features": (n, 62), "spatial_vector_features": (n, 8, 3), "ca_coords": (n, 3)}
    for key, shape in shapes.items():
        value = data[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Invalid raw tensor: "+key)
    raw = data["spatial_scalar_raw_features"]
    names = data["spatial_scalar_feature_names"]
    if (tuple(names) != RAW89_NAMES
            or not torch.equal(raw, data["spatial_scalar_features"]) or not bool((raw[:, 0] > 0).any())):
        raise ValueError("Unexpected raw scalar schema or zero-SASA fallback")
    delta = float(np.max(np.abs(ca-data["ca_coords"].numpy())))
    if delta > .0011:
        raise ValueError("C-alpha pose differs from cached raw features")
    if len(data["all_atom_names"]) != n or len(data["all_atom_coords"]) != n:
        raise ValueError("Incomplete cached all-atom coordinates")
    atom_delta = 0.
    for residue, names, coordinates in zip(residues, data["all_atom_names"], data["all_atom_coords"]):
        heavy = [a for a in residue if a.element.name not in {"H", "D"}]
        if len({a.name.strip() for a in heavy}) != len(heavy):
            raise ValueError("Duplicate CIF heavy-atom names/alternate conformers")
        atoms = {a.name.strip(): a for a in heavy}
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if (len(set(names)) != len(names) or set(names) != set(atoms)
                or coordinates.shape != (len(names), 3) or not np.isfinite(coordinates).all()):
            raise ValueError("Cached heavy-atom identities differ from CIF")
        reference = np.asarray([[atoms[name].pos.x, atoms[name].pos.y, atoms[name].pos.z] for name in names], dtype=np.float32)
        atom_delta = max(atom_delta, float(np.max(np.abs(coordinates-reference))))
    if atom_delta > .0011:
        raise ValueError("All-atom pose differs from cached raw features")
    return q/100., dict(CA_max_abs_difference=delta, heavy_atom_max_abs_difference=atom_delta,
                       global_plddt=global_q, CA_plddt_min=float(q.min()), CA_plddt_max=float(q.max()),
                       SASA_positive_residues=int((raw[:, 0] > 0).sum()),
                       sources={"CIF": cif_id, "confidence": confidence_id, "raw_feature": raw_id})


def worker(job):
    row, summary, snapshot, bank = job
    torch.set_num_threads(1)
    uid, n = row["unique_id"], int(row["length"])
    cif, confidence = source_paths(row, summary, snapshot)
    result = dict(unique_id=uid, length=n, native_split=row["native_split"])
    if not cif.is_file() or not confidence.is_file():
        return dict(result, status="pending", reason="CIF/confidence has not arrived"), None
    try:
        q, audit = validate_content(uid, n, cif, confidence, Path(bank)/uid[5:7]/(uid+".pt"))
        return dict(result, status="ok", q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), **audit), q
    except Exception as error:
        return dict(result, status="failed", reason=repr(error)), None


def bounded(pool, jobs, window):
    iterator = iter(jobs); pending = deque()
    for _ in range(window):
        job = next(iterator, None)
        if job is None:
            break
        pending.append(pool.submit(worker, job))
    while pending:
        value = pending.popleft().result()
        job = next(iterator, None)
        if job is not None:
            pending.append(pool.submit(worker, job))
        yield value


def audit(args):
    torch.set_num_threads(1)
    code = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("catalog_rcsb_structures.py"))}
    catalog = args.catalog.resolve(strict=True)
    if sha(catalog/"summary.json") != args.catalog_sha256:
        raise ValueError("Catalog fingerprint changed")
    summary = json.loads((catalog/"summary.json").read_text())
    if (summary["status"] != "candidate_catalog_complete" or summary["unresolved_queries"]
            or summary["claimed_existing_features"] != args.expected_existing
            or summary["native_queries"] != args.expected_native
            or sha(catalog/"sources.tsv") != summary["outputs_sha256"]["sources.tsv"]):
        raise ValueError("Wrong native/covered source-catalog contract")
    with (catalog/"sources.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != args.expected_existing or len({r["unique_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicated source rows")
    smoke = args.smoke_ids is not None
    if smoke:
        ids = args.smoke_ids.read_text().splitlines()
        mapping = {r["unique_id"]: r for r in rows}
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(mapping):
            raise ValueError("Invalid explicit smoke IDs")
        rows = [mapping[uid] for uid in ids]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    total = sum(int(r["length"]) for r in rows)
    qbank = np.lib.format.open_memmap(output/"absolute_CA_plddt_div100.npy", mode="w+", dtype=np.float32, shape=(total,))
    qbank[:] = np.nan
    jobs = ((r, {"star_root": summary["star_root"]}, str(args.snapshot), str(args.raw_bank)) for r in rows)
    counts = Counter(); offset = 0; start = time.monotonic()
    with ExitStack() as stack:
        log = stack.enter_context((output/"audit.jsonl").open("x"))
        if args.workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")))
            results = bounded(pool, jobs, 2*args.workers)
        else:
            results = map(worker, jobs)
        for i, (record, q) in enumerate(results, 1):
            record["offset"] = offset
            if q is not None:
                qbank[offset:offset+record["length"]] = q
            offset += record["length"]; counts[record["status"]] += 1
            log.write(json.dumps(record, allow_nan=False)+"\n")
            if i % 1000 == 0:
                log.flush(); qbank.flush()
                print(json.dumps(dict(processed=i, expected=len(rows), counts=dict(counts), seconds=time.monotonic()-start)), flush=True)
    qbank.flush(); del qbank
    if sha(catalog/"summary.json") != args.catalog_sha256 or sha(catalog/"sources.tsv") != summary["outputs_sha256"]["sources.tsv"]:
        raise ValueError("Source catalog changed during audit")
    with (output/"audit.jsonl").open() as log:
        for line in log:
            record = json.loads(line)
            if record["status"] != "ok":
                continue
            for source in record["sources"].values():
                stat = Path(source["path"]).stat()
                if any(getattr(stat, k) != source[k] for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                    raise ValueError("Audited source changed before publication: "+source["path"])
    if any(sha(Path(__file__).with_name(name)) != digest for name, digest in code.items()):
        raise ValueError("Audit source code changed during execution")
    complete = counts["ok"] == len(rows)
    receipt = dict(status=("smoke_complete" if smoke else "existing_content_complete") if complete else "incomplete",
                   queries=len(rows), residues=total, counts=dict(counts), native_queries=args.expected_native,
                   existing_feature_scope=args.expected_existing, full_native_structure_coverage=False,
                   source_catalog_sha256=args.catalog_sha256, source_files_sha256=code,
                   confidence_axis="canonical query residue order;CA B_iso divided by100",
                   nonaccepted_confidence="NaN;pending/failed rows are not usable inputs",
                   geometry_training_inputs_published=False, normalization_fitted=False,
                   raw_descriptor_math_recomputed=False, ESM_input_cache_audited=False, labels_read=False,
                   sources_modified=False, source_bank_immutability_required=True,
                   source_inventory_rechecked_before_publication=True, smoke=smoke,
                   files_sha256={p.name:sha(p) for p in (output/"audit.jsonl", output/"absolute_CA_plddt_div100.npy")},
                   seconds=time.monotonic()-start)
    (output/"summary.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if complete and not smoke:
        (output/"existing_content_ready.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--catalog-sha256", required=True)
    p.add_argument("--expected-existing", required=True, type=int)
    p.add_argument("--expected-native", required=True, type=int)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--raw-bank", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke-ids", type=Path)
    args = p.parse_args()
    if args.workers < 1 or args.expected_existing < 1 or args.expected_native < args.expected_existing:
        p.error("workers/counts must be positive and native count must cover the existing scope")
    if audit(args)["status"] == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
