"""Versioned OF3 geometry and descriptor corrections, without reading labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.e-8)


def coordinates(path, sequence):
    model = PDBParser(QUIET=True).get_structure("monomer", path)[0]
    chains = list(model)
    residues = [r for r in chains[0] if "CA" in r]
    actual = "".join(seq1(r.resname, custom_map={"MSE": "M"}) for r in residues)
    if len(chains) != 1 or actual != sequence:
        raise ValueError(f"OF3 sequence/chain mismatch: {path}")
    ca = np.asarray([r["CA"].coord for r in residues], dtype=np.float32)
    q = np.asarray([r["CA"].bfactor for r in residues], dtype=np.float32)
    if not np.isfinite(ca).all() or not np.isfinite(q).all() or q.min() < 0 or q.max() > 100:
        raise ValueError(f"Invalid coordinates/pLDDT: {path}")
    # OF3 selection.tsv avg_plddt and PDB B factors use the 0--100 scale.
    q = q / 100.
    vectors = np.zeros((len(ca), 8, 3), dtype=np.float32)
    frames = np.zeros((len(ca), 3, 3), dtype=np.float32)
    valid = np.zeros(len(ca), dtype=np.float32)
    for i, r in enumerate(residues):
        for j, name in enumerate(("N", "C", "O", "CB")):
            if name in r:
                vectors[i, j] = unit(r[name].coord - ca[i])
        sc = [a.coord for a in r if a.name not in {"N", "CA", "C", "O", "OXT"}
              and a.element not in {"H", "D"}]
        if sc:
            vectors[i, 4] = unit(np.mean(sc, axis=0) - ca[i])
        if i:
            vectors[i, 5] = unit(ca[i - 1] - ca[i])
        if i + 1 < len(ca):
            vectors[i, 6] = unit(ca[i + 1] - ca[i])
        if "N" in r and "C" in r:
            x, n = unit(r["C"].coord - ca[i]), unit(r["N"].coord - ca[i])
            cross = np.cross(x, n)
            if np.linalg.norm(cross) > 1.e-5:
                z = unit(cross)
                frames[i] = np.stack([x, np.cross(z, x), z])
                vectors[i, 7], valid[i] = z, 1.
    return ca, q, vectors, frames, valid


def make_geometry(ca, q, vectors, frames, valid, k=16, radius=16.):
    n = len(ca)
    dist, nbr = cKDTree(ca).query(ca, k=min(n, k + 1), distance_upper_bound=radius)
    if dist.ndim == 1:
        dist, nbr = dist[:, None], nbr[:, None]
    pairs = {(i, int(j)) for i in range(n) for d, j in zip(dist[i], nbr[i])
             if j < n and i != j and np.isfinite(d)}
    pairs |= {(j, i) for i, j in list(pairs)}
    pairs |= {(i, j) for i in range(n) for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < n}
    edges = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2).T
    src, dst = edges
    displacement = ca[dst] - ca[src]
    distance = np.linalg.norm(displacement, axis=-1)
    direction = unit(displacement)
    sep = np.abs(src - dst)
    rbf = np.exp(-((distance[:, None] - np.linspace(0., 20., 16)) / (20. / 15.)) ** 2)
    seq = np.stack([np.log1p(sep) / np.log1p(max(n, 1)), 1. / (sep + 1.),
                    np.sign(dst - src), sep <= 2], axis=-1)
    attrs = np.concatenate([rbf, seq, q[src, None], q[dst, None],
                            np.einsum("ei,eji->ej", direction, frames[src]),
                            np.einsum("ei,eji->ej", -direction, frames[dst]),
                            valid[src, None], valid[dst, None]], axis=-1).astype(np.float32)
    scores = (np.exp(-(distance / 10.) ** 2)
              * np.sqrt(np.maximum(q[src], .05) * np.maximum(q[dst], .05))).astype(np.float32)
    result = dict(geometry_vectors=torch.from_numpy(vectors), geometry_edge_index=torch.from_numpy(edges),
                  geometry_edge_features=torch.from_numpy(attrs),
                  geometry_edge_vectors=torch.from_numpy(direction[:, None].astype(np.float32)),
                  geometry_edge_scores=torch.from_numpy(scores))
    result.update(contact_edge_index=result["geometry_edge_index"], contact_edge_scores=result["geometry_edge_scores"])
    return result


def process(job):
    uid, sequence, inputs, selected, output, norm, k, radius, expected_sha, avg_plddt = job
    torch.set_num_threads(1)
    inputs, selected, output = map(Path, (inputs, selected, output))
    path = selected / "pdb" / f"{uid}.pdb"
    if sha(path) != expected_sha:
        raise ValueError(f"Frozen OF3 PDB changed: {uid}")
    ca, q, vectors, frames, valid = coordinates(path, sequence)
    if not 0. <= avg_plddt <= 100.:
        raise ValueError(f"Unexpected OF3 global confidence scale: {uid}")
    old = torch.load(inputs / "structure" / uid / f"{uid}_seq.pt", map_location="cpu", weights_only=False)
    if old["sequence"] != sequence or old["seq_features"].shape != (len(sequence), 90):
        raise ValueError(f"Structure metadata mismatch: {uid}")
    # Inversion is exact up to float32 rounding; other 87 normalized channels stay unchanged.
    x = old["seq_features"].numpy()
    sasa = np.maximum(x[:, 0] * norm["scalar_stds"][0] + norm["scalar_means"][0], 0.)
    bnorm = x[:, 49] * norm["scalar_stds"][49] + norm["scalar_means"][49]
    spread = np.asarray([np.sqrt(np.square(ca[max(0, i - 3):i + 4]
                         - ca[max(0, i - 3):i + 4].mean(0)).sum(-1).mean()) for i in range(len(ca))])
    spread /= max(float(spread.max()), 1.e-8)
    corrected = np.stack([spread, sasa / max(float(sasa.max()), 1.e-8) * bnorm], -1).astype(np.float32)
    geom = make_geometry(ca, q, vectors, frames, valid, k, radius)
    geom.update(sequence=sequence, source_pdb_sha256=sha(path), source="OF3 selected monomer only")
    if not all(torch.isfinite(t).all() for t in geom.values() if isinstance(t, torch.Tensor)):
        raise ValueError(f"Nonfinite geometry: {uid}")
    torch.save(geom, output / "geometry" / f"{uid}.pt")
    return uid, corrected, q, geom["source_pdb_sha256"], sha(inputs / "structure" / uid / f"{uid}_seq.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; cannot produce formal ready receipt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    inputs = args.base / "training_of3_union_20261102/inputs"
    selected = args.base / "openfold3/selected_for_training_20261102"
    norm = json.loads((inputs / "normalization_train_only.json").read_text())
    if norm["fit_split"] != "train" or norm["n_sequences"] != 4868:
        raise ValueError("Unexpected parent normalization contract")
    with (inputs / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    splits = {s: [r["pdb_id"] for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if {s: len(ids) for s, ids in splits.items()} != {"train": 4868, "val": 1075, "test": 724}:
        raise ValueError("Unexpected manifest split sizes")
    for s, ids in splits.items():
        if set((inputs / "split" / f"{s}_ids.txt").read_text().split()) != set(ids):
            raise ValueError(f"Split IDs changed: {s}")
    if hashlib.sha256("\n".join(splits["train"]).encode()).hexdigest() != norm["train_ids_sha256"]:
        raise ValueError("Parent normalization used different training IDs")
    with (selected / "selection.tsv").open() as f:
        selection = {r["union_seq_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    from Bio import SeqIO
    sequences = {r.id: str(r.seq) for r in SeqIO.parse(args.base / "data/unique_sequences.fasta", "fasta")}
    if (len(sequences) != 6667 or set(sequences) != {r["pdb_id"] for r in rows}
            or set(selection) != set(sequences)):
        raise ValueError("Unexpected unique sequence universe")
    if args.limit:
        rows = sorted(rows, key=lambda r: int(r["n_residues"]))[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("geometry", "fixed90", "fixed91q"):
        (args.output / name).mkdir()
    jobs = [(r["pdb_id"], sequences[r["pdb_id"]], str(inputs), str(selected), str(args.output), norm, 16, 16.,
             selection[r["pdb_id"]]["pdb_sha256"], float(selection[r["pdb_id"]]["avg_plddt"])) for r in rows]
    values = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(process, jobs, chunksize=4), 1):
            values[result[0]] = result[1:]
            if index % 250 == 0:
                print(json.dumps({"geometry_complete": index, "expected": len(rows)}), flush=True)
    train = [values[r["pdb_id"]][0] for r in rows if r["split"] == "train"]
    if not train:
        raise ValueError("No training records for normalization")
    fit = np.concatenate(train).astype(np.float64)
    means, stds = fit.mean(0), fit.std(0)
    stds = np.maximum(stds, 1.e-6)
    audit = []
    for row in rows:
        uid = row["pdb_id"]
        corrected, q, pdb_sha, parent_sha = values[uid]
        old = torch.load(inputs / "structure" / uid / f"{uid}_seq.pt", map_location="cpu", weights_only=False)
        original = old["seq_features"]
        fixed = original.clone()
        fixed[:, [8, 51]] = torch.from_numpy(((corrected - means) / stds).astype(np.float32))
        unchanged = [i for i in range(90) if i not in (8, 51)]
        if not torch.equal(original[:, unchanged], fixed[:, unchanged]):
            raise ValueError("Unrelated normalized descriptor changed")
        names = list(old["feature_names_with_availability"])
        names[8], names[51] = "local_centered_ca_spread_v1", "sasa_relative_flexibility_v1"
        for name, tensor, feature_names in (("fixed90", fixed, names),
                ("fixed91q", torch.cat([fixed[:, :89], torch.from_numpy(q[:, None]), fixed[:, 89:]], 1),
                 names[:89] + ["absolute_plddt_div100", names[89]])):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite corrected descriptor: {uid}")
            payload = {**old, "seq_features": tensor, "feature_names_with_availability": feature_names}
            torch.save(payload, args.output / name / f"{uid}.pt")
        audit.append(dict(uid=uid, length=len(q), split=row["split"], source_pdb_sha256=pdb_sha,
                          parent_features_sha256=parent_sha))
    normalization = dict(parent_sha256=sha(inputs / "normalization_train_only.json"), fit_split="train",
                         fit_sequences=len(train), fit_residues=len(fit), corrected_indices=[8, 51],
                         corrected_means=means.tolist(), corrected_stds=stds.tolist(),
                         unchanged_parent_channels=87, other_columns_byte_preserved=True,
                         raw_recovery="float32 inverse z score for SASA[0] and Bnorm[49]; no clipping in parent")
    (args.output / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    receipt = dict(sequences=len(rows), full_length=True, labels_read=False, fit_split="train",
                   source_code_sha256=sha(__file__), parent_manifest_sha256=sha(inputs / "manifest.csv"),
                   selection_sha256=sha(selected / "selection.tsv"), geometry_edge_dimension=30,
                   vector_channels=8, knn=16, radius_angstrom=16., sequence_neighbor_range=2,
                   q_scale="C-alpha atom pLDDT from PDB B factor 0--100, divided by 100; not all-atom global avg_plddt", normalization=normalization,
                   test_labels_read=False, smoke=bool(args.limit))
    (args.output / ("smoke_ready.json" if args.limit else "ready.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
