"""Select OF3 monomers by official confidence, never by contact labels."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(candidates):
    if len(candidates) != 5 or {x[0] for x in candidates} != set(range(1, 6)):
        raise ValueError("Expected samples 1..5 exactly")
    if not all(math.isfinite(x[1]) for x in candidates):
        raise ValueError("Nonfinite ranking score")
    return max(candidates, key=lambda x: (x[1], -x[0]))


def main():
    import gemmi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()
    queries = {}
    for shard in range(4):
        work = args.run_root / f"shard_{shard}"
        for uid, query in json.loads((work / "queries.json").read_text())["queries"].items():
            if uid in queries:
                raise ValueError("Duplicated query: " + uid)
            queries[uid] = (work, query["chains"][0]["sequence"])
    if len(queries) != 6667:
        raise ValueError("Wrong query universe")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pdb").mkdir(exist_ok=True)
    rows = []
    selected = sorted(queries)[:args.max or None]
    for index, uid in enumerate(selected):
        work, sequence = queries[uid]
        candidates = []
        base = work / "predictions" / uid / "seed_42"
        for sample in range(1, 6):
            prefix = base / f"{uid}_seed_42_sample_{sample}"
            info = json.loads(Path(str(prefix) + "_confidences_aggregated.json").read_text())
            candidates.append((sample, float(info["sample_ranking_score"]), prefix, info))
        sample, score, prefix, info = choose(candidates)
        source = Path(str(prefix) + "_model.cif")
        structure = gemmi.read_structure(str(source))
        if len(structure) != 1 or len(structure[0]) != 1:
            raise ValueError("Expected one protein chain: " + uid)
        chain = structure[0][0]
        observed = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                           for r in chain if r.find_atom("CA", "*") is not None)
        if observed != sequence or len(chain) != len(sequence):
            raise ValueError("CIF sequence differs from exact query: " + uid)
        chain.name = "A"
        for number, residue in enumerate(chain, 1):
            residue.seqid = gemmi.SeqId(number, " ")
            if not all(math.isfinite(v) for atom in residue
                       for v in (atom.pos.x, atom.pos.y, atom.pos.z)):
                raise ValueError("Nonfinite coordinates: " + uid)
        target = args.output / "pdb" / (uid + ".pdb")
        text = structure.make_pdb_string()
        roundtrip = gemmi.read_pdb_string(text)
        restored = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                            for r in roundtrip[0][0])
        if restored != sequence:
            raise ValueError("PDB roundtrip sequence mismatch: " + uid)
        if target.exists() and target.read_text() != text:
            raise ValueError("Refusing to change frozen selection: " + uid)
        target.write_text(text)
        rows.append({"union_seq_id": uid, "length": len(sequence), "sample": sample,
                     "sample_ranking_score": score, "avg_plddt": info["avg_plddt"],
                     "ptm": info["ptm"], "source_cif": str(source), "source_cif_sha256": sha(source),
                     "pdb_file": target.name, "pdb_sha256": sha(target)})
        if (index + 1) % 250 == 0:
            print(json.dumps({"selected": index + 1, "total": len(selected)}), flush=True)
    path = args.output / "selection.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "selection_summary.json").write_text(json.dumps({
        "count": len(rows), "complete": len(rows) == 6667,
        "rule": "maximum official sample_ranking_score; ties choose smallest sample index",
        "labels_used": False, "sequence_coordinate": "exact observed query, sequential 1-based, chain A",
        "selection_sha256": sha(path), "source_run": str(args.run_root)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
