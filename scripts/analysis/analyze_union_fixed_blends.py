"""Evaluate prespecified equal-logit blends on the frozen union validation set."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from analyze_union_msa_complementarity import csv_rows, index_predictions, point_metrics, sha
from build_validation_logit_ensemble import logit, sigmoid


def combine(scores, members):
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("A fixed blend needs at least two distinct source models")
    return sigmoid(np.mean(np.stack([logit(scores[name]) for name in members]), axis=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    spec = json.loads(a.spec.read_text())
    groups = {r["union_seq_id"]: r for r in csv_rows(a.base / "data/sequence_groups.tsv")}
    ids = sorted(uid for uid, row in groups.items() if row["split"] == "val")
    lengths = np.asarray([int(groups[uid]["length"]) for uid in ids])
    if len(ids) != 1075 or lengths.sum() != 404267:
        raise ValueError("Wrong frozen validation universe")
    if set(spec["points"]) & set(spec["blends"]):
        raise ValueError("Source and blend names overlap")
    scores, sources, y = {}, {}, None
    for name, item in spec["points"].items():
        path = Path(item["prediction"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["epoch"] != item["epoch"]:
            raise ValueError("Wrong prediction epoch")
        indexed = index_predictions(payload, set(ids), groups)
        labels = np.concatenate([indexed[uid][0] for uid in ids])
        if y is not None and not np.array_equal(labels, y):
            raise ValueError("Different labels between models")
        y = labels
        scores[name] = np.concatenate([indexed[uid][1] for uid in ids])
        records = [json.loads(line) for line in Path(item["metrics"]).read_text().splitlines()]
        source = next(row["val"] for row in records if row.get("event") == "epoch"
                      and row["epoch"] == item["epoch"])
        offsets = np.r_[0, np.cumsum(lengths)]
        macro = np.mean([average_precision_score(y[i:j], scores[name][i:j])
                         for i, j in zip(offsets[:-1], offsets[1:])])
        pooled = average_precision_score(y, scores[name])
        if abs(macro - source["chain_ap_macro"]) > 1.e-8 or abs(pooled - source["pooled_ap"]) > 1.e-8:
            raise ValueError("Saved predictions do not reproduce original AP")
        sources[name] = {**item, "prediction_sha256": sha(path),
                         "metrics_sha256": sha(item["metrics"]),
                         "macro_AP": float(macro), "pooled_AP": float(pooled)}
    if y is None:
        raise ValueError("No sources")
    for name, members in spec["blends"].items():
        if not set(members) <= set(spec["points"]):
            raise ValueError("Blend members must be original source points")
        scores[name] = combine(scores, members)
    a.output.mkdir(parents=True, exist_ok=False)
    metrics = {}
    for name, values in scores.items():
        metrics[name] = point_metrics(y, values, lengths)
        np.savez_compressed(a.output / f"{name}_val.npz", seq_ids=np.asarray(ids), lengths=lengths,
                            offsets=np.r_[0, np.cumsum(lengths)], labels=y, scores=values)
    result = dict(
        validation_sequences=len(ids), validation_residues=len(y), test_read=False,
        sources=sources, fixed_blends=spec["blends"], metrics=metrics, alpha_search=False,
        averaging="Equal clipped-probability logit mean, epsilon1e-7; raw logits unavailable",
        scope="Exploratory validation-only fixed ensembles of selected checkpoints; no significance claim",
        spec_sha256=sha(a.spec), source_code_sha256=sha(__file__),
        helper_sha256={name: sha(Path(__file__).with_name(name)) for name in
                       ("analyze_union_msa_complementarity.py", "build_validation_logit_ensemble.py")},
    )
    (a.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"macro_AP": m["chain_ap_macro"], "pooled_AP": m["average_precision"]}
                      for name, m in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
