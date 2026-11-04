"""Fixed validation-only blends and component-paired MSA error decomposition."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import torch

from build_validation_logit_ensemble import (
    confusion_metrics, evaluate, logit, select_threshold, sigmoid,
)


POINTS = {"D_none": ("D_none_seed49", 27), "E_none": ("E_none_seed49", 19),
          "D_encoder": ("D_encoder_seed49", 13), "E_encoder": ("E_encoder_seed49", 28)}
BLENDS = {"blend_baseline_msa": ("D_none", "D_encoder"),
          "blend_pareto": ("D_encoder", "E_encoder")}
CONTRASTS = {"D_encoder_minus_D_none": ("D_none", "D_encoder"),
             "E_encoder28_minus_E_none19": ("E_none", "E_encoder"),
             "blend_baseline_msa_minus_D_none": ("D_none", "blend_baseline_msa"),
             "blend_pareto_minus_D_encoder": ("D_encoder", "blend_pareto")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_metrics(y, p, lengths):
    grid = np.arange(1, 100) / 100.
    threshold = select_threshold(y, p, grid)["threshold"]
    result = evaluate(y, p, lengths, threshold)
    result["best_mcc"] = max((confusion_metrics(y, p, t) for t in grid), key=lambda r: r["mcc"])
    # The reusable legacy helper uses floor; current training uses ceil, so align explicitly.
    for denominator in (20, 10, 5):
        hits, counts, offset = [], [], 0
        for length in lengths:
            k = max(1, math.ceil(int(length) / denominator))
            selected = np.argsort(-p[offset:offset + length], kind="stable")[:k]
            hits.append(int(y[offset:offset + length][selected].sum()))
            counts.append(k)
            offset += length
        result[f"L/{denominator}_ACC_macro"] = float(np.mean(np.array(hits) / counts))
        result[f"L/{denominator}_ACC_micro"] = float(sum(hits) / sum(counts))
    result["top_k_policy"] = "ceil(L/denominator), matching current training metrics"
    return result


def index_predictions(payload, expected, groups):
    result = {}
    for row in payload["records"]:
        uid = row["pdb_id"]
        if uid in result or uid not in expected:
            raise ValueError(f"Duplicate/non-validation query: {uid}")
        length = int(groups[uid]["length"])
        y = torch.as_tensor(row["targets"]).numpy()
        p = torch.as_tensor(row["scores"]).numpy().astype(np.float64)
        idx = torch.as_tensor(row["model_residue_index_0based"]).numpy()
        if (y.shape != (length,) or p.shape != y.shape or not np.array_equal(idx, np.arange(length))
                or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
                or not np.isin(y, [0, 1]).all()):
            raise ValueError(f"Incomplete/nonfinite/misaligned query: {uid}")
        if (row["sequence_sha1"] != groups[uid]["seq_sha1"]
                or row["union_label_sha256"] != groups[uid]["union_label_sha256"]
                or hashlib.sha256(y.astype(np.int8).tobytes()).hexdigest() != row["union_label_sha256"]):
            raise ValueError(f"Frozen sequence/label hash mismatch: {uid}")
        result[uid] = (y.astype(np.uint8), p)
    if set(result) != set(expected):
        raise ValueError("Incomplete validation coverage")
    return result


def ap_curve(y, p, component):
    order = np.argsort(-p, kind="stable")
    sorted_scores = p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(p) - 1]
    return y[order].astype(np.float64), component[order], ends


def weighted_ap(curve, component_counts):
    y, components, ends = curve
    weights = component_counts[components]
    tp = np.cumsum(weights * y)[ends]
    total = np.cumsum(weights)[ends]
    if tp[-1] == 0:
        return 0.
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    return float(np.sum(np.diff(np.r_[0., tp]) * precision) / tp[-1])


_BOOT = None


def init_bootstrap(curves, names, sums, chain_counts):
    global _BOOT
    _BOOT = curves, names, sums, chain_counts


def bootstrap_one(counts):
    curves, names, sums, chain_counts = _BOOT
    pooled = [weighted_ap(c, counts) for c in curves]
    macro = (sums @ counts) / (chain_counts @ counts)
    return np.asarray([macro, pooled])


def grouped_bootstrap(y, predictions, ap_per_chain, chain_components, lengths, repeats, workers,
                      contrasts=None):
    names = list(predictions)
    components, chain_group = np.unique(chain_components, return_inverse=True)
    n = len(components)
    residue_group = np.repeat(chain_group, lengths)
    curves = [ap_curve(y, predictions[name], residue_group) for name in names]
    sums = np.stack([np.bincount(chain_group, weights=ap_per_chain[name], minlength=n) for name in names])
    chain_counts = np.bincount(chain_group, minlength=n)
    counts = np.random.default_rng(20261104).multinomial(n, np.full(n, 1. / n), size=repeats)
    if workers == 1:
        init_bootstrap(curves, names, sums, chain_counts)
        values = list(map(bootstrap_one, counts))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_bootstrap,
                                 initargs=(curves, names, sums, chain_counts)) as pool:
            values = list(pool.map(bootstrap_one, counts, chunksize=16))
    values = np.stack(values)
    out = {}
    for contrast, (left, right) in (CONTRASTS if contrasts is None else contrasts).items():
        a, b = names.index(left), names.index(right)
        out[contrast] = {}
        for metric, axis in (("macro_AP", 0), ("pooled_AP", 1)):
            delta = values[:, axis, b] - values[:, axis, a]
            ci = np.quantile(delta, [.025, .975])
            point = (float(np.mean(ap_per_chain[right]) - np.mean(ap_per_chain[left])) if axis == 0
                     else weighted_ap(curves[b], np.ones(n)) - weighted_ap(curves[a], np.ones(n)))
            out[contrast][metric] = dict(difference=point, ci95=ci.tolist(), excludes_zero=bool(ci[0] > 0 or ci[1] < 0))
    return dict(unit="original SI30 component; resample whole components, retain all chains/residues",
                components=n, repeats=repeats, seed=20261104, contrasts=out,
                limitation="Exploratory fixed-checkpoint conditional intervals on selected validation; not corrected for checkpoint/architecture selection and not independent confirmation")


def bins(value, cuts):
    if value is None:
        return "not_informative"
    for edge, label in cuts:
        if value < edge:
            return label
    return cuts[-1][1]


def safe_spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.bootstrap < 100 or args.workers < 1:
        raise ValueError("Need >=100 bootstrap replicates and >=1 worker")
    torch.set_num_threads(1)
    base = args.base
    root = base / "msa_factorial_20261104"
    groups = {r["union_seq_id"]: r for r in csv_rows(base / "data/sequence_groups.tsv")}
    ids = sorted(uid for uid, r in groups.items() if r["split"] == "val")
    if len(ids) != 1075:
        raise ValueError("Unexpected validation universe")
    components = {}
    for row in csv_rows(base / "data/chain_to_union.tsv"):
        uid = row["union_seq_id"]
        if uid in ids:
            if row["split"] != "val" or components.setdefault(uid, row["component_id"]) != row["component_id"]:
                raise ValueError("A validation sequence spans SI components/splits")
    if set(components) != set(ids):
        raise ValueError("Missing SI component membership")
    args.output.mkdir(parents=True, exist_ok=False)
    sources, scores, lengths, y = {}, {}, np.array([int(groups[uid]["length"]) for uid in ids]), None
    for name, (run, epoch) in POINTS.items():
        path = root / "runs" / run / "predictions" / f"val_epoch_{epoch:03d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["epoch"] != epoch:
            raise ValueError("Wrong prediction epoch")
        indexed = index_predictions(payload, set(ids), groups)
        labels = np.concatenate([indexed[uid][0] for uid in ids])
        if y is not None and not np.array_equal(labels, y):
            raise ValueError("Different labels between models")
        y = labels
        scores[name] = np.concatenate([indexed[uid][1] for uid in ids])
        source_metrics = [json.loads(x) for x in (root / "runs" / run / "metrics.jsonl").read_text().splitlines()]
        source_val = next(r["val"] for r in source_metrics if r.get("event") == "epoch" and r["epoch"] == epoch)
        reconstructed = average_precision_score(y, scores[name])
        if abs(reconstructed - source_val["pooled_ap"]) > 1.e-8:
            raise ValueError("Saved scores do not reproduce source AP")
        sources[name] = dict(path=str(path), sha256=sha(path), epoch=epoch, source_pooled_ap=source_val["pooled_ap"],
                              source_macro_ap=source_val["chain_ap_macro"], saturated_scores=int(np.count_nonzero((scores[name] == 0) | (scores[name] == 1))))
    offsets = np.r_[0, np.cumsum(lengths)]
    if len(y) != 404267:
        raise ValueError("Unexpected validation residue count")
    for name, (left, right) in BLENDS.items():
        scores[name] = sigmoid(.5 * logit(scores[left]) + .5 * logit(scores[right]))
    ap, summary = {}, {}
    for name, p in scores.items():
        ap[name] = np.array([average_precision_score(y[a:b], p[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])
        if name in sources and abs(ap[name].mean() - sources[name]["source_macro_ap"]) > 1.e-8:
            raise ValueError("Saved scores do not reproduce source macro AP")
        result = point_metrics(y, p, lengths)
        summary[name] = result
        np.savez_compressed(args.output / f"{name}_val.npz", seq_ids=np.asarray(ids), lengths=lengths,
                            offsets=offsets, labels=y, scores=p)
    print(json.dumps({"point_metrics": {k: {"macro_AP": v["chain_ap_macro"], "pooled_AP": v["average_precision"]} for k, v in summary.items()}}), flush=True)
    msa_index = {r["union_seq_id"]: r for r in csv_rows(root / "assets/unique_msa_index.tsv")}
    chain_rows = []
    for i, uid in enumerate(ids):
        a, b = offsets[i:i + 2]
        ep = root / "features/encoder" / f"{uid}.pt"
        pp = root / "features/profile" / f"{uid}.pt"
        encoder = torch.load(ep, map_location="cpu", weights_only=False)
        profile = torch.load(pp, map_location="cpu", weights_only=False)
        for data, mode in ((encoder, "encoder"), (profile, "profile")):
            if (data["feature_mode"] != mode or hashlib.sha1(data["sequence"].encode()).hexdigest() != groups[uid]["seq_sha1"]
                    or data["source_msa_sha256"] != msa_index[uid]["msa_sha256"]):
                raise ValueError("Wrong MSA feature source")
        feature = profile["embeddings"].numpy()
        if feature.shape != (lengths[i], 25) or not np.isfinite(feature).all():
            raise ValueError("Invalid profile statistics")
        valid = feature[:, 24] > .5
        kish = np.expm1(feature[valid, 23].astype(float) * math.log1p(16384))
        row = dict(sequence_uid=uid, SI_component=components[uid], length=int(lengths[i]),
                   positive_ratio=float(y[a:b].mean()), MSA_unique_rows=encoder["unique_alignment_rows"],
                   MSA_effective_depth_mean=float(kish.mean()) if len(kish) else None,
                   MSA_informative_column_fraction=float(valid.mean()),
                   encoder_actual_depth=encoder["details"]["selected_depth"],
                   encoder_windows=len(encoder["details"]["windows"]),
                   AP_baseline=float(ap["D_none"][i]), AP_MSA=float(ap["D_encoder"][i]),
                   delta_AP=float(ap["D_encoder"][i] - ap["D_none"][i]),
                   mean_score_baseline=float(scores["D_none"][a:b].mean()),
                   mean_score_MSA=float(scores["D_encoder"][a:b].mean()))
        row["delta_mean_score"] = row["mean_score_MSA"] - row["mean_score_baseline"]
        row["length_group"] = bins(row["length"], [(200, "<200"), (400, "200-399"), (800, "400-799"),
                                                     (1000, "800-999"), (math.inf, ">=1000")])
        row["positive_ratio_group"] = bins(row["positive_ratio"], [(.1, "<0.10"), (.25, "0.10-0.25"), (.5, "0.25-0.50"), (math.inf, ">=0.50")])
        row["effective_depth_group"] = bins(row["MSA_effective_depth_mean"], [(8, "<8"), (32, "8-32"), (128, "32-128"), (math.inf, ">=128")])
        chain_rows.append(row)
    write_csv(args.output / "per_chain_deltas.csv", chain_rows)
    strata = []
    for axis in ("length_group", "positive_ratio_group", "effective_depth_group", "encoder_actual_depth", "encoder_windows"):
        for group in sorted({str(row[axis]) for row in chain_rows}):
            selected = [row for row in chain_rows if str(row[axis]) == group]
            deltas = np.array([row["delta_AP"] for row in selected])
            strata.append(dict(axis=axis, group=group, n=len(selected), delta_AP_mean=float(deltas.mean()),
                               delta_AP_median=float(np.median(deltas)), improved_fraction=float(np.mean(deltas > 0)),
                               delta_mean_score_mean=float(np.mean([row["delta_mean_score"] for row in selected]))))
    write_csv(args.output / "stratified_deltas.csv", strata)
    diagnostics = {}
    for field in ("length", "positive_ratio", "MSA_effective_depth_mean", "encoder_actual_depth"):
        keep = [row for row in chain_rows if row[field] is not None]
        x = [row[field] for row in keep]
        diagnostics[field] = {}
        for effect in ("delta_AP", "delta_mean_score"):
            value = safe_spearman(x, [row[effect] for row in keep])
            diagnostics[field][effect + "_spearman"] = value
    delta_ap = np.array([row["delta_AP"] for row in chain_rows])
    delta_score = np.array([row["delta_mean_score"] for row in chain_rows])
    decomposition = dict(mean_delta_AP=float(delta_ap.mean()), median_delta_AP=float(np.median(delta_ap)),
                         improved_chains=int(np.count_nonzero(delta_ap > 0)),
                         worsened_chains=int(np.count_nonzero(delta_ap < 0)), tied_chains=int(np.count_nonzero(delta_ap == 0)),
                         delta_AP_quantiles=np.quantile(delta_ap, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                         mean_delta_mean_score=float(delta_score.mean()), std_delta_mean_score=float(delta_score.std()),
                         residue_weighted_mean_delta_chain_AP=float(np.average(delta_ap, weights=lengths)),
                         positive_weighted_mean_delta_chain_AP=float(np.average(delta_ap,
                             weights=[row["positive_ratio"] * row["length"] for row in chain_rows])),
                         warning="Weighted mean chain AP is a descriptive diagnostic, not pooled AP or a decomposition of its change")
    bootstrap = grouped_bootstrap(y, scores, ap, np.array([components[uid] for uid in ids]), lengths,
                                  args.bootstrap, args.workers)
    geometry = []
    for path in sorted((base / "architecture_screen_20261104/runs").glob("*/metrics.jsonl")):
        records = [r for r in map(json.loads, path.read_text().splitlines()) if r.get("event") == "epoch"]
        if not records:
            continue
        best = max(records, key=lambda r: r["val"]["chain_ap_macro"])
        geometry.append(dict(run=path.parent.name, epochs=len(records), epoch=best["epoch"],
                             macro_AP=best["val"]["chain_ap_macro"], pooled_AP=best["val"]["pooled_ap"],
                             ratio_MAE=best["val"]["chain_ratio_mae_raw"], metrics_sha256=sha(path)))
    report = dict(validation_sequences=len(ids), validation_residues=len(y), test_read=False,
                  sources=sources, metrics=summary, fixed_blends=BLENDS, alpha_search=False,
                  averaging="equal clipped-probability logit mean, epsilon=1e-7; raw logits not stored",
                  checkpoint_policy="Prespecified D_none27, E_none19, D_encoder13, E_encoder28; E28 is pooled-selected, not macro-selected",
                  MSA_depth_definition="Mean Kish effective depth over informative columns, recovered from cached profile; not HHsuite Neff; query-only/uninformative is missing",
                  bootstrap=bootstrap, strata=strata, chain_decomposition=decomposition,
                  diagnostic_correlations=diagnostics, geometry_controls=geometry,
                  alignment="Full UID, sequence and union-label hashes, residue indices and target arrays; source AP reproduced to1e-8",
                  source_code_sha256=sha(__file__), component_mapping_sha256=sha(base / "data/chain_to_union.tsv"))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": True, "output": str(args.output), "components": bootstrap["components"]}), flush=True)


if __name__ == "__main__":
    main()
