"""Validation-only paired ranking diagnostics, without generating new blends."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from analyze_union_msa_complementarity import csv_rows, grouped_bootstrap, sha, write_csv
from analyze_union_fixed_blends import combine


def selected_systems(summary, include_fixed_blends=False):
    names = list(summary["sources"])
    if not include_fixed_blends:
        return names
    blends = summary.get("fixed_blends", {})
    if set(names) & set(blends):
        raise ValueError("Source and fixed-blend names overlap")
    for members in blends.values():
        if (not isinstance(members, list) or len(members) < 2
                or not all(isinstance(name, str) for name in members)
                or len(set(members)) != len(members)
                or not set(members) <= set(names)):
            raise ValueError("Fixed blends must contain distinct original sources")
    return names + list(blends)


def check_fixed_blend(values, scores, members):
    if not np.array_equal(values, combine(scores, members)):
        raise ValueError("Fixed-blend NPZ does not reproduce its declared equal-logit rule")


def check_arrays(data, ids, lengths, groups, labels=None):
    if not np.array_equal(data["seq_ids"], ids) or not np.array_equal(data["lengths"], lengths):
        raise ValueError("Wrong validation sequence order or lengths")
    offsets = np.r_[0, np.cumsum(lengths)]
    if not np.array_equal(data["offsets"], offsets):
        raise ValueError("Wrong residue offsets")
    y, p = data["labels"], data["scores"]
    if (y.shape != (offsets[-1],) or p.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError("Invalid prediction/label values")
    if labels is not None and not np.array_equal(labels, y):
        raise ValueError("Labels differ between systems")
    for uid, i, j in zip(ids, offsets[:-1], offsets[1:]):
        if hashlib.sha256(y[i:j].astype(np.int8).tobytes()).hexdigest() != groups[uid]["union_label_sha256"]:
            raise ValueError("Wrong frozen union labels")
    return y, p.astype(np.float64)


def centered_unit(values):
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    norm = np.linalg.norm(centered)
    return None if norm == 0 else centered / norm


def correlation(first, second):
    a, b = centered_unit(first), centered_unit(second)
    return None if a is None or b is None else float(np.dot(a, b))


def pairwise(scores, lengths):
    names = list(scores)
    offsets = np.r_[0, np.cumsum(lengths)]
    ranks = {
        name: [centered_unit(rankdata(p[i:j], method="average")) for i, j in zip(offsets[:-1], offsets[1:])]
        for name, p in scores.items()
    }
    global_ranks = {name: rankdata(p, method="average") for name, p in scores.items()}
    rows = []
    for a, first in enumerate(names):
        for second in names[a + 1:]:
            values = [float(np.dot(x, y)) for x, y in zip(ranks[first], ranks[second])
                      if x is not None and y is not None]
            rows.append(dict(
                first=first, second=second,
                pooled_score_Pearson=correlation(scores[first], scores[second]),
                pooled_score_Spearman=correlation(global_ranks[first], global_ranks[second]),
                within_chain_Spearman_macro=float(np.mean(values)) if values else None,
                within_chain_Spearman_median=float(np.median(values)) if values else None,
                valid_chains=len(values),
            ))
    return rows


def label_bin(value, edges, names):
    return names[int(np.searchsorted(edges, value, side="right"))]


def checked_depth(row, length, ratio, component):
    observed_ratio = float(row["positive_ratio"])
    if (not np.isfinite(observed_ratio) or not 0 <= observed_ratio <= 1
            or int(row["length"]) != length or row["SI_component"] != component
            or abs(observed_ratio - ratio) > 1.e-12):
        raise ValueError("MSA covariates do not match the frozen chain")
    value = row["MSA_effective_depth_mean"]
    depth = float(value) if value else None
    if depth is not None and (not np.isfinite(depth) or depth < 0):
        raise ValueError("Invalid MSA Kish depth")
    return depth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--covariates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--include-fixed-blends", action="store_true",
                   help="Verify and diagnose existing fixed ensembles; do not search new combinations")
    a = p.parse_args()
    if a.bootstrap < 100 or a.workers < 1:
        raise ValueError("Invalid bootstrap budget")
    summary = json.loads((a.points / "summary.json").read_text())
    if summary["test_read"] or summary["validation_sequences"] != 1075 or summary["validation_residues"] != 404267:
        raise ValueError("Wrong analysis universe")
    groups = {r["union_seq_id"]: r for r in csv_rows(a.base / "data/sequence_groups.tsv")}
    ids = np.asarray(sorted(uid for uid, row in groups.items() if row["split"] == "val"))
    lengths = np.asarray([int(groups[uid]["length"]) for uid in ids])
    if len(ids) != 1075 or lengths.sum() != 404267:
        raise ValueError("Wrong frozen validation set")
    offsets = np.r_[0, np.cumsum(lengths)]
    components = {}
    for row in csv_rows(a.base / "data/chain_to_union.tsv"):
        uid = row["union_seq_id"]
        if uid in groups and groups[uid]["split"] == "val":
            if row["split"] != "val" or components.setdefault(uid, row["component_id"]) != row["component_id"]:
                raise ValueError("Sequence crosses split/component")
    if set(components) != set(ids):
        raise ValueError("Incomplete SI components")
    with a.covariates.open() as handle:
        cov_rows = list(csv.DictReader(handle))
    cov = {row["sequence_uid"]: row for row in cov_rows}
    if len(cov) != len(cov_rows) or set(cov) != set(ids):
        raise ValueError("Wrong or duplicate MSA covariates")
    scores, ap, means, sources, y = {}, {}, {}, {}, None
    for name in selected_systems(summary, a.include_fixed_blends):
        path = a.points / f"{name}_val.npz"
        with np.load(path, allow_pickle=False) as data:
            labels, values = check_arrays(data, ids, lengths, groups, y)
        y = labels
        if name not in summary["sources"]:
            check_fixed_blend(values, scores, summary["fixed_blends"][name])
        scores[name] = values
        ap[name] = np.asarray([average_precision_score(y[i:j], values[i:j]) for i, j in zip(offsets[:-1], offsets[1:])])
        means[name] = np.asarray([values[i:j].mean() for i, j in zip(offsets[:-1], offsets[1:])])
        raw = summary["metrics"][name]
        if (abs(ap[name].mean() - raw["chain_ap_macro"]) > 1.e-8
                or abs(average_precision_score(y, values) - raw["average_precision"]) > 1.e-8):
            raise ValueError("Point NPZ does not reproduce source AP")
        sources[name] = dict(npz_sha256=sha(path), **(
            dict(original=summary["sources"][name]) if name in summary["sources"] else
            dict(existing_fixed_blend_members=summary["fixed_blends"][name],
                 fixed_blend_reconstruction="exact equal clipped-probability logit mean")))
    if a.reference not in scores:
        raise ValueError("Missing reference")
    table = []
    for k, uid in enumerate(ids):
        i, j = offsets[k:k + 2]
        ratio = float(y[i:j].mean())
        depth = checked_depth(cov[uid], lengths[k], ratio, components[uid])
        row = dict(sequence_uid=uid, SI_component=components[uid], length=int(lengths[k]), positive_ratio=ratio,
                   MSA_Kish_effective_depth=depth,
                   length_group=label_bin(lengths[k], [200, 400, 800], ["<200", "200-399", "400-799", ">=800"]),
                   positive_ratio_group=label_bin(ratio, [.1, .25, .5], ["<.1", ".1-.25", ".25-.5", ">=.5"]),
                   MSA_depth_group="missing" if depth is None else
                       label_bin(depth, [8, 32, 128], ["<8", "8-31", "32-127", ">=128"]))
        for name in scores:
            row["AP_" + name] = float(ap[name][k])
            row["mean_score_" + name] = float(means[name][k])
            row["delta_AP_" + name] = float(ap[name][k] - ap[a.reference][k])
        table.append(row)
    contrasts = {name + "_minus_" + a.reference: (a.reference, name) for name in scores if name != a.reference}
    boot = grouped_bootstrap(y, scores, ap, np.asarray([components[uid] for uid in ids]), lengths,
                             a.bootstrap, a.workers, contrasts=contrasts)
    strata = []
    for axis in ("length_group", "positive_ratio_group", "MSA_depth_group"):
        for group in sorted({row[axis] for row in table}):
            indices = [i for i, row in enumerate(table) if row[axis] == group]
            for name in scores:
                if name == a.reference:
                    continue
                delta = ap[name][indices] - ap[a.reference][indices]
                strata.append(dict(axis=axis, group=group, model=name, n=len(indices),
                                   mean_delta_AP=float(delta.mean()), median_delta_AP=float(np.median(delta)),
                                   improved_fraction=float(np.mean(delta > 0))))
    effects = {}
    for name in scores:
        delta = ap[name] - ap[a.reference]
        effects[name] = dict(macro_AP=float(ap[name].mean()), pooled_AP=float(average_precision_score(y, scores[name])),
                            delta_AP_macro=float(delta.mean()), delta_AP_median=float(np.median(delta)),
                            improved_chains=int((delta > 0).sum()), worsened_chains=int((delta < 0).sum()),
                            chain_macro_mean_score_shift=float((means[name] - means[a.reference]).mean()))
    a.output.mkdir(parents=True, exist_ok=False)
    write_csv(a.output / "per_chain.csv", table)
    write_csv(a.output / "stratified.csv", strata)
    write_csv(a.output / "pairwise.csv", pairwise(scores, lengths))
    result = dict(reference=a.reference, validation_sequences=len(ids), validation_residues=len(y), test_read=False,
                  sources=sources, effects=effects, bootstrap=boot, new_blends_computed=False,
                  included_fixed_blends=a.include_fixed_blends,
                  covariates_sha256=sha(a.covariates), points_summary_sha256=sha(a.points / "summary.json"),
                  source_code_sha256=sha(__file__),
                  bootstrap_helper_sha256=sha(Path(__file__).with_name("analyze_union_msa_complementarity.py")),
                  blend_helper_sha256=sha(Path(__file__).with_name("analyze_union_fixed_blends.py")),
                  interpretation="Descriptive selected-validation diagnostics;MSA depth is Kish,not HHsuite Neff;no seed uncertainty or multiple-selection correction")
    (a.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(effects=effects, components=boot["components"], output=str(a.output))), flush=True)


if __name__ == "__main__":
    main()
