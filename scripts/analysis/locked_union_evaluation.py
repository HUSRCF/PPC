"""Freeze a validation-selected system and score only its locked test predictions."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.metrics import average_precision_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ppcbind.models import ESMSiteClassifier
from analyze_union_fixed_blends import combine
from analyze_union_msa_complementarity import csv_rows, index_predictions, point_metrics, sha
from analyze_validation_error_complementarity import check_arrays
from build_validation_logit_ensemble import confusion_metrics, evaluate

VAL_QUERIES, VAL_RESIDUES, TEST_QUERIES = 1075, 404267, 724
TARGET, MACRO_GUARDRAIL = .80, .7417948466002535


def load(path):
    return json.loads(Path(path).read_text())


def same_metric(first, second):
    return (math.isfinite(float(first)) and math.isfinite(float(second))
            and abs(float(first) - float(second)) <= 1.e-8)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("Invalid system/member name")
    return name


def groups_from(base):
    rows = csv_rows(Path(base) / "data/sequence_groups.tsv")
    groups = {row["union_seq_id"]: row for row in rows}
    if len(groups) != len(rows):
        raise ValueError("Duplicate sequence-group IDs")
    return groups


def cohort(groups, split, expected):
    ids = sorted(uid for uid, row in groups.items() if row["split"] == split)
    lengths = np.asarray([int(groups[uid]["length"]) for uid in ids])
    if len(ids) != expected or (lengths <= 0).any():
        raise ValueError(f"Wrong {split} cohort")
    return ids, lengths


def packed(path, ids, lengths, groups, labels=None):
    with np.load(path, allow_pickle=False) as values:
        return check_arrays(values, np.asarray(ids), lengths, groups, labels)


def validation_candidate(path):
    # Gate using only the validation artifact before opening shared split metadata.
    with np.load(path, allow_pickle=False) as data:
        ids, lengths = data["seq_ids"], data["lengths"]
        y, values = data["labels"], data["scores"]
        if (ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) != VAL_QUERIES
                or len(set(ids)) != len(ids) or list(ids) != sorted(ids)
                or lengths.shape != ids.shape or lengths.dtype.kind not in "iu"
                or (lengths <= 0).any() or lengths.sum() != VAL_RESIDUES
                or not np.array_equal(data["offsets"], np.r_[0, np.cumsum(lengths)])
                or y.shape != (VAL_RESIDUES,) or values.shape != y.shape
                or not np.isin(y, [0, 1]).all() or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()):
            raise ValueError("Invalid validation-only candidate arrays")
        return ids, lengths, y, values.astype(np.float64)


def strict_model(model_config, state):
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite checkpoint tensors")
    model = ESMSiteClassifier(model_config)
    model.load_state_dict(state, strict=True)
    return model


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_lock(base, points, system, completion_path):
    base, points = Path(base).resolve(), Path(points).resolve()
    safe_name(system)
    summary = load(points / "summary.json")
    if (summary["test_read"] or summary["validation_sequences"] != VAL_QUERIES
            or summary["validation_residues"] != VAL_RESIDUES):
        raise ValueError("Not a full validation-only candidate")
    ids, lengths, y, values = validation_candidate(points / f"{system}_val.npz")
    offsets = np.r_[0, np.cumsum(lengths)]
    macro = float(np.mean([average_precision_score(y[i:j], values[i:j])
                           for i, j in zip(offsets[:-1], offsets[1:])]))
    pooled = float(average_precision_score(y, values))
    if pooled < TARGET or macro < MACRO_GUARDRAIL:
        raise ValueError("Validation target/guardrail not reached; test remains closed")
    groups = groups_from(base)
    expected_ids, expected_lengths = cohort(groups, "val", VAL_QUERIES)
    checked_y, checked_values = packed(points / f"{system}_val.npz", expected_ids, expected_lengths, groups)
    if (not np.array_equal(ids, expected_ids) or not np.array_equal(lengths, expected_lengths)
            or not np.array_equal(y, checked_y) or not np.array_equal(values, checked_values)):
        raise ValueError("Validation candidate changed during locking")
    reported = summary["metrics"][system]
    if not same_metric(pooled, reported["average_precision"]) or not same_metric(macro, reported["chain_ap_macro"]):
        raise ValueError("Candidate does not reproduce its reported AP")
    members = summary.get("fixed_blends", {}).get(system, [system])
    if (not isinstance(members, list) or not members or len(set(members)) != len(members)
            or not set(members) <= set(summary["sources"])):
        raise ValueError("Candidate members are not distinct original sources")
    completion = load(completion_path)
    if set(completion) != set(members) or any(
            row.get("state") != "COMPLETED" or row.get("exit_code") != "0:0"
            for row in completion.values()):
        raise ValueError("Successful terminal-state receipts required for every member")
    sources, scores = {}, {}
    for name in members:
        safe_name(name)
        source = summary["sources"][name]
        for key, digest in (("prediction", "prediction_sha256"), ("metrics", "metrics_sha256")):
            if sha(source[key]) != source[digest]:
                raise ValueError("Original validation source changed")
        _, scores[name] = packed(points / f"{name}_val.npz", ids, lengths, groups, y)
        indexed = index_predictions(torch.load(source["prediction"], map_location="cpu", weights_only=False), set(ids), groups)
        if not np.array_equal(scores[name], np.concatenate([indexed[uid][1] for uid in ids])):
            raise ValueError("Cached source scores differ from original predictions")
        records = [json.loads(line) for line in Path(source["metrics"]).read_text().splitlines()]
        epochs = [row for row in records if row.get("event") == "epoch"]
        if ([row["epoch"] for row in epochs] != list(range(1, 51)) or any("test" in row for row in records)
                or any(row["val"]["valid_chains"] != VAL_QUERIES
                       or row["val"]["valid_residues"] != VAL_RESIDUES
                       or row["val"]["skipped_nonfinite"] != 0 for row in epochs)):
            raise ValueError("Incomplete or test-exposed source training record")
        checkpoint = Path(source["metrics"]).parent / "epochs" / f"epoch_{source['epoch']:03d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload["epoch"] != source["epoch"] or payload["config"]["n_test"] != 0
                or payload["config"]["n_train"] != 4868 or payload["config"]["n_val"] != VAL_QUERIES):
            raise ValueError("Wrong checkpoint epoch or test-exposed training")
        config = payload["config"]
        strict_model(config["model_config"], payload["model"])
        original = next(row["val"] for row in epochs if row["epoch"] == source["epoch"])
        if (not same_metric(payload["val"]["pooled_ap"], original["pooled_ap"])
                or not same_metric(average_precision_score(y, scores[name]), original["pooled_ap"])):
            raise ValueError("Checkpoint validation metadata mismatch")
        sources[name] = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
                             model_state_sha256=state_sha(payload["model"]),
                             epoch=source["epoch"], model_config=config["model_config"],
                             training_config=config, validation_source=source,
                             validation_npz=str(points / f"{name}_val.npz"),
                             validation_npz_sha256=sha(points / f"{name}_val.npz"))
    expected = scores[members[0]] if len(members) == 1 else combine(scores, members)
    if not np.array_equal(values, expected):
        raise ValueError("Candidate does not match its declared fixed combination")
    metrics = point_metrics(y, values, lengths)
    # Test metadata is recorded only after the validation gate has passed.
    test_ids, _ = cohort(groups, "test", TEST_QUERIES)
    return dict(schema_version=1, system=system, members=members, sources=sources,
                mode="single" if len(members) == 1 else "equal_logit", epsilon=1.e-7,
                validation=dict(pooled_AP=pooled, macro_AP=macro, scores_sha256=sha(points / f"{system}_val.npz")),
                gates=dict(f1=metrics["selected_threshold"]["threshold"], mcc=metrics["best_mcc"]["threshold"]),
                base=str(base), sequence_groups_sha256=sha(base / "data/sequence_groups.tsv"),
                test_ids=test_ids, completion_receipt=str(Path(completion_path).resolve()),
                completion_sha256=sha(completion_path), candidate_summary_sha256=sha(points / "summary.json"),
                test_model_selection=False, test_threshold_search=False,
                source_code_sha256=sha(__file__),
                limitation="Validation-selected lock;historically inspected test universe is not a new sealed holdout")


def verified_lock(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("Lock checksum mismatch")
    lock = load(path)
    if (lock["schema_version"] != 1 or not math.isfinite(float(lock["validation"]["pooled_AP"]))
            or not math.isfinite(float(lock["validation"]["macro_AP"]))
            or lock["validation"]["pooled_AP"] < TARGET or lock["validation"]["macro_AP"] < MACRO_GUARDRAIL
            or lock["validation"]["pooled_AP"] > 1 or lock["validation"]["macro_AP"] > 1
            or lock["test_model_selection"] is not False or lock["test_threshold_search"] is not False
            or lock["epsilon"] != 1.e-7):
        raise ValueError("Invalid frozen evaluation contract")
    if (not lock["members"] or len(set(lock["members"])) != len(lock["members"])
            or set(lock["members"]) != set(lock["sources"])
            or lock["mode"] != ("single" if len(lock["members"]) == 1 else "equal_logit")
            or set(lock["gates"]) != {"f1", "mcc"}
            or any(not math.isfinite(float(gate)) or not 0 <= gate <= 1 for gate in lock["gates"].values())):
        raise ValueError("Invalid frozen members or thresholds")
    base = Path(lock["base"])
    if (sha(base / "data/sequence_groups.tsv") != lock["sequence_groups_sha256"]
            or sha(lock["completion_receipt"]) != lock["completion_sha256"]):
        raise ValueError("Frozen cohort or completion receipt changed")
    for name in lock["members"]:
        safe_name(name)
        source = lock["sources"][name]
        if (sha(source["checkpoint"]) != source["checkpoint_sha256"]
                or sha(source["validation_npz"]) != source["validation_npz_sha256"]):
            raise ValueError("Frozen checkpoint or validation scores changed")
    return lock


def fixed_gate_metrics(y, p, lengths, f1_gate, mcc_gate):
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any()
            or len(y) != sum(lengths) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any() or p.shape != y.shape):
        raise ValueError("Invalid fixed-evaluation arrays")
    if any(not math.isfinite(float(gate)) or not 0 <= gate <= 1 for gate in (f1_gate, mcc_gate)):
        raise ValueError("Invalid validation-frozen thresholds")
    result = evaluate(y, p, lengths, f1_gate)
    result["at_validation_MCC_gate"] = confusion_metrics(y, p, mcc_gate)
    offsets = np.r_[0, np.cumsum(lengths)]
    for denominator in (20, 10, 5):
        hits, budgets = [], []
        for i, j in zip(offsets[:-1], offsets[1:]):
            k = max(1, math.ceil((j - i) / denominator))
            chosen = np.argsort(-p[i:j], kind="stable")[:k]
            hits.append(int(y[i:j][chosen].sum()))
            budgets.append(k)
        result[f"L/{denominator}_ACC_macro"] = float(np.mean(np.asarray(hits) / budgets))
        result[f"L/{denominator}_ACC_micro"] = float(sum(hits) / sum(budgets))
    result.update(test_threshold_search=False, thresholds_source="validation lock", top_k_policy="ceil(L/denominator)")
    return result


def score_test(lock_path, lock_sha, predictions, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to repeat/overwrite locked test evaluation: {output}")
    lock = verified_lock(lock_path, lock_sha)
    if set(predictions) != set(lock["members"]):
        raise ValueError("Exactly the frozen member predictions are required")
    # Reserve atomically before opening test data; failed runs remain for audit.
    output.mkdir(parents=True, exist_ok=False)
    groups = groups_from(lock["base"])
    ids, lengths = cohort(groups, "test", TEST_QUERIES)
    if ids != lock["test_ids"]:
        raise ValueError("Frozen test cohort changed")
    y, scores, receipts = None, {}, {}
    for name in lock["members"]:
        path = Path(predictions[name])
        receipt_path = path.with_suffix(".receipt.json")
        receipt = load(receipt_path)
        if (receipt["lock_sha256"] != lock_sha or receipt["member"] != name or receipt["split"] != "test"
                or receipt["checkpoint_sha256"] != lock["sources"][name]["checkpoint_sha256"]
                or receipt["prediction_sha256"] != sha(path)
                or receipt["no_optimizer_created"] is not True or receipt["unchanged_model_state"] is not True
                or receipt["validation_reproduced"] is not True
                or receipt["state_before_sha256"] != lock["sources"][name]["model_state_sha256"]
                or receipt["state_after_sha256"] != lock["sources"][name]["model_state_sha256"]
                or receipt["eval_grad_mode_enabled"] is not False or receipt["parameter_grads_present"] != 0
                or receipt["optimizer_steps"] != 0):
            raise ValueError("Missing frozen-inference integrity or validation-reproduction proof")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["epoch"] != lock["sources"][name]["epoch"]:
            raise ValueError("Wrong test prediction checkpoint epoch")
        provenance = payload.get("provenance", {})
        if (provenance.get("lock_sha256") != lock_sha or provenance.get("member") != name
                or provenance.get("split") != "test"
                or provenance.get("checkpoint_sha256") != lock["sources"][name]["checkpoint_sha256"]):
            raise ValueError("Test prediction provenance differs from the lock")
        indexed = index_predictions(payload, set(ids), groups)
        labels = np.concatenate([indexed[uid][0] for uid in ids])
        if y is not None and not np.array_equal(y, labels):
            raise ValueError("Member test labels differ")
        y = labels
        scores[name] = np.concatenate([indexed[uid][1] for uid in ids])
        receipts[name] = dict(path=str(path), prediction_sha256=sha(path), receipt_sha256=sha(receipt_path))
    p = scores[lock["members"][0]] if lock["mode"] == "single" else combine(scores, lock["members"])
    metrics = fixed_gate_metrics(y, p, lengths, lock["gates"]["f1"], lock["gates"]["mcc"])
    np.savez_compressed(output / "test.npz", seq_ids=np.asarray(ids), lengths=lengths,
                        offsets=np.r_[0, np.cumsum(lengths)], labels=y, scores=p)
    write_new(output / "metrics.json", dict(system=lock["system"], lock_sha256=lock_sha,
              test_queries=len(ids), test_residues=len(y), metrics=metrics, sources=receipts,
              test_model_selection=False, test_threshold_search=False, source_code_sha256=sha(__file__)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    freeze = sub.add_parser("lock")
    for arg in ("base", "points", "completion", "output"):
        freeze.add_argument("--" + arg, type=Path, required=True)
    freeze.add_argument("--system", required=True)
    score = sub.add_parser("score-test")
    for arg in ("lock", "predictions", "output"):
        score.add_argument("--" + arg, type=Path, required=True)
    score.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "lock":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Refusing to replace an existing lock: {args.output}")
        value = make_lock(args.base, args.points, args.system, args.completion)
        write_new(args.output, value)
        print(json.dumps(dict(lock=str(args.output), sha256=sha(args.output), validation=value["validation"])))
    else:
        print(json.dumps(score_test(args.lock, args.lock_sha256, load(args.predictions), args.output)))


if __name__ == "__main__":
    main()
