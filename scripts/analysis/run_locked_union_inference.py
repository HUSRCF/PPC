"""Replay a validation lock, then infer its fixed union test cohort without fitting."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from locked_union_evaluation import (
    MACRO_GUARDRAIL, TARGET, TEST_QUERIES, VAL_QUERIES, cohort, combine,
    groups_from, index_predictions, packed, sha, state_sha, strict_model,
    verified_lock, write_new,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import train_esm_site as training
from ppcbind.data.esm_site_dataset import _esm_id

MAX_SCORE_ERROR, MAX_SCORE_RMSE, AP_TOLERANCE = 2.e-5, 2.e-6, 2.e-6
LOADER_PARAMETERS = inspect.signature(training._build_loader).parameters


def loader_for(config, ids, lengths, split):
    if config.get("chain_filter_manifest") is not None:
        raise ValueError("Union inference requires per-sequence payloads, not a chain filter")
    kwargs = {key: value for key, value in config.items() if key in LOADER_PARAMETERS}
    for key in training.PATH_KEYS & kwargs.keys():
        kwargs[key] = Path(kwargs[key]) if kwargs[key] is not None else None
        if kwargs[key] is not None and not kwargs[key].is_absolute():
            raise ValueError(f"Relative frozen input root is forbidden: {key}")
    kwargs.update(ids=ids, lengths_by_id=dict(zip(ids, map(int, lengths))),
                  batch_size=int(config.get("eval_batch_size") or config["batch_size"]),
                  max_residues=0, crop_mode="none", shuffle=False, preload=False,
                  seed=int(config["seed"]) + (1 if split == "val" else 2),
                  strict_ids=True, require_labels=True, strict_label_metadata=True,
                  strict_sequence_feature_metadata=True,
                  require_contact_graph=bool(config["require_contact_graph_resolved"]))
    if kwargs["batch_size"] <= 0:
        raise ValueError("Invalid frozen evaluation batch size")
    return training._build_loader(**kwargs)


def payload_sequence(payload, path):
    if not isinstance(payload, dict):
        raise ValueError(f"Input has no sequence metadata: {path}")
    sequence = payload.get("sequence")
    residues = payload.get("residue_names_1", payload.get("residue_name_1"))
    if sequence is None and residues is None:
        raise ValueError(f"Input has no sequence metadata: {path}")
    if sequence is None:
        sequence = "".join(residues)
    if not isinstance(sequence, str) or not sequence or (residues is not None and "".join(residues) != sequence):
        raise ValueError(f"Inconsistent sequence metadata: {path}")
    return sequence


def external_path(root, uid, kind):
    root = Path(root)
    if kind == "graph":
        paths = [root / uid / f"{uid}{suffix}.pt" for suffix in
                 ("_contact_graph", "_predcontact", "_pred_struct_contact")]
        paths += [root / f"{uid}{suffix}.pt" for suffix in ("_contact_graph", "_predcontact", "")]
        path = next((path for path in paths if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing frozen graph input for {uid}")
        return path
    suffixes = ("_seq", "_sequence") if kind == "sequence" else (
        "_prottrans", "_protbert", "_protbert_bfd", "_embeddings", "")
    paths = [root / uid / f"{uid}{suffix}.pt" for suffix in suffixes]
    paths += [root / f"{uid}{suffix}.pt" for suffix in (suffixes if kind != "sequence" else ("_seq", ""))]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing frozen {kind} input for {uid}")
    return path


def audit_inputs(config, dataset, ids, groups):
    paths = {_esm_id(path): path for path in dataset.esm_paths}
    if set(paths) != set(ids) or len(paths) != len(dataset.esm_paths):
        raise ValueError("Input metadata paths differ from frozen cohort")
    actual, inventory = {}, {}
    roots = [("sequence_feature_root", "sequence"), ("primary_embedding_root", "embedding"),
             ("prottrans_embedding_root", "embedding"), ("geometry_feature_root", "geometry"),
             ("contact_graph_root", "graph"), ("aux_contact_graph_root", "graph")]
    for uid in ids:
        candidates = [paths[uid]]
        for key, kind in roots:
            if config.get(key) is not None:
                candidates.append(Path(config[key]) / f"{uid}.pt" if kind == "geometry" else
                                  external_path(config[key], uid, kind))
        sequence = None
        for path in candidates:
            before = path.stat()
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            observed = payload_sequence(payload, path)
            if sequence is not None and observed != sequence:
                raise ValueError(f"Input sequence mismatch between feature sources: {uid}")
            sequence = observed
            fingerprint = sha(path)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Input changed during metadata audit")
            inventory[str(path)] = fingerprint
        digest = hashlib.sha1(sequence.encode()).hexdigest()
        if digest != groups[uid]["seq_sha1"] or len(sequence) != int(groups[uid]["length"]):
            raise ValueError(f"Actual input sequence differs from frozen cohort: {uid}")
        actual[uid] = digest
    return actual, inventory


def predict_batches(model, loader, ids, groups, device, input_sequence_hashes):
    model.eval()
    model.requires_grad_(False)
    before = state_sha(model.state_dict())
    records = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for batch in loader:
            inputs, labels = training._move_batch(batch, device)
            outputs = model(**inputs)
            logits = outputs["logits"]
            if torch.is_grad_enabled() or logits.requires_grad or not torch.isfinite(logits).all():
                raise ValueError("Nonfinite or gradient-enabled inference")
            scores = torch.softmax(logits.float(), dim=-1)[..., 1]
            for item, uid in enumerate(batch["pdb_id"]):
                if uid not in groups or uid not in ids:
                    raise ValueError("Unexpected inference sequence")
                length = int(groups[uid]["length"])
                if input_sequence_hashes[uid] != groups[uid]["seq_sha1"]:
                    raise ValueError("Actual input sequence differs from frozen cohort")
                mask = inputs["protein_mask"][item].bool()
                valid = labels[item] != -100
                expected = torch.arange(len(mask), device=device) < length
                if not torch.equal(mask, expected) or not torch.equal(valid, expected):
                    raise ValueError("Truncated, ignored or non-prefix query residues")
                records.append(dict(
                    pdb_id=uid, scores=scores[item, :length].cpu(),
                    targets=labels[item, :length].long().cpu(),
                    model_residue_index_0based=torch.arange(length),
                    sequence_sha1=input_sequence_hashes[uid],
                    union_label_sha256=groups[uid]["union_label_sha256"],
                ))
    payload = dict(records=records)
    index_predictions(payload, set(ids), groups)
    after = state_sha(model.state_dict())
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Inference mutated model state or produced gradients")
    return payload, dict(state_before_sha256=before, state_after_sha256=after,
                         no_optimizer_created=True, unchanged_model_state=True,
                         eval_grad_mode_enabled=False, parameter_grads_present=0,
                         optimizer_steps=0)


def ap_values(y, scores, lengths):
    offsets = np.r_[0, np.cumsum(lengths)]
    return dict(pooled_AP=float(average_precision_score(y, scores)),
                macro_AP=float(np.mean([average_precision_score(y[i:j], scores[i:j])
                                       for i, j in zip(offsets[:-1], offsets[1:])])))


def check_replay(y, actual, expected, lengths):
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or ((actual < 0) | (actual > 1)).any()):
        raise ValueError("Invalid validation replay scores")
    error = actual - expected
    stats = dict(max_score_error=float(np.max(np.abs(error))),
                 score_RMSE=float(np.sqrt(np.mean(error ** 2))),
                 actual=ap_values(y, actual, lengths), expected=ap_values(y, expected, lengths))
    if (stats["max_score_error"] > MAX_SCORE_ERROR or stats["score_RMSE"] > MAX_SCORE_RMSE
            or any(abs(stats["actual"][key] - stats["expected"][key]) > AP_TOLERANCE
                   for key in ("pooled_AP", "macro_AP"))):
        raise ValueError(f"Validation replay differs from frozen predictions: {stats}")
    return stats


def infer_member(source, ids, lengths, groups, split, device):
    if sha(source["checkpoint"]) != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after lock")
    payload = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    saved_config = json.loads(json.dumps(payload["config"], default=str))
    if (saved_config != source["training_config"]
            or saved_config["model_config"] != source["model_config"]
            or payload["epoch"] != source["epoch"]):
        raise ValueError("Checkpoint configuration/epoch differs from lock")
    model = strict_model(source["model_config"], payload["model"]).to(device)
    if state_sha(model.state_dict()) != source["model_state_sha256"]:
        raise ValueError("Model state differs from lock")
    loader = loader_for(source["training_config"], ids, lengths, split)
    hashes, inventory = audit_inputs(source["training_config"], loader.dataset, ids, groups)
    result, proof = predict_batches(model, loader, ids, groups, device, hashes)
    after_hashes, after_inventory = audit_inputs(source["training_config"], loader.dataset, ids, groups)
    if hashes != after_hashes or inventory != after_inventory:
        raise ValueError("Input contents or selected resolver paths changed during inference")
    proof["input_files_sha256"] = inventory
    result["epoch"] = source["epoch"]
    if proof["state_after_sha256"] != source["model_state_sha256"]:
        raise ValueError("Model state changed during inference")
    return result, proof


def run_locked(lock_path, lock_sha, output, device, validation_only=False):
    lock = verified_lock(lock_path, lock_sha)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable; refusing silent CPU fallback")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = groups_from(lock["base"])
    ids, lengths = cohort(groups, "val", VAL_QUERIES)
    val_scores, comparisons, y = {}, {}, None
    for name in lock["members"]:
        source = lock["sources"][name]
        labels, expected = packed(source["validation_npz"], ids, lengths, groups, y)
        payload, proof = infer_member(source, ids, lengths, groups, "val", device)
        indexed = index_predictions(payload, set(ids), groups)
        actual = np.concatenate([indexed[uid][1] for uid in ids])
        y = labels
        comparisons[name] = check_replay(y, actual, expected, lengths)
        val_scores[name] = actual
        torch.save(payload, output / f"{name}_val.pt")
        write_new(output / f"{name}_val_integrity.json", proof)
    actual = val_scores[lock["members"][0]] if lock["mode"] == "single" else combine(val_scores, lock["members"])
    metrics = ap_values(y, actual, lengths)
    if (metrics["pooled_AP"] < TARGET or metrics["macro_AP"] < MACRO_GUARDRAIL
            or any(abs(metrics[key] - lock["validation"][key]) > AP_TOLERANCE for key in metrics)):
        raise ValueError("Replayed validation system does not reproduce the locked milestone")
    replay_path = output / "validation_reproduction.json"
    write_new(replay_path, dict(lock_sha256=lock_sha, members=comparisons, metrics=metrics,
              tolerances=dict(max_score_error=MAX_SCORE_ERROR, score_RMSE=MAX_SCORE_RMSE,
                              absolute_AP=AP_TOLERANCE), test_read=False))
    if validation_only:
        return dict(validation_reproduced=True, test_read=False, metrics=metrics)
    # No test labels/feature loader is opened until every member and the blend replay pass.
    ids, lengths = cohort(groups, "test", TEST_QUERIES)
    if ids != lock["test_ids"]:
        raise ValueError("Test cohort differs from lock")
    verified_lock(lock_path, lock_sha)
    predictions = {}
    for name in lock["members"]:
        source = lock["sources"][name]
        payload, proof = infer_member(source, ids, lengths, groups, "test", device)
        provenance = dict(lock_sha256=lock_sha, member=name, split="test",
                          checkpoint_sha256=source["checkpoint_sha256"])
        payload["provenance"] = provenance
        path = output / f"{name}_test.pt"
        torch.save(payload, path)
        write_new(path.with_suffix(".receipt.json"), dict(
            **provenance, **proof, validation_reproduced=True,
            prediction_sha256=sha(path), validation_reproduction_sha256=sha(replay_path),
            source_code_sha256=sha(__file__), training_helper_sha256=sha(training.__file__),
        ))
        predictions[name] = str(path.resolve())
    write_new(output / "predictions.json", predictions)
    return dict(validation_reproduced=True, test_inferred=True, test_scored=False,
                predictions=str(output / "predictions.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_locked(args.lock, args.lock_sha256, args.output, args.device,
                                args.validation_only)), flush=True)


if __name__ == "__main__":
    main()
