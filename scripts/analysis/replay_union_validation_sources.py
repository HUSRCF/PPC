"""Validation-only runtime replay of existing frozen points; never opens test data."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_locked_union_inference as inference
from locked_union_evaluation import VAL_QUERIES, VAL_RESIDUES, safe_name


def watch_file(watched, path, digest):
    key = str(path)
    if key in watched and watched[key] != digest:
        raise ValueError("Frozen source artifact changed between members")
    watched.setdefault(key, digest)


def validation_groups(payload, config):
    if (config["n_train"] != 4868 or config["n_val"] != VAL_QUERIES
            or config["n_test"] != 0 or config["eval_test_each_epoch"] is not False):
        raise ValueError("Expected validation-only training configuration")
    split = Path(config["split_dir"])
    if not split.is_absolute():
        raise ValueError("Relative frozen split directory")
    ids_path = split / "val_ids.txt"
    ids = ids_path.read_text().split()
    if len(ids) != VAL_QUERIES or len(set(ids)) != len(ids):
        raise ValueError("Wrong validation ID cohort")
    groups = {}
    for row in payload["records"]:
        uid = row["pdb_id"]
        if uid in groups:
            raise ValueError("Duplicate validation query")
        targets = torch.as_tensor(row["targets"])
        if targets.ndim != 1:
            raise ValueError("Invalid validation target shape")
        groups[uid] = dict(length=len(targets), seq_sha1=row["sequence_sha1"],
                           union_label_sha256=row["union_label_sha256"])
    if set(groups) != set(ids):
        raise ValueError("Source prediction IDs differ from validation split")
    if sum(row["length"] for row in groups.values()) != VAL_RESIDUES:
        raise ValueError("Wrong validation residue count")
    inference.index_predictions(payload, set(ids), groups)
    return groups, inference.sha(ids_path)


def _replay(points, output, device, state):
    points, output = Path(points), Path(output)
    summary_bytes = (points / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    watched_files = {str(points / "summary.json"): summary_sha}
    state.update(stage="validation_manifest", summary_sha256=summary_sha)
    if (summary["test_read"] is not False or summary["validation_sequences"] != VAL_QUERIES
            or summary["validation_residues"] != VAL_RESIDUES):
        raise ValueError("Expected full validation-only point artifacts")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable")
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
    sources, scores, original_scores, reports, common, y = {}, {}, {}, {}, None, None
    for name, item in summary["sources"].items():
        state.update(stage="validation_source_integrity", member=name)
        safe_name(name)
        for key in ("prediction", "metrics"):
            if inference.sha(item[key]) != item[key + "_sha256"]:
                raise ValueError("Frozen validation source changed")
            watch_file(watched_files, item[key], item[key + "_sha256"])
        prediction = torch.load(item["prediction"], map_location="cpu", weights_only=False)
        if prediction["epoch"] != item["epoch"]:
            raise ValueError("Prediction epoch differs from frozen source")
        checkpoint = Path(item["metrics"]).parent / "epochs" / f"epoch_{item['epoch']:03d}.pt"
        checkpoint_sha = inference.sha(checkpoint)
        watch_file(watched_files, checkpoint, checkpoint_sha)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["epoch"] != item["epoch"]:
            raise ValueError("Checkpoint epoch differs from frozen source")
        config = json.loads(json.dumps(saved["config"], default=str))
        groups, ids_sha = validation_groups(prediction, config)
        watch_file(watched_files, Path(config["split_dir"]) / "val_ids.txt", ids_sha)
        if common is not None and groups != common:
            raise ValueError("Validation sequence/label metadata differ across members")
        common = groups
        ids = sorted(groups)
        lengths = np.asarray([groups[uid]["length"] for uid in ids])
        npz_path = points / f"{name}_val.npz"
        watch_file(watched_files, npz_path, inference.sha(npz_path))
        labels, expected = inference.packed(npz_path, ids, lengths, groups, y)
        indexed = inference.index_predictions(prediction, set(ids), groups)
        if not np.array_equal(expected, np.concatenate([indexed[uid][1] for uid in ids])):
            raise ValueError("Validation NPZ differs from original predictions")
        y = labels
        source = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                      epoch=item["epoch"], model_config=config["model_config"], training_config=config,
                      model_state_sha256=inference.state_sha(saved["model"]))
        print(json.dumps(dict(event="validation_source_start", member=name,
                              epoch=item["epoch"], queries=len(ids), test_read=False)), flush=True)
        start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        state["stage"] = "validation_inference"
        actual_payload, proof = inference.infer_member(source, ids, lengths, groups, "val", device)
        actual_payload["provenance"] = dict(split="val", runtime_replay=True, test_read=False)
        actual_index = inference.index_predictions(actual_payload, set(ids), groups)
        actual = np.concatenate([actual_index[uid][1] for uid in ids])
        torch.save(actual_payload, output / f"{name}_val.pt")
        inference.write_new(output / f"{name}_integrity.json", proof)
        state["stage"] = "source_reproduction"
        reports[name] = inference.check_replay(y, actual, expected, lengths)
        reports[name]["elapsed_seconds"] = time.monotonic() - start
        reports[name]["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        print(json.dumps(dict(event="validation_source_reproduced", member=name,
                              report=reports[name], test_read=False)), flush=True)
        sources[name] = dict(checkpoint_sha256=source["checkpoint_sha256"],
                             validation_ids_sha256=ids_sha,
                             original_prediction_sha256=item["prediction_sha256"])
        scores[name], original_scores[name] = actual, expected
    if y is None:
        raise ValueError("No original validation sources")
    for name, members in summary.get("fixed_blends", {}).items():
        state.update(stage="fixed_blend_reproduction", member=name)
        safe_name(name)
        if name in scores or not set(members) <= set(original_scores):
            raise ValueError("Invalid preexisting blend")
        expected = inference.combine(original_scores, members)
        npz_path = points / f"{name}_val.npz"
        watch_file(watched_files, npz_path, inference.sha(npz_path))
        _, cached = inference.packed(npz_path, ids, lengths, common, y)
        if not np.array_equal(expected, cached):
            raise ValueError("Existing blend is not its declared equal-logit combination")
        reports[name] = inference.check_replay(y, inference.combine(scores, members), expected, lengths)
    state.update(stage="frozen_artifact_integrity", member=None)
    if any(inference.sha(path) != digest for path, digest in watched_files.items()):
        raise ValueError("Frozen replay manifest or source artifacts changed during replay")
    result = dict(points=str(points.resolve()), summary_sha256=summary_sha,
                  validation_sequences=len(ids), validation_residues=len(y), reports=reports,
                  sources=sources, test_read=False, test_inference_permitted=False,
                  model_selection=False, new_combinations=False, tolerance_changed=False,
                  source_code_sha256=inference.sha(__file__),
                  inference_code_sha256=inference.sha(inference.__file__),
                  frozen_input_sha256=watched_files,
                  runtime=dict(torch=torch.__version__, device=str(device),
                               device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                               float32_matmul_precision=torch.get_float32_matmul_precision(),
                               matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
    inference.write_new(output / "summary.json", result)
    return result


def replay(points, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(stage="preflight", member=None)
    try:
        return _replay(points, output, device, state)
    except Exception as error:
        inference.write_new(output / "failure.json", dict(
            **state, error_type=type(error).__name__, reason=str(error),
            test_read=False, test_inference_permitted=False, tolerance_changed=False,
        ))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = replay(args.points, args.output, args.device)
    print(json.dumps({key: result[key] for key in ("validation_sequences", "validation_residues", "reports", "test_read")}), flush=True)


if __name__ == "__main__":
    main()
