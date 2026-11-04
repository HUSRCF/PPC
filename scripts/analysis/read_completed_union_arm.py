"""Resolve one completed, prospectively declared validation arm; never run test."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


QUERIES, RESIDUES, EPOCHS = 1075, 404267, 50
GUARDRAIL, TARGET = .7417948466002535, .80


def read_json(path, watched):
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path in watched and watched[path] != digest:
        raise ValueError(f"Previously read input changed: {path}")
    watched[path] = digest
    return json.loads(raw)


def watch(path, watched):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path in watched and watched[path] != digest:
        raise ValueError(f"Previously read input changed: {path}")
    watched[path] = digest
    return digest


def terminal_state(job):
    raw = subprocess.check_output([
        "sacct", "-j", job, "-X", "--format=JobID%64,State%32,ExitCode,Elapsed",
        "-n", "-P",
    ], text=True)
    rows = [line.strip().split("|") for line in raw.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == job]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][1:3] != ["COMPLETED", "0:0"]:
        raise ValueError(f"Successful terminal state required for {job}: {rows}")
    return dict(job=job, state=rows[0][1], exit_code=rows[0][2], elapsed=rows[0][3])


def validate_epochs(records):
    epochs = [row for row in records if row.get("event") == "epoch"]
    if ([row["epoch"] for row in epochs] != list(range(1, EPOCHS + 1))
            or any("test" in row or row.get("split") == "test"
                   or row.get("event") == "test" for row in records)):
        raise ValueError("Incomplete or test-exposed training history")
    for row in epochs:
        v = row["val"]
        if (v.get("valid_chains") != QUERIES or v.get("valid_residues") != RESIDUES
                or v.get("prediction_records") != QUERIES or v.get("skipped_nonfinite") != 0
                or v.get("eval_grad_mode_enabled") is not False
                or v.get("eval_logits_require_grad") is not False
                or v.get("eval_parameter_grads_present") != 0
                or v.get("eval_parameter_version_changes") != 0
                or any(not math.isfinite(float(v[k])) or not 0 <= float(v[k]) <= 1
                       for k in ("chain_ap_macro", "pooled_ap"))):
            raise ValueError(f"Invalid validation epoch: {row['epoch']}")
    return epochs


def selected_epoch(epochs):
    eligible = [row for row in epochs if row["val"]["chain_ap_macro"] >= GUARDRAIL]
    if not eligible:
        raise ValueError("No epoch meets the unchanged macro AP guardrail")
    return max(eligible, key=lambda row: (row["val"]["pooled_ap"], -row["epoch"]))


def resolve(base, experiment, policy_path, arm):
    base, experiment, policy_path = map(lambda p: Path(p).resolve(), (base, experiment, policy_path))
    watched = {}
    policy = read_json(policy_path, watched)
    frozen_policy = experiment / "code/DOCs" / policy_path.name
    if watch(frozen_policy, watched) != watched[policy_path]:
        raise ValueError("Policy differs from the pre-training source snapshot")
    if (arm not in policy["arms"] or policy["alpha_search"] is not False
            or policy["test_evaluation"] is not False or policy["target_pooled_AP"] != TARGET
            or policy["averaging"] != "Equal clipped-probability logit mean;epsilon1e-7"):
        raise ValueError("Wrong prospective validation-only policy")
    selection = read_json(experiment / "selection_policy.json", watched)
    if (selection["macro_guardrail"] != GUARDRAIL or selection["epochs"] != EPOCHS
            or selection["seed"] != 49 or selection["target_pooled_AP"] != TARGET):
        raise ValueError("Selection contract changed")
    jobs = read_json(experiment / "jobs.json", watched)
    matches = [(i, job) for i, job in enumerate(jobs) if job["name"] == arm]
    if len(matches) != 1:
        raise ValueError("Arm must identify exactly one submitted task")
    index, job = matches[0]
    launch = read_json(experiment / "launch.json", watched)
    array = launch["job_array"]
    if not isinstance(array, str) or not array.isdigit():
        raise ValueError("Invalid job array identity")
    terminal = terminal_state(f"{array}_{index}")
    cfg = read_json(job["config"], watched)
    if watched[Path(job["config"]).resolve()] != job["sha256"]:
        raise ValueError("Submitted config changed")
    run = Path(job["output"]).resolve()
    if run != experiment / "runs" / f"{arm}_seed49" or cfg["training"]["output_dir"] != str(run):
        raise ValueError("Wrong run identity")
    runtime = read_json(run / "config.json", watched)
    for section in ("data", "training", "model"):
        actual = runtime.get("model_config", {}) if section == "model" else runtime
        if any(key not in actual or actual[key] != value for key, value in cfg[section].items()):
            raise ValueError(f"Submitted/runtime {section} configuration mismatch")
    required = dict(n_train=4868, n_val=QUERIES, n_test=0, seed=49, epochs=EPOCHS,
                    validation_only=True, eval_test_each_epoch=False, init_checkpoint=None,
                    eval_only=False, max_residues=0, train_crop_mode="none", eval_crop_mode="none")
    if any(k not in runtime or runtime[k] != value for k, value in required.items()):
        raise ValueError("Runtime data/training contract changed")
    metrics = (run / "metrics.jsonl").resolve()
    raw = metrics.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if metrics in watched and watched[metrics] != digest:
        raise ValueError("Previously read metrics path changed")
    watched[metrics] = digest
    epochs = validate_epochs([json.loads(line) for line in raw.splitlines() if line.strip()])
    best = selected_epoch(epochs)
    macro_best = max(epochs, key=lambda row: (row["val"]["chain_ap_macro"], -row["epoch"]))
    short = arm.split("_")[0]
    blend = f"G2_plus_MSA_D_plus_{short}"
    if (policy["fixed_blends"].get(blend) != ["G2", "MSA_D", short]
            or set(policy["fixed_sources"]) != {"G2", "MSA_D"}):
        raise ValueError("Blend membership was not prospectively declared")
    points = {}
    for name, source in policy["fixed_sources"].items():
        source_run = (base / source["run"]).resolve()
        if not source_run.is_relative_to(base) or source["epoch"] != {"G2": 27, "MSA_D": 13}[name]:
            raise ValueError("Wrong fixed reference source")
        points[name] = dict(epoch=source["epoch"], metrics=str(source_run / "metrics.jsonl"),
                            prediction=str(source_run / f"predictions/val_epoch_{source['epoch']:03d}.pt"))
    points[short] = dict(epoch=best["epoch"], metrics=str(metrics),
                         prediction=str(run / f"predictions/val_epoch_{best['epoch']:03d}.pt"))
    for point in points.values():
        for key in ("prediction", "metrics"):
            watch(point[key], watched)
    checkpoint = run / f"epochs/epoch_{best['epoch']:03d}.pt"
    checkpoint_sha = watch(checkpoint, watched)
    receipt = dict(**terminal, run=str(run), epochs=EPOCHS, coverage_ok_all_epochs=True,
                   validation_skips=0, test_seen=False, metrics_sha256=watched[metrics],
                   selected={"epoch": best["epoch"], **{k: best["val"][k]
                             for k in ("chain_ap_macro", "pooled_ap", "auroc")}},
                   macro_best={"epoch": macro_best["epoch"], **{k: macro_best["val"][k]
                               for k in ("chain_ap_macro", "pooled_ap")}},
                   checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                   AMP_skips=sum(row["train"].get("amp_skipped_steps", 0) for row in epochs),
                   sanitized_gradient_elements=sum(row["train"].get("sanitized_gradient_elements", 0)
                                                   for row in epochs))
    return dict(policy_path=str(policy_path), policy_sha256=watched[policy_path],
                completed_receipt=receipt, points=points, blends={blend: policy["fixed_blends"][blend]}), watched


def unchanged(watched):
    for path, digest in watched.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen readout input changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "experiment", "policy", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Output already exists; do not overwrite previous readouts")
    spec, watched = resolve(args.base, args.experiment, args.policy, args.arm)
    watch(args.base / "data/sequence_groups.tsv", watched)
    spec_path = args.output.with_name(args.output.name + ".resolved.json")
    with spec_path.open("x") as handle:
        handle.write(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    watch(spec_path, watched)
    subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("analyze_union_fixed_blends.py")),
                    "--base", str(args.base), "--spec", str(spec_path), "--output", str(args.output)], check=True)
    unchanged(watched)
    terminal_state(spec["completed_receipt"]["job"])
    with (args.output / "completion_audit.json").open("x") as handle:
        handle.write(json.dumps(dict(inputs={str(k): v for k, v in watched.items()},
                                    spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                                    source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    completed_receipt=spec["completed_receipt"], test_read=False,
                                    test_inference_permitted=False,
                                    limitation="Validation readout only;strict checkpoint lock/replay still required"),
                                indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
