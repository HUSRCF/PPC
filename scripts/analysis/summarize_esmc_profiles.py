#!/usr/bin/env python3
"""Summarize ESM-C throughput profiles from epoch and nvidia-smi logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


def _number(value: str) -> float | None:
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    return float(match.group(0)) if match else None


def _gpu_metrics(path: Path) -> dict[str, float | int | None]:
    util: list[float] = []
    memory: list[float] = []
    power: list[float] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            gpu_util = _number(row.get("utilization_gpu", ""))
            memory_used = _number(row.get("memory_used", ""))
            power_draw = _number(row.get("power_draw", ""))
            if gpu_util is not None:
                util.append(gpu_util)
            if memory_used is not None:
                memory.append(memory_used)
            if power_draw is not None:
                power.append(power_draw)
    return {
        "gpu_samples": len(util),
        "gpu_util_mean": statistics.fmean(util) if util else None,
        "gpu_util_median": statistics.median(util) if util else None,
        "gpu_active_fraction": sum(value >= 50.0 for value in util) / len(util) if util else None,
        "gpu_memory_peak_mib": max(memory) if memory else None,
        "gpu_power_mean_w": statistics.fmean(power) if power else None,
    }


def _epoch_metrics(run_dir: Path) -> dict[str, Any]:
    records = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "epoch":
                records.append(record)
    if not records:
        return {"epochs_completed": 0}
    record = records[-1]
    train = record.get("train", {})
    return {
        "epochs_completed": len(records),
        "profile_epoch": record.get("epoch"),
        "epoch_seconds_with_eval": record.get("seconds"),
        "train_loop_seconds": train.get("train_loop_seconds"),
        "data_wait_seconds": train.get("data_wait_seconds"),
        "compute_seconds": train.get("compute_seconds"),
        "data_wait_fraction": train.get("data_wait_fraction"),
        "residues_per_second": train.get("residues_per_second"),
        "val_f1": record.get("val", {}).get("f1_best_threshold"),
        "val_auprc": record.get("val", {}).get("pr_auc"),
    }


def _parse_slurm_header(path: Path) -> tuple[str | None, str | None]:
    config = None
    gpu_log = None
    if not path.exists():
        return config, gpu_log
    with path.open(errors="replace") as handle:
        for _ in range(80):
            line = handle.readline()
            if not line:
                break
            if line.startswith("Config: "):
                config = line.split(": ", 1)[1].strip()
            elif line.startswith("GPU log: "):
                gpu_log = line.split(": ", 1)[1].strip()
    return config, gpu_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    rows: list[dict[str, Any]] = []
    for out_path in sorted((root / "logs").glob("ppc_esmc_*.out")):
        match = re.search(r"_(\d+)\.out$", out_path.name)
        if not match:
            continue
        job_id = match.group(1)
        config_path, gpu_log_path = _parse_slurm_header(out_path)
        if not config_path or "/profiles/" not in config_path:
            continue
        config = root / config_path
        config_text = config.read_text() if config.exists() else ""
        output_match = re.search(r"^\s*output_dir:\s*(\S+)\s*$", config_text, flags=re.MULTILINE)
        run_dir = root / output_match.group(1) if output_match else root / "runs" / "missing"
        gpu_log = Path(gpu_log_path) if gpu_log_path else root / "logs" / f"ppc_esmc_{job_id}_gpu.csv"
        row: dict[str, Any] = {
            "job_id": job_id,
            "profile": config.stem,
            "config": config_path,
            "run_dir": str(run_dir),
        }
        row.update(_epoch_metrics(run_dir))
        row.update(_gpu_metrics(gpu_log) if gpu_log.exists() else {"gpu_samples": 0})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No completed profile logs under {root / 'logs'}")

    fields = sorted({key for row in rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "tsv": str(args.output), "json": str(json_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
