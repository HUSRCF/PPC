#!/usr/bin/env python3
"""Run GraphPPIS-fast on current PPC chain manifest and emit residue TSV.

This runner is for the strict residue effect-site benchmark where structures are
deduplicated by sequence. It runs GraphPPIS once per unique structure in
`run_list.tsv`, then maps the residue scores back to every `seq_id` listed for
that unique sequence.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphppis-dir", required=True, type=Path)
    parser.add_argument("--run-list", required=True, type=Path)
    parser.add_argument("--chain-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--dssp-bin", default="/usr/bin/mkdssp")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-workdirs", action="store_true")
    return parser.parse_args()


def load_chain_labels(path: Path) -> dict[str, dict[str, list[Any]]]:
    labels: dict[str, dict[str, list[Any]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            seq_id = row["seq_id"]
            rec = labels.setdefault(seq_id, {"residue": [], "label": []})
            rec["residue"].append(row["residue"])
            rec["label"].append(int(row["label"]))
    return labels


def setup_runtime(graphppis_dir: Path, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name in ("GraphPPIS_predict.py", "getchain.pl", "caldis_CA", "blosum_dict.pkl"):
        dst = runtime_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(graphppis_dir / name)
    model_dst = runtime_dir / "Model"
    if model_dst.exists() or model_dst.is_symlink():
        model_dst.unlink()
    model_dst.symlink_to(graphppis_dir / "Model")


def parse_result(path: Path) -> tuple[list[str], list[float]]:
    aas: list[str] = []
    scores: list[float] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("The threshold") or line.startswith("AA\t"):
                continue
            aa, prob, _pred = line.split("\t")
            aas.append(aa)
            scores.append(float(prob))
    return aas, scores


def detect_chain_id(pdb_path: Path) -> str:
    with pdb_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                chain = line[21].strip()
                return chain or "A"
    return "A"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def run_one(task: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    runtime_dir = task["runtime_dir"]
    tmp_dir = task["tmp_dir"]
    setup_runtime(task["graphppis_dir"], runtime_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pdb = tmp_dir / f"{task['input_id']}.pdb"
    shutil.copyfile(task["pdb_path"], tmp_pdb)
    chain_id = detect_chain_id(tmp_pdb)

    before = {p.name for p in runtime_dir.glob("data_*")}
    env = os.environ.copy()
    env["GRAPHPPIS_DSSP"] = task["dssp_bin"]
    env["GRAPHPPIS_DEVICE"] = "cpu"
    env.setdefault("GRAPHPPIS_NUM_WORKERS", "0")
    proc = subprocess.run(
        [task["python_bin"], "GraphPPIS_predict.py", "-f", str(tmp_pdb), "-c", chain_id, "-m", "fast"],
        cwd=runtime_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    after = {p.name for p in runtime_dir.glob("data_*")}
    new_dirs = sorted(after - before)
    if proc.returncode != 0 or not new_dirs:
        errors.append(
            {
                "input_id": task["input_id"],
                "status": "RUN_FAIL",
                "chain_id": chain_id,
                "log_tail": proc.stdout[-3000:],
            }
        )
    else:
        result_path = runtime_dir / new_dirs[-1] / f"user{chain_id}_pred_results.txt"
        if not result_path.exists():
            errors.append(
                {
                    "input_id": task["input_id"],
                    "status": "NO_OUTPUT",
                    "chain_id": chain_id,
                    "log_tail": proc.stdout[-3000:],
                }
            )
        else:
            pred_aas, scores = parse_result(result_path)
            for seq_id in task["seq_ids"]:
                label_rec = task["labels"].get(seq_id)
                if label_rec is None:
                    errors.append({"input_id": task["input_id"], "seq_id": seq_id, "status": "MISSING_LABEL"})
                    continue
                residues = label_rec["residue"]
                labels = label_rec["label"]
                status = "OK"
                if len(pred_aas) != len(residues):
                    status = "LENGTH_MISMATCH"
                elif any(a != b for a, b in zip(pred_aas, residues)):
                    status = "AA_MISMATCH"
                n = min(len(pred_aas), len(residues), len(scores), len(labels))
                for idx in range(n):
                    rows.append(
                        {
                            "seq_id": seq_id,
                            "position": idx + 1,
                            "residue": residues[idx],
                            "score": scores[idx],
                            "method": task["method"],
                            "input_id": task["input_id"],
                            "status": status,
                        }
                    )
                if status != "OK":
                    errors.append(
                        {
                            "input_id": task["input_id"],
                            "seq_id": seq_id,
                            "status": status,
                            "n_label": len(residues),
                            "n_pred": len(pred_aas),
                        }
                    )
    if not task["keep_workdirs"]:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return task["task_index"], rows, errors


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_chain_labels(args.chain_labels)
    tasks: list[dict[str, Any]] = []
    with args.run_list.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            seq_ids = [x for x in row["chain_ids"].replace(",", ";").split(";") if x]
            if not seq_ids:
                continue
            pdb_path = Path(row["pdb_path"])
            if not pdb_path.exists():
                # Allow star absolute paths to be mirrored under local /media root.
                text = str(pdb_path)
                marker = "/data/husrcf/ProtBind/PPC/benchmark/"
                if marker in text:
                    local = Path("/media/990Pro/ProtBind/PPC/benchmark") / text.split(marker, 1)[1]
                    pdb_path = local
            task_index = len(tasks) + 1
            task_name = f"{task_index:05d}_{safe_name(row['input_id'])}"
            tasks.append(
                {
                    "task_index": task_index,
                    "input_id": row["input_id"],
                    "pdb_path": pdb_path,
                    "seq_ids": seq_ids,
                    "labels": labels,
                    "graphppis_dir": args.graphppis_dir,
                    "python_bin": args.python_bin,
                    "dssp_bin": args.dssp_bin,
                    "runtime_dir": args.output_dir / "graphppis_runtime" / task_name,
                    "tmp_dir": args.output_dir / "tmp_inputs" / task_name,
                    "keep_workdirs": args.keep_workdirs,
                    "method": "graphppis_fast_pypropel_no_len_limit",
                }
            )
    if args.limit > 0:
        tasks = tasks[: args.limit]

    t0 = time.time()
    pred_path = args.output_dir / "graphppis_fast_residue_predictions.with_status.tsv"
    clean_pred_path = args.output_dir / "graphppis_fast_residue_predictions.tsv"
    errors: list[dict[str, Any]] = []
    rows_written = 0
    ok_rows_written = 0
    fieldnames = ["seq_id", "position", "residue", "score", "method", "input_id", "status"]
    with pred_path.open("w", newline="") as full_handle, clean_pred_path.open("w", newline="") as clean_handle:
        full_writer = csv.DictWriter(full_handle, fieldnames=fieldnames, delimiter="\t")
        clean_writer = csv.DictWriter(
            clean_handle, fieldnames=["seq_id", "position", "residue", "score", "method"], delimiter="\t"
        )
        full_writer.writeheader()
        clean_writer.writeheader()
        with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {executor.submit(run_one, task): task for task in tasks}
            for completed, future in enumerate(futures.as_completed(future_map), 1):
                task = future_map[future]
                try:
                    task_index, rows, task_errors = future.result()
                except Exception as exc:  # noqa: BLE001
                    task_index = int(task["task_index"])
                    rows = []
                    task_errors = [{"input_id": task["input_id"], "status": "EXCEPTION", "error": repr(exc)}]
                errors.extend(task_errors)
                for row in rows:
                    full_writer.writerow(row)
                    rows_written += 1
                    if row["status"] == "OK":
                        clean_writer.writerow(
                            {
                                "seq_id": row["seq_id"],
                                "position": row["position"],
                                "residue": row["residue"],
                                "score": row["score"],
                                "method": row["method"],
                            }
                        )
                        ok_rows_written += 1
                print(
                    f"[{completed}/{len(tasks)} task={task_index}] rows={len(rows)} errors={len(task_errors)}",
                    flush=True,
                )

    summary = {
        "method": "graphppis_fast_pypropel_no_len_limit",
        "run_list": str(args.run_list),
        "chain_labels": str(args.chain_labels),
        "predictions_tsv": str(clean_pred_path),
        "predictions_with_status_tsv": str(pred_path),
        "n_tasks": len(tasks),
        "n_rows": rows_written,
        "n_ok_rows": ok_rows_written,
        "n_errors": len(errors),
        "errors_first": errors[:200],
        "elapsed_sec": time.time() - t0,
    }
    (args.output_dir / "graphppis_fast_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
