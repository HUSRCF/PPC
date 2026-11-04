"""Repair RCSB observed-sequence monomers, including the previous ge100 short-chain gap."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import gemmi


PARENT_QUEUE_SHA256 = "0ac7ba922146322b14d69542bf1bd61f7abaaaa125a4c1d93070d3395ec8a85e"
_QC_TORCH = None


def preflight_qc_runtime():
    global _QC_TORCH
    if _QC_TORCH is None:
        import torch
        data = io.BytesIO()
        torch.save(torch.ones(1), data); data.seek(0)
        if not bool(torch.isfinite(torch.load(data, weights_only=True)).all()):
            raise RuntimeError("QC tensor serialization is unavailable")
        _QC_TORCH = torch
    return _QC_TORCH


def read_manifest(path, min_length=50):
    raw = Path(path).resolve(strict=True).read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("ascii")), delimiter="\t"))
    if not rows or len({r["unique_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate repair manifest")
    for r in rows:
        seq = r["sequence"]
        if (len(seq) != int(r["n_residues"]) or len(seq) < min_length
                or set(seq)-set("ACDEFGHIKLMNPQRSTVWYX")
                or hashlib.sha256(seq.encode("ascii")).hexdigest() != r["seq_sha256"]
                or r["unique_id"] != "useq_"+r["seq_sha256"]):
            raise ValueError("Invalid or below-minimum query;no silent filtering: "+r["unique_id"])
    return sorted(rows, key=lambda r: r["unique_id"]), hashlib.sha256(raw).hexdigest()


def valid_esm_input(directory, row):
    if directory.parent.name != "outputs":
        return False
    work = directory.parent.parent
    label = row["unique_id"]+"_1"
    base = work/"esm_embeddings"
    if (base/"esm_error"/(label+".txt")).exists():
        return False
    with (base/"input_prot_sequences.csv").open() as f:
        matches = [r for r in csv.DictReader(f) if r["seq_label"] == label]
    if (len(matches) != 1 or matches[0]["seq"] != row["sequence"]
            or matches[0]["part_id"] != label):
        return False
    # The SDK catches ESM failures and can continue with zero features; reject that fallback.
    torch = preflight_qc_runtime()
    value = torch.load(base/"esm2-3b"/label/(label+".pt"), map_location="cpu", weights_only=True)
    return (isinstance(value, torch.Tensor) and tuple(value.shape) == (len(row["sequence"]), 2560)
            and not value.requires_grad and bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0))


def validate(directory, row):
    preflight_qc_runtime()
    directory = directory.resolve()
    files = list(directory.glob("seed_101/predictions/*sample_0.cif"))
    if len(files) != 1:
        return None
    try:
        structure = gemmi.read_structure(str(files[0]))
        if len(structure) != 1 or len(structure[0]) != 1:
            return None
        residues = list(structure[0][0])
        sequence = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in residues)
        if sequence != row["sequence"] or any(not r.find_atom("CA", "*") for r in residues):
            return None
        if not all(math.isfinite(v) for r in residues for a in r for v in (a.pos.x, a.pos.y, a.pos.z, a.b_iso)):
            return None
        summary = files[0].parent / (row["unique_id"]+"_summary_confidence_sample_0.json")
        if not summary.is_file():
            return None
        confidence = list(directory.rglob("*.json"))
        if not confidence:
            return None
        for path in confidence:
            json.loads(path.read_text())
        if not valid_esm_input(directory, row):
            return None
        return str(files[0].resolve())
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--protenix-cli", type=Path)
    p.add_argument("--reuse", type=Path)
    p.add_argument("--batch-size", type=int, default=128, help="Targets per CLI process,not a neural-network batch size")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--min-length", type=int, default=50)
    args = p.parse_args()
    if args.batch_size <= 0 or args.max_attempts <= 0 or args.min_length < 50:
        raise ValueError("Invalid process chunk,retry budget or minimum length")
    preflight_qc_runtime()
    args.manifest = args.manifest.resolve(strict=True)
    args.project = args.project.resolve()
    if args.protenix_cli is not None:
        args.protenix_cli = args.protenix_cli.resolve()
    if args.reuse is not None:
        args.reuse = args.reuse.resolve()
    rows, manifest_sha256 = read_manifest(args.manifest, args.min_length)
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = dict(manifest_sha256=manifest_sha256,
                    min_length=args.min_length, model="protenix_mini_esm_v0.5.0", seed=101,
                    sample=1, msa=False, order="unique_id ascending", dtype="bf16",
                    process_chunk_size=args.batch_size, max_attempts=args.max_attempts,
                    filter_policy="All manifest queries required;no length-based omission",
                    source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    parent_queue_sha256=PARENT_QUEUE_SHA256)
    contract_path = args.root / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Changed queue contract")
    else:
        contract_path.write_text(json.dumps(contract, indent=2))
    state = {}
    ledger = args.root / "progress.jsonl"
    if ledger.exists():
        lines = ledger.read_text().splitlines(keepends=True)
        # A killed write may leave only the final JSONL record incomplete.
        if lines and not lines[-1].endswith("\n"):
            (args.root / ("ledger_tail_" + str(time.time_ns()) + ".txt")).write_text(lines[-1])
            ledger.write_text("".join(lines[:-1]))
            lines = lines[:-1]
        for line in lines:
            record = json.loads(line)
            state[record["unique_id"]] = record
    handle = ledger.open("a", buffering=1)

    def record(row, **values):
        event = dict(unique_id=row["unique_id"], n_residues=int(row["n_residues"]),
                     time=time.time(), **values)
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        state[row["unique_id"]] = event

    def summary():
        counts = {}
        for r in rows:
            status = state.get(r["unique_id"], {}).get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        data = dict(total=len(rows), counts=counts, time=time.time())
        temp = args.root / "summary.tmp"
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(args.root / "summary.json")
        print(json.dumps(data), flush=True)

    for row in rows:
        old = state.get(row["unique_id"], {})
        directory = Path(old["output_dir"]) if old.get("output_dir") else None
        cif = validate(directory, row) if directory else None
        if cif:
            record(row, status="ok", attempt=old.get("attempt", 0), output_dir=str(directory), cif=cif)
        elif old.get("status") == "ok":
            record(row, status="invalidated", attempt=0, reason="Existing output failed resume QC")
        elif old.get("status") == "started":
            record(row, **{k: v for k, v in old.items() if k not in ("unique_id", "n_residues", "time", "status", "attempt")},
                   attempt=max(0, old.get("attempt", 1)-1), status="interrupted",
                   reason="Interrupted attempt does not consume completed-failure retry budget")
        elif not old and args.reuse:
            directory = args.reuse / row["unique_id"]
            cif = validate(directory, row)
            if cif:
                record(row, status="ok", attempt=0, output_dir=str(directory), cif=cif, reused=True)
    summary()
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    zero_batches = 0
    while not stopping:
        pending = [r for r in rows if state.get(r["unique_id"], {}).get("status") != "ok"
                   and state.get(r["unique_id"], {}).get("attempt", 0) < args.max_attempts]
        if not pending:
            break
        # Retry failed sequences alone in a fresh process; no infinite OOM loop.
        batch = pending[:1] if state.get(pending[0]["unique_id"], {}).get("attempt", 0) else pending[:args.batch_size]
        work = args.root / "attempts" / str(time.time_ns())
        work.mkdir(parents=True)
        input_path = work / "input.json"
        input_path.write_text(json.dumps([dict(name=r["unique_id"], sequences=[dict(proteinChain=dict(
            sequence=r["sequence"], count=1, id=["A"]))]) for r in batch]))
        for row in batch:
            record(row, status="started", attempt=state.get(row["unique_id"], {}).get("attempt", 0)+1,
                   output_dir=str(work / "outputs" / row["unique_id"]), log=str(work / "inference.log"))
        executable = args.protenix_cli or args.project / ".venv/bin/protenix"
        cmd = [str(executable), "pred", "--input", str(input_path),
               "--out_dir", str(work / "outputs"), "--model_name", contract["model"],
               "--use_default_params", "true", "--use_msa", "false", "--sample", "1", "--seeds", "101",
               "--dtype", "bf16", "--trimul_kernel", "torch", "--triatt_kernel", "torch", "--enable_fusion", "false"]
        (work / "command.json").write_text(json.dumps(cmd, indent=2))
        with (work / "inference.log").open("w") as log:
            if stopping:
                break
            # Keep ownership locked if the supervisor is killed before this inference child exits.
            child = subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
            # A signal can arrive between the pre-launch check and assigning child.
            if stopping and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            code = child.wait()
        child = None
        if stopping:
            break
        text = (work / "inference.log").read_text(errors="replace")
        success = 0
        for row in batch:
            old = state[row["unique_id"]]
            cif = validate(Path(old["output_dir"]), row)
            errors = list((work / "outputs").rglob(row["unique_id"] + ".txt"))
            errors += list((work/"esm_embeddings/esm_error").glob(row["unique_id"]+"_*.txt"))
            detail = "\n".join(x.read_text(errors="replace") for x in errors)
            if not detail and len(batch) == 1:
                detail = text
            oom = "out of memory" in detail.lower() or "cuda error: memory allocation" in detail.lower()
            status = "ok" if cif else ("oom" if oom else "failed")
            success += bool(cif)
            record(row, status=status, attempt=old["attempt"], output_dir=old["output_dir"],
                   cif=cif, log=str(work / "inference.log"), exit_code=code,
                   error_files=[str(x) for x in errors], error_excerpt=detail[-2000:],
                   batch_oom_evidence="out of memory" in text.lower())
        summary()
        diagnostics = list((work/"outputs").rglob("*.txt")) + list((work/"esm_embeddings/esm_error").glob("*.txt"))
        infrastructure_failure = not success and code != 0 and not diagnostics
        zero_batches = zero_batches + 1 if infrastructure_failure else 0
        if zero_batches >= 3:
            print("Three process-level failures without per-sequence diagnostics: stopping for infrastructure review", flush=True)
            break
    summary()
    if any(state.get(r["unique_id"], {}).get("status") != "ok" for r in rows):
        raise SystemExit(143 if stopping else 2)


if __name__ == "__main__":
    main()
