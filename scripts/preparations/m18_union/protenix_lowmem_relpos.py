#!/data/user/shuang886/.conda/envs/fold/bin/python
"""Bitwise-equivalent row-chunked relative position features for Mini inference."""

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch
from torch.nn import functional as F

import protenix_full_esm as full


BASE_SOURCE_SHA256 = "6cf9adccdc89538eed7a8d7b3b6f6149849d192284d64186d849c210dbd0907b"
EMBEDDERS_SHA256 = "05e9e8d7bc8e9f917a6cadb3534b4b95764ea9968c19bc38d4743a29daf982f8"
FIELDS = ("asym_id", "residue_index", "entity_id", "sym_id", "token_index")
CHUNK_ROWS = 64


@torch.no_grad()
def generate_relp_chunked(self, features, chunk_rows=CHUNK_ROWS):
    values = [features[k] for k in FIELDS]
    first = values[0]
    if (chunk_rows <= 0 or first.ndim < 1 or first.shape[-1] < 1
            or any(x.shape != first.shape or x.device != first.device
                   or x.dtype not in (torch.int32, torch.int64) for x in values)):
        raise ValueError("Expected equal-shaped integer token fields and a positive row chunk")
    r, s = self.r_max, self.s_max
    if r < 0 or s < 0:
        raise ValueError("Negative relative position bounds")
    n = first.shape[-1]
    width = 4*r + 2*s + 7
    result = torch.empty((*first.shape[:-1], n, n, width), dtype=torch.float32, device=first.device)
    asym, residue, entity, sym, token = values
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        same_chain = (asym[..., start:end, None] == asym[..., None, :]).long()
        same_residue = (residue[..., start:end, None] == residue[..., None, :]).long()
        same_entity = (entity[..., start:end, None] == entity[..., None, :]).long()
        d_residue = torch.clip(residue[..., start:end, None] - residue[..., None, :] + r, 0, 2*r)
        d_residue = d_residue*same_chain + (1-same_chain)*(2*r+1)
        d_token = torch.clip(token[..., start:end, None] - token[..., None, :] + r, 0, 2*r)
        d_token = d_token*same_chain*same_residue + (1-same_chain*same_residue)*(2*r+1)
        d_chain = torch.clip(sym[..., start:end, None] - sym[..., None, :] + s, 0, 2*s)
        d_chain = d_chain*same_entity + (1-same_entity)*(2*s+1)
        block = torch.cat((F.one_hot(d_residue, 2*(r+1)), F.one_hot(d_token, 2*(r+1)),
                           same_entity[..., None], F.one_hot(d_chain, 2*(s+1))), dim=-1).float()
        result[..., start:end, :, :] = block
    features["relp"] = result
    return features


def checked_runtime():
    if full.digest(__file__) != os.environ["PPC_LOWMEM_RELP_SOURCE_SHA256"]:
        raise ValueError("Low-memory adapter source changed")
    if full.digest(full.__file__) != BASE_SOURCE_SHA256:
        raise ValueError("Full-query cache adapter changed")
    full.runtime()
    module = importlib.import_module("protenix.model.modules.embedders")
    if full.digest(module.__file__) != EMBEDDERS_SHA256:
        raise ValueError("Stock relative-position module changed")
    return module


def synthetic(n, batch_shape=(), dtype=torch.int64, mixed=False):
    base = torch.arange(n, dtype=dtype).expand(*batch_shape, n).clone()
    return dict(asym_id=(base//3 if mixed else torch.ones_like(base)),
                residue_index=(base//2 if mixed else base), entity_id=(base//5 if mixed else torch.ones_like(base)),
                sym_id=(base%3 if mixed else torch.zeros_like(base)), token_index=base*2)


def prove(output, full_n):
    if full_n != full.MAX_QUERY_LENGTH:
        raise ValueError("Proof must cover the complete historical maximum, not only a small smoke")
    module = checked_runtime()
    original = module.RelativePositionEncoding.generate_relp
    if Path(inspect.getsourcefile(original)).resolve() != Path(module.__file__).resolve():
        raise ValueError("Stock method is already replaced")
    torch.cuda.init()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    cases = []
    for n, batch, r, s, dtype in ((1, (), 32, 2, torch.int64), (7, (), 3, 0, torch.int32),
                                  (13, (2,), 0, 1, torch.int64), (17, (2,), 32, 2, torch.int64)):
        obj = SimpleNamespace(r_max=r, s_max=s)
        fields = synthetic(n, batch, dtype, mixed=True)
        expected = original(obj, dict(fields))["relp"]
        for device in ("cpu", "cuda"):
            actual = generate_relp_chunked(obj, {k: v.to(device) for k, v in fields.items()}, 3)["relp"]
            if not torch.equal(expected, actual.cpu()) or actual.dtype != torch.float32 or not actual.is_contiguous():
                raise ValueError("Stock/row-chunked relative position mismatch")
        cases.append(dict(n=n, batch_shape=batch, r_max=r, s_max=s, dtype=str(dtype), CPU_equal=True, GPU_equal=True))
    del expected, actual
    fields = synthetic(full_n)
    obj = SimpleNamespace(r_max=32, s_max=2)
    start = time.perf_counter()
    reference = original(obj, dict(fields))["relp"]
    original_seconds = time.perf_counter()-start
    gpu_fields = {k: v.cuda() for k, v in fields.items()}
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    candidate = generate_relp_chunked(obj, gpu_fields)["relp"]
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter()-start
    equal = torch.equal(reference, candidate.cpu())
    peak = torch.cuda.max_memory_allocated()
    if not equal or candidate.dtype != reference.dtype or not candidate.is_contiguous():
        raise ValueError("Full4579stockCPU versus chunkedGPU mismatch")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise ValueError("Allocation optimization unexpectedly consumed RNG")
    record = dict(status="passed", source_sha256=full.digest(__file__), base_source_sha256=BASE_SOURCE_SHA256,
                  stock_embedders_sha256=EMBEDDERS_SHA256, chunk_rows=CHUNK_ROWS, cases=cases,
                  full_n=full_n, full_width=139, full_CPU_stock_vs_GPU_chunked_equal=True,
                  RNG_unchanged=True, dtype="torch.float32", contiguous=True,
                  original_CPU_seconds=original_seconds, chunked_GPU_seconds=gpu_seconds,
                  chunked_GPU_peak_bytes=peak, model_weights_or_input_changed=False,
                  time=time.time(), scope="Operator equivalence only;not a structure-folding success")
    del reference, candidate, gpu_fields, fields
    torch.cuda.empty_cache()
    checked_runtime()
    output = Path(output)
    if output.exists():
        raise ValueError("Do not overwrite an accepted operator proof")
    output.parent.mkdir(parents=True, exist_ok=True)
    full.write_atomic(output, json.dumps(record, indent=2).encode())
    print(json.dumps(record), flush=True)


def checked_proof():
    raw = Path(os.environ["PPC_LOWMEM_RELP_ACCEPTANCE"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ["PPC_LOWMEM_RELP_ACCEPTANCE_SHA256"]:
        raise ValueError("Operator proof changed")
    proof = json.loads(raw)
    expected = dict(status="passed", source_sha256=full.digest(__file__), base_source_sha256=BASE_SOURCE_SHA256,
                    stock_embedders_sha256=EMBEDDERS_SHA256, chunk_rows=CHUNK_ROWS, full_n=full.MAX_QUERY_LENGTH,
                    full_width=139, full_CPU_stock_vs_GPU_chunked_equal=True, RNG_unchanged=True,
                    dtype="torch.float32", contiguous=True, model_weights_or_input_changed=False)
    if any(proof[k] != value for k, value in expected.items()):
        raise ValueError("Incomplete or wrong full-length operator proof")
    return proof


def predict(argv):
    args, rows, _ = full.input_rows(argv)
    checked_proof()
    module = checked_runtime()
    cls = module.RelativePositionEncoding
    original = cls.generate_relp
    if Path(inspect.getsourcefile(original)).resolve() != Path(module.__file__).resolve():
        raise ValueError("Stock method is already replaced")
    output = Path(args.out_dir).resolve()
    unpublished = output.with_name(output.name + ".lowmem_unaccepted")
    if output.exists() or unpublished.exists():
        raise ValueError("Fresh output directory required")
    adapted = list(argv); adapted[adapted.index("--out_dir")+1] = str(unpublished)
    def replacement(self, features):
        return generate_relp_chunked(self, features)
    try:
        cls.generate_relp = replacement
        try:
            full.fold(adapted)
            if cls.generate_relp is not replacement:
                raise ValueError("In-process relative-position hook changed")
        finally:
            cls.generate_relp = original
        checked_runtime(); checked_proof()
        unpublished.replace(output)
    except (Exception, SystemExit) as error:
        errors = Path.cwd()/"esm_embeddings/esm_error"; errors.mkdir(parents=True, exist_ok=True)
        for uid in rows:
            p = errors/(uid+"_1.txt")
            if not p.exists():
                full.write_atomic(p, ("Low-memory adapter rejected unpublished output: "+repr(error)).encode())
        raise
    print(json.dumps(dict(event="relative_position_adapter_returned", chunk_rows=CHUNK_ROWS,
                          installed_files_modified=False, inference_parameters_changed=False)), flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=="validate":
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["validate"])
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--full-n", required=True, type=int)
        args = parser.parse_args(); prove(args.output, args.full_n)
    else:
        predict(sys.argv[1:])


if __name__ == "__main__":
    main()
