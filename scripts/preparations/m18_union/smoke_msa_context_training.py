"""Real-payload, two-update GPU acceptance before the sequence-context arms."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
from ppcbind.models import ESMSiteClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    cfg = copy.deepcopy(json.loads(args.config.read_text()))
    t = cfg["training"]
    root = Path(t["output_dir"]).parent.parent / "smoke"
    output = root / (Path(t["output_dir"]).name + "_" + os.environ["SLURM_JOB_ID"])
    path = output.with_suffix(".json")
    root.mkdir(parents=True, exist_ok=True)
    if path.exists() or output.exists():
        raise FileExistsError(output)
    t.update(output_dir=str(output), epochs=1, max_train_samples=2, max_val_samples=2,
             batch_size=1, eval_batch_size=1, max_batch_tokens=512, num_workers=0,
             persistent_workers=False, checkpoint_every_epoch=False, save_epoch_predictions=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-u", str(repo / "scripts/training/train_esm_site.py"), "--config", str(path)], check=True)
    record = next(r for r in map(json.loads, (output / "metrics.jsonl").read_text().splitlines()) if r.get("event") == "epoch")
    assert record["train"]["optimizer_steps"] == 2 and record["train"]["amp_skipped_steps"] == 0
    assert record["train"]["valid_chains"] == record["val"]["valid_chains"] == 2
    assert record["val"]["prediction_records"] == 2 and record["val"]["skipped_nonfinite"] == 0
    assert record["val"]["eval_parameter_version_changes"] == record["val"]["eval_parameter_grads_present"] == 0
    assert all(math.isfinite(record[s]["loss"]) for s in ("train", "val")) and "test" not in record
    ckpt = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["n_test"] == 0
    torch.manual_seed(49)
    initial = ESMSiteClassifier(cfg["model"]).state_dict()
    state = ckpt["model"]
    assert not torch.equal(state["esm_proj.0.weight"], initial["esm_proj.0.weight"])
    assert torch.count_nonzero(state["prottrans_residual_adapter.3.weight"]) > 0
    mode = cfg["model"]["post_fusion_context_mode"]
    key = ("post_fusion_context.context.weight_ih_l0" if mode == "bigru"
           else "post_fusion_context.context.0.weight" if mode == "mlp" else None)
    if key:
        assert not torch.equal(state[key], initial[key]), key
    (output / "verification.json").write_text(json.dumps(dict(two_updates=True, finite=True,
        backbone_updated=True, context_updated=key, MSA_adapter_updated=True,
        test_read=False, validation_isolated=True), indent=2) + "\n")
    print(json.dumps({"smoke": "passed", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
