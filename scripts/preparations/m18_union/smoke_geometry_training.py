"""Two or three real-payload updates and validation before each architecture run."""

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
    updates = 3 if cfg["model"].get("geometry_msa_edge_bias") else 2
    root = Path(t["output_dir"]).parent.parent / "smoke"
    output = root / (Path(t["output_dir"]).name + "_" + os.environ.get("SLURM_JOB_ID", "local"))
    path = output.with_suffix(".json")
    root.mkdir(parents=True, exist_ok=True)
    if path.exists() or output.exists():
        raise FileExistsError(output)
    t.update(output_dir=str(output), epochs=1, max_train_samples=updates, max_val_samples=2,
             batch_size=1, eval_batch_size=1, max_batch_tokens=512, num_workers=0,
             persistent_workers=False, checkpoint_every_epoch=False, save_epoch_predictions=True)
    if cfg["model"].get("geometry_msa_edge_bias"):
        t["weight_decay"] = 0.  # Parameter-change proof must not be explained by weight decay.
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-u", str(repo / "scripts/training/train_esm_site.py"), "--config", str(path)], check=True)
    record = next(json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()
                  if json.loads(x).get("event") == "epoch")
    assert record["train"]["optimizer_steps"] == updates and record["train"]["amp_skipped_steps"] == 0
    assert record["train"]["valid_chains"] == updates and record["val"]["valid_chains"] == 2
    assert record["val"]["skipped_nonfinite"] == 0 and record["val"]["successful_batches"] == 2
    assert record["val"]["prediction_records"] == 2
    assert all(math.isfinite(record[s]["loss"]) for s in ("train", "val"))
    assert "test" not in record
    ckpt = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["n_test"] == 0
    torch.manual_seed(49)
    initial = ESMSiteClassifier(cfg["model"]).state_dict()
    state = ckpt["model"]
    assert not torch.equal(state["esm_proj.0.weight"], initial["esm_proj.0.weight"])
    checks = [k for k in state if (k.startswith("geometry_blocks.") and k.endswith("update.4.weight"))
              or (k.startswith("linear_attention_blocks.") and k.endswith("output.weight"))]
    if cfg["model"].get("use_prottrans_embeddings"):
        checks.append("prottrans_residual_adapter.3.weight")
    for k in checks:
        assert torch.count_nonzero(state[k]) > 0, k
    learned_edge_parameters = []
    if cfg["model"].get("geometry_msa_edge_bias"):
        learned_edge_parameters = [k for k in state if ".evolution_bias." in k
                                  and (k.endswith("scale") or k.endswith("query.weight") or k.endswith("key.weight"))]
        assert len(learned_edge_parameters) == 3 * cfg["model"]["geometry_layers"]
        for k in learned_edge_parameters:
            assert not torch.equal(state[k], initial[k]), k
    assert record["val"]["eval_grad_mode_enabled"] is False
    assert record["val"]["eval_parameter_grads_present"] == 0
    assert record["val"]["eval_parameter_version_changes"] == 0
    (output / "verification.json").write_text(json.dumps(dict(optimizer_updates=updates, two_updates=updates == 2,
        finite=True, backbone_updated=True, new_zero_init_parameters_updated=checks,
        learned_edge_parameters_updated=learned_edge_parameters, test_read=False), indent=2) + "\n")
    print(json.dumps({"smoke": "passed", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
