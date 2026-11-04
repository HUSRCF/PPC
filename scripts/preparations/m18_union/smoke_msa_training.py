"""Two-update, real-payload scratch smoke before a formal MSA arm."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from ppcbind.models.esm_site import ESMSiteClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.config.read_text())
    if not original["data"].get("require_prottrans_embeddings"):
        return
    cfg = copy.deepcopy(original)
    t = cfg["training"]
    root = Path(t["output_dir"]).parent.parent
    smoke_root = root / "smoke" / (Path(t["output_dir"]).name + "_" + os.environ["SLURM_JOB_ID"])
    cfg_path = smoke_root.with_suffix(".json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists() or smoke_root.exists():
        raise FileExistsError(smoke_root)
    t.update(output_dir=str(smoke_root), epochs=1, max_train_samples=2, max_val_samples=2,
             batch_size=1, eval_batch_size=1, max_batch_tokens=512, num_workers=0,
             persistent_workers=False, checkpoint_every_epoch=False, save_epoch_predictions=False)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-u", str(repo / "scripts/training/train_esm_site.py"),
                    "--config", str(cfg_path)], check=True)
    records = [json.loads(line) for line in (smoke_root / "metrics.jsonl").read_text().splitlines()]
    record = next(r for r in records if r.get("event") == "epoch")
    assert record["train"]["optimizer_steps"] == 2
    assert record["train"]["amp_skipped_steps"] == 0
    assert math.isfinite(record["train"]["loss"]) and math.isfinite(record["val"]["loss"])
    assert record["val"]["skipped_nonfinite"] == 0
    assert record["val"]["valid_chains"] == 2
    assert record["val"]["successful_batches"] == 2
    assert record["train"]["valid_chains"] == 2
    assert "test" not in record
    checkpoint = torch.load(smoke_root / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["n_test"] == 0
    torch.manual_seed(t["seed"])
    initial = ESMSiteClassifier(cfg["model"])
    state = checkpoint["model"]
    assert not torch.equal(state["esm_proj.0.weight"], initial.state_dict()["esm_proj.0.weight"])
    assert torch.count_nonzero(state["prottrans_residual_adapter.3.weight"]) > 0
    (smoke_root / "verification.json").write_text(json.dumps({
        "two_optimizer_updates": True, "two_validation_chains": True,
        "zero_skipped_validation_batches": True, "finite_losses": True, "backbone_updated": True,
        "zero_init_adapter_updated": True, "from_scratch": True, "test_read": False
    }, indent=2) + "\n")
    print(json.dumps({"smoke": "passed", "path": str(smoke_root)}), flush=True)


if __name__ == "__main__":
    main()
