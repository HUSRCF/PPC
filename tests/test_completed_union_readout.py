import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("completed_readout", ROOT / "scripts/analysis/read_completed_union_arm.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def history():
    return [dict(event="epoch", epoch=i, train=dict(amp_skipped_steps=0),
                 val=dict(valid_chains=1075, valid_residues=404267, prediction_records=1075,
                          skipped_nonfinite=0, eval_grad_mode_enabled=False, eval_logits_require_grad=False,
                          eval_parameter_grads_present=0, eval_parameter_version_changes=0,
                          chain_ap_macro=.76, pooled_ap=.79, auroc=.89)) for i in range(1, 51)]


class CompletedReadoutTests(unittest.TestCase):
    def test_epoch_selection_uses_guardrail_and_earliest_tie(self):
        rows = history()
        rows[1]["val"]["pooled_ap"] = .8
        rows[2]["val"]["pooled_ap"] = .8
        rows[3]["val"].update(pooled_ap=.9, chain_ap_macro=.7)
        self.assertEqual(reader.selected_epoch(reader.validate_epochs(rows))["epoch"], 2)

    def test_incomplete_duplicate_test_or_incomplete_coverage_is_rejected(self):
        variants = [history()[:-1], history() + history()[-1:]]
        for key, value in (("valid_residues", 404266), ("prediction_records", 1074),
                           ("skipped_nonfinite", 1), ("eval_grad_mode_enabled", True),
                           ("eval_parameter_version_changes", 1), ("pooled_ap", float("nan"))):
            rows = history(); rows[9]["val"][key] = value; variants.append(rows)
        rows = history(); rows[10]["test"] = {}; variants.append(rows)
        rows = history(); rows.append(dict(event="test")); variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                reader.validate_epochs(rows)

    def test_empty_guardrail_is_not_relaxed(self):
        rows = history()
        for row in rows: row["val"]["chain_ap_macro"] = .7
        with self.assertRaises(ValueError): reader.selected_epoch(rows)

    def test_scheduler_requires_exact_successful_task(self):
        for raw in ("800_1|RUNNING|0:0|1:00\n", "800_1|FAILED|1:0|1:00\n",
                    "800_0|COMPLETED|0:0|1:00\n", ""):
            with patch.object(reader.subprocess, "check_output", return_value=raw):
                with self.assertRaises(ValueError): reader.terminal_state("800_1")
        with patch.object(reader.subprocess, "check_output", return_value="800_1|COMPLETED|0:0|1:00\n"):
            self.assertEqual(reader.terminal_state("800_1")["state"], "COMPLETED")

    def fixture(self, root):
        base = root; experiment = root / "exp"; arm = "K26_G2_MSA_twohop_preTCN_pure_CE"
        (root / "data").mkdir()
        (root / "data/sequence_groups.tsv").write_text("union_seq_id\tsplit\tlength\n")
        run = experiment / "runs" / f"{arm}_seed49"; run.mkdir(parents=True)
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
        policy = dict(arms=[arm], alpha_search=False, test_evaluation=False, target_pooled_AP=.8,
                      averaging="Equal clipped-probability logit mean;epsilon1e-7",
                      fixed_sources={"G2":dict(run="g2", epoch=27), "MSA_D":dict(run="msa", epoch=13)},
                      fixed_blends={"G2_plus_MSA_D_plus_K26":["G2", "MSA_D", "K26"]})
        p = experiment / "policy.json"; write(p, policy); write(experiment / "code/DOCs/policy.json", policy)
        config = experiment / "config.json"; write(config,dict(data={},model={},training=dict(output_dir=str(run),lr=8e-5)))
        watched = {}; checksum = reader.watch(config, watched)
        write(experiment / "jobs.json", [dict(name=arm, config=str(config), sha256=checksum, output=str(run))])
        write(experiment / "launch.json",dict(job_array="800"))
        write(experiment / "selection_policy.json",dict(macro_guardrail=reader.GUARDRAIL, epochs=50, seed=49, target_pooled_AP=.8))
        write(run / "config.json",dict(output_dir=str(run),lr=8e-5,model_config={},n_train=4868,n_val=1075,n_test=0,seed=49,epochs=50,validation_only=True,
              eval_test_each_epoch=False,init_checkpoint=None,eval_only=False,max_residues=0,train_crop_mode="none",eval_crop_mode="none"))
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in history()))
        for folder, epoch in ((root / "g2",27),(root / "msa",13),(run,1)):
            pred = folder / f"predictions/val_epoch_{epoch:03d}.pt"; pred.parent.mkdir(parents=True); pred.write_bytes(b"opaque validation prediction")
            if folder != run: (folder / "metrics.jsonl").write_text("reference metrics")
        checkpoint = run / "epochs/epoch_001.pt"; checkpoint.parent.mkdir(); checkpoint.write_bytes(b"opaque checkpoint")
        return base, experiment, p, arm, run

    def test_resolve_only_declared_point_and_hashes_then_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            base, experiment, p, arm, run = self.fixture(Path(temp))
            receipt = dict(job="800_0",state="COMPLETED",exit_code="0:0",elapsed="1:00")
            with patch.object(reader,"terminal_state",return_value=receipt):
                result, watched = reader.resolve(base,experiment,p,arm)
            self.assertEqual(result["points"]["K26"]["epoch"],1)
            self.assertEqual(result["blends"],{"G2_plus_MSA_D_plus_K26":["G2","MSA_D","K26"]})
            reader.unchanged(watched)
            (run/"metrics.jsonl").write_text("changed")
            with self.assertRaises(ValueError):reader.unchanged(watched)

    def test_no_analysis_or_output_for_live_task(self):
        with tempfile.TemporaryDirectory() as temp:
            base,experiment,p,arm,run=self.fixture(Path(temp))
            with patch.object(reader,"terminal_state",side_effect=ValueError("RUNNING")):
                with self.assertRaises(ValueError):reader.resolve(base,experiment,p,arm)
            self.assertEqual(set(run.iterdir()),{run/"config.json",run/"metrics.jsonl",run/"predictions",run/"epochs"})

    def test_changed_policy_or_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base,experiment,p,arm,run=self.fixture(Path(temp))
            original=p.read_text(); policy=json.loads(original); policy["alpha_search"]=True;p.write_text(json.dumps(policy))
            with self.assertRaises(ValueError):reader.resolve(base,experiment,p,arm)
            p.write_text(original)
            runtime=json.loads((run/"config.json").read_text());runtime["n_test"]=724;(run/"config.json").write_text(json.dumps(runtime))
            with patch.object(reader,"terminal_state",return_value=dict(job="800_0",state="COMPLETED",exit_code="0:0",elapsed="1:00")):
                with self.assertRaises(ValueError):reader.resolve(base,experiment,p,arm)

    def test_runtime_override_of_submitted_optimizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base,experiment,p,arm,run=self.fixture(Path(temp))
            runtime=json.loads((run/"config.json").read_text());runtime["lr"]=1e-2
            (run/"config.json").write_text(json.dumps(runtime))
            with patch.object(reader,"terminal_state",return_value=dict(job="800_0",state="COMPLETED",exit_code="0:0",elapsed="1:00")):
                with self.assertRaisesRegex(ValueError,"training configuration mismatch"):
                    reader.resolve(base,experiment,p,arm)

    def test_spec_mutation_during_analysis_cannot_receive_completion_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            base,experiment,p,arm,run=self.fixture(Path(temp));output=experiment/"readout"
            def mutate_spec(*args,**kwargs):
                output.mkdir();output.with_name(output.name+".resolved.json").write_text("changed")
            argv=["read_completed_union_arm.py","--base",str(base),"--experiment",str(experiment),
                  "--policy",str(p),"--arm",arm,"--output",str(output)]
            receipt=dict(job="800_0",state="COMPLETED",exit_code="0:0",elapsed="1:00")
            with patch.object(reader.sys,"argv",argv),patch.object(reader,"terminal_state",return_value=receipt),\
                    patch.object(reader.subprocess,"run",side_effect=mutate_spec):
                with self.assertRaisesRegex(ValueError,"Frozen readout input changed"):reader.main()
            self.assertFalse((output/"completion_audit.json").exists())

    def test_cohort_mutation_during_analysis_cannot_receive_completion_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            base,experiment,p,arm,run=self.fixture(Path(temp));output=experiment/"readout"
            def mutate_cohort(*args,**kwargs):
                output.mkdir();(base/"data/sequence_groups.tsv").write_text("changed")
            argv=["read_completed_union_arm.py","--base",str(base),"--experiment",str(experiment),
                  "--policy",str(p),"--arm",arm,"--output",str(output)]
            receipt=dict(job="800_0",state="COMPLETED",exit_code="0:0",elapsed="1:00")
            with patch.object(reader.sys,"argv",argv),patch.object(reader,"terminal_state",return_value=receipt),\
                    patch.object(reader.subprocess,"run",side_effect=mutate_cohort):
                with self.assertRaisesRegex(ValueError,"Frozen readout input changed"):reader.main()
            self.assertFalse((output/"completion_audit.json").exists())


if __name__ == "__main__": unittest.main()
