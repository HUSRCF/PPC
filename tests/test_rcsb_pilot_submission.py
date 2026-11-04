import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/"scripts/preparations/m18_union"
sys.path.insert(0,str(PREP));sys.path.insert(0,str(ROOT/"tests"))
import test_rcsb_sequence_inputs as fixture_data
import pack_rcsb_sequence_inputs as packer
spec=importlib.util.spec_from_file_location("rcsb_submit",PREP/"submit_rcsb_union_pilot.py")
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class RCSBSubmissionTests(unittest.TestCase):
    def fixture(self,base):
        args,rows=fixture_data.RCSBInputTests().fixture(base);packer.pack(args)
        code=base/"code";code.mkdir()
        for rel,source in [("scripts/preparations/m18_union/submit_rcsb_union_pilot.py",PREP/"submit_rcsb_union_pilot.py"),
                           ("scripts/preparations/m18_union/pack_rcsb_sequence_inputs.py",PREP/"pack_rcsb_sequence_inputs.py"),
                           ("scripts/training/train_esm_site.py",ROOT/"scripts/training/train_esm_site.py")]:
            p=code/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes())
        cfg=json.loads((ROOT/"configs/experiments/20261105_rcsb_single_union/R0_esmc_only_pilot_seed49.json").read_text())
        for key,name in dict(esm_root="metadata",primary_embedding_root="embeddings",label_root="labels",manifest="manifest.csv",split_dir="split").items():
            cfg["data"][key]=str(args.output/name)
        cfg["training"]["output_dir"]=str(base/"model_run")
        cfg["metadata"].update(expected_train_queries=1,expected_val_queries=1)
        config=base/"config.json";config.write_text(json.dumps(cfg))
        smoke=base/"smoke.json";smoke.write_text(json.dumps(dict(job="smoke123",test_queries=0,optimizer_steps=1,AMP_skips=0,val_skips=0,eval_gradient_isolation=True)))
        smoke_launch=base/"smoke_launch.json";smoke_launch.write_text(json.dumps(dict(job="smoke123",parent_config_sha256=packer.sha(config))))
        prod=base/"packer_launch.json";prod.write_text(json.dumps(dict(host=socket.gethostname(),pid=999999)))
        plan=dict(code_root=str(code),config=str(config),config_sha256=packer.sha(config),input_root=str(args.output),
                  union_root=str(args.union),union_summary_sha256=packer.sha(args.union/"summary.json"),
                  packer_source_sha256=packer.sha(PREP/"pack_rcsb_sequence_inputs.py"),
                  smoke_acceptance=str(smoke),smoke_acceptance_sha256=packer.sha(smoke),
                  smoke_launch=str(smoke_launch),smoke_launch_sha256=packer.sha(smoke_launch),
                  expected_split_counts=dict(train=1,val=1,test=1),expected_val_residues=54,
                  split_choice="native_RCSB",PDBTM_test_independent_after_RCSB_training=False,
                  partition="acd_u",time_limit="12:00:00",state_root=str(base/"state"),model_run=str(base/"model_run"),
                  identity_host=socket.gethostname(),
                  packer_launch=str(prod),packer_launch_sha256=packer.sha(prod),python=sys.executable,
                  source_files_sha256={str(p.relative_to(code)):packer.sha(p) for p in code.rglob("*.py")})
        path=base/"plan.json";path.write_text(json.dumps(plan))
        return plan,path,packer.sha(path)

    def test_verified_full_inputs_submit_once_without_actual_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,digest=self.fixture(Path(tmp))
            with mock.patch.object(launcher.subprocess,"check_output",return_value="12345\n") as submit:
                launcher.submit(path,digest,0)
            self.assertEqual(submit.call_count,1)
            cmd=submit.call_args.args[0]
            self.assertIn("--partition=acd_u",cmd);self.assertIn("--gres=gpu:1",cmd)
            self.assertFalse(json.loads((Path(plan["state_root"])/"submitted.json").read_text())["test_enabled"])
            with self.assertRaisesRegex(ValueError,"already attempted"):launcher.submit(path,digest,0)

    def test_missing_formal_ready_never_submits_and_timeout_does_not_cancel_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,digest=self.fixture(Path(tmp));(Path(plan["input_root"])/"ready.json").unlink()
            cmd=("python pack_rcsb_sequence_inputs.py --output "+plan["input_root"]).encode()
            with mock.patch.object(Path,"read_bytes",return_value=cmd), mock.patch.object(launcher.time,"monotonic",side_effect=[0,1]), \
                    mock.patch.object(launcher.subprocess,"check_output") as scheduler:
                with self.assertRaises(TimeoutError):launcher.await_inputs(plan,0)
                scheduler.assert_not_called()
            self.assertFalse((Path(plan["input_root"])/"failure.json").exists())

    def test_no_test_scratch_and_full_cohort_guard(self):
        for change in ("test","warmstart","limit","pose","split"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                plan,path,digest=self.fixture(Path(tmp));config=Path(plan["config"]);cfg=json.loads(config.read_text())
                if change=="test":cfg["training"]["eval_test_each_epoch"]=True
                elif change=="warmstart":cfg["training"]["init_checkpoint"]="best.pt"
                elif change=="limit":cfg["training"]["max_train_samples"]=1
                elif change=="pose":cfg["model"]["geometry_mode"]="edge_attention"
                else:plan["split_choice"]="protect_PDBTM"
                config.write_text(json.dumps(cfg));plan["config_sha256"]=packer.sha(config)
                with self.assertRaises(ValueError):launcher.config_contract(plan)

    def test_original_native_assignments_cannot_be_replaced_by_consistent_packed_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,digest=self.fixture(Path(tmp));root=Path(plan["input_root"])
            train=(root/"split/train_ids.txt").read_text();val=(root/"split/val_ids.txt").read_text()
            (root/"split/train_ids.txt").write_text(val);(root/"split/val_ids.txt").write_text(train)
            text=(root/"manifest.csv").read_text().replace(",train,",",TEMP,").replace(",val,",",train,").replace(",TEMP,",",val,")
            (root/"manifest.csv").write_text(text)
            ready=json.loads((root/"ready.json").read_text())
            for rel in ready["files_sha256"]:ready["files_sha256"][rel]=packer.sha(root/rel)
            (root/"ready.json").write_text(json.dumps(ready))
            with self.assertRaisesRegex(ValueError,"original native"):launcher.verify_inputs(plan)

    def test_changed_payload_or_frozen_plan_stops_before_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,digest=self.fixture(Path(tmp));path.write_text(path.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"launch plan"):launcher.load_plan(path,digest)
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,digest=self.fixture(Path(tmp));root=Path(plan["input_root"])
            p=next((root/"labels").glob("*.pt"));p.write_bytes(b"corrupt")
            with mock.patch.object(launcher.subprocess,"check_output") as scheduler:
                with self.assertRaisesRegex(ValueError,"payload changed"):launcher.submit(path,digest,0)
                scheduler.assert_not_called()

    def test_cross_node_device_number_changes_are_not_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,digest=self.fixture(Path(tmp))
            original=launcher.file_identity
            def another_client(stat):
                result=original(stat);result[0]+=1000;return result
            with mock.patch.object(launcher.socket,"gethostname",return_value="other-GPU-node"), \
                    mock.patch.object(launcher,"file_identity",side_effect=another_client):
                launcher.verify_inputs(plan)
            with mock.patch.object(launcher,"file_identity",side_effect=another_client):
                with self.assertRaisesRegex(ValueError,"reference changed"):launcher.verify_inputs(plan)
            def changed_inode(stat):
                result=original(stat);result[0]+=1000;result[1]+=1;return result
            with mock.patch.object(launcher.socket,"gethostname",return_value="other-GPU-node"), \
                    mock.patch.object(launcher,"file_identity",side_effect=changed_inode):
                with self.assertRaisesRegex(ValueError,"reference changed"):launcher.verify_inputs(plan)


if __name__=="__main__":unittest.main()
