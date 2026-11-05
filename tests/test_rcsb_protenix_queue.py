import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import gemmi
import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/preparations/m18_union/run_rcsb_protenix_queue.py"
spec=importlib.util.spec_from_file_location("rcsb_fold_queue",SCRIPT)
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class RCSBProtenixQueueTests(unittest.TestCase):
    def test_error_index_preserves_nested_output_and_ESM_matches_without_rescanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);output=work/"outputs";esm=work/"esm_embeddings/esm_error"
            (output/"a/nested").mkdir(parents=True);(output/"b").mkdir();esm.mkdir(parents=True)
            for path in (output/"a/nested/uid.txt",output/"b/uid.txt",output/"b/uid_extra.txt",
                         esm/"uid_1.txt",esm/"uid_2.txt",esm/"uid_other_1.txt",esm/"uid2_1.txt"):
                path.write_text("diagnostic")
            expected=list(output.rglob("uid.txt"))+list(esm.glob("uid_*.txt"))
            index=queue.index_error_files(work)
            with mock.patch.object(Path,"rglob",side_effect=AssertionError("Repeated scan")), \
                    mock.patch.object(Path,"glob",side_effect=AssertionError("Repeated scan")):
                self.assertEqual(queue.query_error_files(index,"uid"),expected)
                self.assertEqual(queue.query_error_files(index,"absent"),[])
            self.assertEqual(set(index[2]),set(output.rglob("*.txt"))|set(esm.glob("*.txt")))

    def test_pinned_upgrade_retains_contract_and_retry_ledger_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=b'{"status":"failed","attempt":2}\n'
            (root/"progress.jsonl").write_bytes(ledger)
            previous=dict(source_code_sha256=queue.ERROR_INDEX_PARENT_SHA256,model="mini",max_attempts=2)
            raw=json.dumps(previous,indent=2).encode();(root/"contract.json").write_bytes(raw)
            current=dict(previous,source_code_sha256="new-performance-code")
            with self.assertRaisesRegex(ValueError,"Changed queue contract"):
                queue.ensure_contract(root,current)
            self.assertFalse((root/"error_index_upgrade.json").exists())
            queue.ensure_contract(root,current,True)
            self.assertEqual((root/"progress.jsonl").read_bytes(),ledger)
            self.assertEqual((root/"contract.before_error_index_upgrade.json").read_bytes(),raw)
            self.assertEqual(json.loads((root/"contract.json").read_text()),current)
            receipt=(root/"error_index_upgrade.json").read_bytes()
            queue.ensure_contract(root,current,True)
            self.assertEqual((root/"error_index_upgrade.json").read_bytes(),receipt)

    def test_upgrade_cannot_change_other_options_or_accept_unrecognized_parent(self):
        for old_hash,attempts in ((queue.ERROR_INDEX_PARENT_SHA256,3),("unreviewed",2)):
            with self.subTest(old_hash=old_hash,attempts=attempts),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);previous=dict(source_code_sha256=old_hash,max_attempts=2)
                raw=json.dumps(previous).encode();(root/"contract.json").write_bytes(raw)
                with self.assertRaisesRegex(ValueError,"Changed queue contract"):
                    queue.ensure_contract(root,dict(source_code_sha256="new-code",max_attempts=attempts),True)
                self.assertEqual((root/"contract.json").read_bytes(),raw)
                self.assertFalse((root/"error_index_upgrade.json").exists())

    def test_upgrade_recovers_after_interrupted_atomic_metadata_writes(self):
        for target_name in ("contract.before_error_index_upgrade.json","error_index_upgrade.json","contract.json"):
            with self.subTest(target=target_name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);previous=dict(source_code_sha256=queue.ERROR_INDEX_PARENT_SHA256,max_attempts=2)
                raw=json.dumps(previous).encode();(root/"contract.json").write_bytes(raw)
                (root/"progress.jsonl").write_text('{"status":"failed","attempt":2}\n')
                current=dict(previous,source_code_sha256="new-performance-code")
                real_write=queue.write_atomic
                def interrupted(path,data):
                    if path.name==target_name:
                        path.with_name(path.name+".tmp").write_bytes(data[:5])
                        raise RuntimeError("interrupted metadata write")
                    real_write(path,data)
                with mock.patch.object(queue,"write_atomic",side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,"interrupted"):
                        queue.ensure_contract(root,current,True)
                self.assertEqual((root/"contract.json").read_bytes(),raw)
                queue.ensure_contract(root,current,True)
                self.assertEqual(json.loads((root/"contract.json").read_text()),current)
                self.assertEqual(json.loads((root/"error_index_upgrade.json").read_text())["current_source_sha256"],current["source_code_sha256"])

    def test_prepared_upgrade_can_recover_after_old_runner_advances_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);previous=dict(source_code_sha256=queue.ERROR_INDEX_PARENT_SHA256,max_attempts=2)
            raw=json.dumps(previous).encode();(root/"contract.json").write_bytes(raw)
            ledger=root/"progress.jsonl";ledger.write_text('{"status":"failed","attempt":2}\n')
            current=dict(previous,source_code_sha256="new-performance-code");real_write=queue.write_atomic
            def before_commit(path,data):
                if path.name=="contract.json":raise RuntimeError("pre-commit crash")
                real_write(path,data)
            with mock.patch.object(queue,"write_atomic",side_effect=before_commit):
                with self.assertRaisesRegex(RuntimeError,"pre-commit"):
                    queue.ensure_contract(root,current,True)
            prepared=(root/"error_index_upgrade.json").read_bytes()
            with ledger.open("a") as f:f.write('{"status":"ok","attempt":1}\n')
            all_events=ledger.read_bytes();queue.ensure_contract(root,current,True)
            self.assertEqual(ledger.read_bytes(),all_events)
            self.assertEqual(json.loads((root/"error_index_upgrade.json").read_text())["ledger_sha256"],hashlib.sha256(all_events).hexdigest())
            archives=list(root.glob("error_index_upgrade.prepared_*.json"))
            self.assertEqual(len(archives),1);self.assertEqual(archives[0].read_bytes(),prepared)

    def manifest(self,base,sequences):
        path=base/"manifest.tsv"
        rows=[dict(sequence=s,n_residues=len(s),seq_sha256=hashlib.sha256(s.encode()).hexdigest(),
                   unique_id="useq_"+hashlib.sha256(s.encode()).hexdigest()) for s in sequences]
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
        return path,rows

    def output(self,work,sequence,atom="CA"):
        uid="useq_"+hashlib.sha256(sequence.encode()).hexdigest()
        directory=work/"outputs"/uid
        out=directory/"seed_101/predictions";out.mkdir(parents=True)
        s=gemmi.Structure();s.add_model(gemmi.Model(1));c=gemmi.Chain("A")
        for i,aa in enumerate(sequence):
            r=gemmi.Residue();r.name={"A":"ALA","C":"CYS","X":"UNK"}[aa];r.seqid=gemmi.SeqId(i+1," ")
            a=gemmi.Atom();a.name=atom;a.element=gemmi.Element("C");a.pos=gemmi.Position(i*3.8,0,0);a.b_iso=80;r.add_atom(a);c.add_residue(r)
        s[0].add_chain(c);s.setup_entities();s.make_mmcif_document().write_file(str(out/(uid+"_sample_0.cif")))
        summary=out/(uid+"_summary_confidence_sample_0.json");summary.write_text(json.dumps(dict(plddt=80)))
        label=uid+"_1";esm=work/"esm_embeddings";cache=esm/"esm2-3b"/label;cache.mkdir(parents=True)
        torch.save(torch.ones(len(sequence),2560),cache/(label+".pt"))
        with (esm/"input_prot_sequences.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["seq","seq_label","part_id"]);w.writeheader();w.writerow(dict(seq=sequence,seq_label=label,part_id=label))
        return directory,dict(unique_id=uid,sequence=sequence),summary

    def test_short_and_unknown_residues_retained_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,expected=self.manifest(Path(tmp),["A"*50,"C"*99,"AX"*30])
            rows,digest=queue.read_manifest(p)
            self.assertEqual(len(rows),3);self.assertEqual({r["sequence"] for r in rows},{r["sequence"] for r in expected})
            self.assertEqual([r["unique_id"] for r in rows],sorted(r["unique_id"] for r in rows))
            self.assertEqual(digest,hashlib.sha256(p.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError,"no silent filtering"):queue.read_manifest(p,100)

    def test_invalid_or_empty_scope_cannot_succeed_as_zero_query_queue(self):
        for sequences in [["A"*49],["A"*50,"A"*50],["A"*49+"B"]]:
            with self.subTest(sequences=sequences),tempfile.TemporaryDirectory() as tmp:
                p,_=self.manifest(Path(tmp),sequences)
                with self.assertRaises(ValueError):queue.read_manifest(p)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"empty.tsv";p.write_text("unique_id\tsequence\tn_residues\tseq_sha256\n")
            with self.assertRaises(ValueError):queue.read_manifest(p)

    def test_CIF_exact_sequence_CA_and_official_summary_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);directory,row,summary=self.output(base,"AXC")
            self.assertIsNotNone(queue.validate(directory,row))
            self.assertIsNone(queue.validate(directory,dict(row,sequence="AAC")))
            summary.unlink();(summary.parent/"unrelated.json").write_text("{}")
            self.assertIsNone(queue.validate(directory,row))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);directory,row,_=self.output(base,"AAA",atom="C")
            self.assertIsNone(queue.validate(directory,row))

    def test_existing_valid_output_is_reused_without_CLI_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);p,rows=self.manifest(base,["A"*50]);reuse=base/"reuse";reuse.mkdir()
            directory,_,_=self.output(base/"work","A"*50)
            (reuse/rows[0]["unique_id"]).symlink_to(directory)
            run=base/"run"
            result=subprocess.run([sys.executable,str(SCRIPT),"--manifest",str(p),"--root",str(run),"--project",str(base),
                                   "--protenix-cli","/does/not/exist","--reuse",str(reuse)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((run/"summary.json").read_text())["counts"],dict(ok=1))
            contract=json.loads((run/"contract.json").read_text())
            self.assertEqual(contract["min_length"],50);self.assertFalse(contract["msa"])
            self.assertFalse((run/"attempts").exists())

    def test_missing_or_zero_ESM_and_recorded_errors_cannot_count_as_coverage(self):
        for change in ("missing","zeros","nonfinite","error","wrong_query"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);directory,row,_=self.output(base,"AXC");label=row["unique_id"]+"_1"
                p=base/"esm_embeddings/esm2-3b"/label/(label+".pt")
                if change=="missing":p.unlink()
                elif change=="zeros":torch.save(torch.zeros(3,2560),p)
                elif change=="nonfinite":torch.save(torch.full((3,2560),float("nan")),p)
                elif change=="error":
                    error=base/"esm_embeddings/esm_error";error.mkdir();(error/(label+".txt")).write_text("SDK used a zero fallback")
                else:
                    f=base/"esm_embeddings/input_prot_sequences.csv";f.write_text(f.read_text().replace("AXC","AAC"))
                self.assertIsNone(queue.validate(directory,row))

    def test_QC_runtime_failure_stops_before_mutation_or_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);p,_=self.manifest(base,["A"*50]);root=base/"run"
            argv=[str(SCRIPT),"--manifest",str(p),"--root",str(root),"--project",str(base)]
            with mock.patch.object(sys,"argv",argv),mock.patch.object(queue,"preflight_qc_runtime",side_effect=ImportError("broken Torch ABI")), \
                    mock.patch.object(queue.subprocess,"Popen") as launch:
                with self.assertRaises(ImportError):queue.main()
                launch.assert_not_called();self.assertFalse(root.exists())
                with self.assertRaises(ImportError):queue.validate(base,dict(sequence="A",unique_id="irrelevant"))

    def test_ESM_error_diagnostics_keep_bounded_query_retries_not_infrastructure_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);p,rows=self.manifest(base,["A"*50,"C"*50,"A"*51,"C"*51])
            cli=base/"fake_cli"
            cli.write_text("#!"+sys.executable+"\nimport sys,json\nfrom pathlib import Path\n"
                           "items=json.loads(Path(sys.argv[sys.argv.index('--input')+1]).read_text())\n"
                           "d=Path('esm_embeddings/esm_error');d.mkdir(parents=True)\n"
                           "for r in items:(d/(r['name']+'_1.txt')).write_text('ESM input error')\nraise SystemExit(1)\n")
            cli.chmod(0o755);run=base/"run"
            result=subprocess.run([sys.executable,str(SCRIPT),"--manifest",str(p),"--root",str(run),"--project",str(base),
                                   "--protenix-cli",str(cli),"--batch-size","1","--max-attempts","2"],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertEqual(json.loads((run/"summary.json").read_text())["counts"],dict(failed=4))
            events=[json.loads(x) for x in (run/"progress.jsonl").read_text().splitlines()]
            failed=[x for x in events if x["status"]=="failed"]
            self.assertEqual(len(failed),8);self.assertTrue(all(x["error_files"] for x in failed))

    def fake_cli(self,base,sleep=0):
        cli=base/"project/.venv/bin/protenix";cli.parent.mkdir(parents=True)
        cli.write_text("#!"+sys.executable+"\nimport os,time\nfrom pathlib import Path\n"
                       "Path(os.environ['FAKE_CALLED']).write_text(str(os.getpid()))\n"
                       "time.sleep("+str(sleep)+")\nraise SystemExit(1)\n")
        cli.chmod(0o755)
        return cli

    def test_relative_project_and_CLI_paths_survive_attempt_workdir(self):
        for explicit in (False,True):
            with self.subTest(explicit=explicit),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);p,_=self.manifest(base,["A"*50]);cli=self.fake_cli(base)
                env={**os.environ,"FAKE_CALLED":str(base/"called")}
                cmd=[sys.executable,str(SCRIPT),"--manifest",p.name,"--root","run","--project","project","--max-attempts","1"]
                if explicit:cmd += ["--protenix-cli",str(cli.relative_to(base))]
                result=subprocess.run(cmd,cwd=base,env=env,capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertTrue((base/"called").exists())

    def test_orphan_child_keeps_queue_lock_until_it_exits(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);p,_=self.manifest(base,["A"*50]);cli=self.fake_cli(base,sleep=30)
            env={**os.environ,"FAKE_CALLED":str(base/"called")}
            cmd=[sys.executable,str(SCRIPT),"--manifest",str(p),"--root",str(base/"run"),"--project",str(base/"project"),"--protenix-cli",str(cli)]
            parent=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            child=None
            try:
                deadline=time.monotonic()+5
                while not (base/"called").exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue((base/"called").exists());child=int((base/"called").read_text())
                parent.kill();parent.communicate(timeout=5)
                retry=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=5)
                self.assertNotEqual(retry.returncode,0)
                self.assertIn("BlockingIOError",retry.stderr)
                self.assertEqual(len(list((base/"run/attempts").iterdir())),1)
            finally:
                if parent.poll() is None:parent.kill();parent.communicate(timeout=5)
                if child is not None:
                    try:os.killpg(child,signal.SIGTERM)
                    except ProcessLookupError:pass
                    deadline=time.monotonic()+5
                    with (base/"run/queue.lock").open("a") as lock:
                        while True:
                            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                            except BlockingIOError:
                                if time.monotonic()>=deadline:raise
                                time.sleep(.02)


if __name__=="__main__":unittest.main()
