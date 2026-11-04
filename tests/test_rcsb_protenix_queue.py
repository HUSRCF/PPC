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

import gemmi

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/preparations/m18_union/run_rcsb_protenix_queue.py"
spec=importlib.util.spec_from_file_location("rcsb_fold_queue",SCRIPT)
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class RCSBProtenixQueueTests(unittest.TestCase):
    def manifest(self,base,sequences):
        path=base/"manifest.tsv"
        rows=[dict(sequence=s,n_residues=len(s),seq_sha256=hashlib.sha256(s.encode()).hexdigest(),
                   unique_id="useq_"+hashlib.sha256(s.encode()).hexdigest()) for s in sequences]
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
        return path,rows

    def output(self,base,sequence,atom="CA"):
        uid="useq_"+hashlib.sha256(sequence.encode()).hexdigest()
        out=base/"seed_101/predictions";out.mkdir(parents=True)
        s=gemmi.Structure();s.add_model(gemmi.Model(1));c=gemmi.Chain("A")
        for i,aa in enumerate(sequence):
            r=gemmi.Residue();r.name={"A":"ALA","C":"CYS","X":"UNK"}[aa];r.seqid=gemmi.SeqId(i+1," ")
            a=gemmi.Atom();a.name=atom;a.element=gemmi.Element("C");a.pos=gemmi.Position(i*3.8,0,0);a.b_iso=80;r.add_atom(a);c.add_residue(r)
        s[0].add_chain(c);s.setup_entities();s.make_mmcif_document().write_file(str(out/(uid+"_sample_0.cif")))
        summary=out/(uid+"_summary_confidence_sample_0.json");summary.write_text(json.dumps(dict(plddt=80)))
        return dict(unique_id=uid,sequence=sequence),summary

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
            base=Path(tmp);row,summary=self.output(base,"AXC")
            self.assertIsNotNone(queue.validate(base,row))
            self.assertIsNone(queue.validate(base,dict(row,sequence="AAC")))
            summary.unlink();(summary.parent/"unrelated.json").write_text("{}")
            self.assertIsNone(queue.validate(base,row))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);row,_=self.output(base,"AAA",atom="C")
            self.assertIsNone(queue.validate(base,row))

    def test_existing_valid_output_is_reused_without_CLI_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);p,rows=self.manifest(base,["A"*50]);reuse=base/"reuse";reuse.mkdir()
            directory=reuse/rows[0]["unique_id"];self.output(directory,"A"*50)
            run=base/"run"
            result=subprocess.run([sys.executable,str(SCRIPT),"--manifest",str(p),"--root",str(run),"--project",str(base),
                                   "--protenix-cli","/does/not/exist","--reuse",str(reuse)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((run/"summary.json").read_text())["counts"],dict(ok=1))
            contract=json.loads((run/"contract.json").read_text())
            self.assertEqual(contract["min_length"],50);self.assertFalse(contract["msa"])
            self.assertFalse((run/"attempts").exists())

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
