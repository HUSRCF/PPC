import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/preparations/m18_union"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("full_esm", SCRIPTS / "protenix_full_esm.py")
full = importlib.util.module_from_spec(spec); spec.loader.exec_module(full)


class FullESMTests(unittest.TestCase):
    def input(self, base, sequence):
        uid="useq_"+hashlib.sha256(sequence.encode()).hexdigest()
        path=base/"input.json";path.write_text(json.dumps([dict(name=uid,sequences=[dict(proteinChain=dict(sequence=sequence,count=1,id=["A"]))])]))
        argv=["pred"]
        for key,value in dict(input=str(path),out_dir=str(base/"out"),model_name="protenix_mini_esm_v0.5.0",
                              use_default_params="true",use_msa="false",sample="1",seeds="101",dtype="bf16",
                              trimul_kernel="torch",triatt_kernel="torch",enable_fusion="false").items():argv += ["--"+key,value]
        return uid,path,argv

    def ready(self,root,uid,sequence):
        p=full.cache_path(root,uid);p.parent.mkdir(parents=True);torch.save(torch.ones(len(sequence),2560),p)
        (root/"contract.json").write_text('{}');contract_sha=full.digest(root/"contract.json")
        record=dict(unique_id=uid,sequence_sha256=uid[5:],contract_sha256=contract_sha,**full.check_tensor(p,len(sequence),torch))
        (p.parent/"cache_receipt.json").write_text(json.dumps(record))
        value=dict(status="complete",full_query=True,truncation=False,source_sha256=full.digest(full.__file__),
                   source_hashes=full.SOURCE_HASHES,queue_source_sha256=full.QUEUE_SOURCE_SHA256,
                   cache_contract_sha256=contract_sha,records={uid:record})
        (root/"ready.json").write_text(json.dumps(value));return p

    def test_long_query_is_retained_and_parameter_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);uid,_,argv=self.input(base,"A"*4579)
            _,rows,_=full.input_rows(argv);self.assertEqual(len(rows[uid]),4579)
            argv[argv.index("--sample")+1]="5"
            with self.assertRaisesRegex(ValueError,"Changed historical"):full.input_rows(argv)
        with tempfile.TemporaryDirectory() as tmp:
            _,_,argv=self.input(Path(tmp),"A"*4580)
            with self.assertRaisesRegex(ValueError,"Wrong query"):full.input_rows(argv)

    def test_tensor_requires_complete_finite_nonzero_residue_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.pt"
            for value in (torch.ones(49,2560),torch.zeros(50,2560),torch.full((50,2560),float("nan")),torch.ones(50,20),torch.ones(50,2560,dtype=torch.int64)):
                torch.save(value,p)
                with self.assertRaises(ValueError):full.check_tensor(p,50,torch)

    def test_existing_full_cache_is_copied_and_stock_CLI_runs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);uid,path,argv=self.input(base,"A"*50);bank=base/"bank";bank.mkdir();source=self.ready(bank,uid,"A"*50)
            work=base/"work";work.mkdir();prior=os.getcwd();called=[]
            def stock_cli(**kw):
                called.append(kw);args=kw["args"];Path(args[args.index("--out_dir")+1]).mkdir()
            cli=types.SimpleNamespace(main=stock_cli)
            env=dict(PPC_FULL_ESM_ROOT=str(bank),PPC_FULL_ESM_READY_SHA256=full.digest(bank/"ready.json"))
            try:
                os.chdir(work)
                with mock.patch.dict(os.environ,env),mock.patch.object(full,"runtime",return_value=torch), \
                        mock.patch.dict(sys.modules,{"runner.batch_inference":types.SimpleNamespace(protenix_cli=cli)}):
                    full.fold(argv)
                destination=full.cache_path(work/"esm_embeddings",uid)
                self.assertEqual(source.read_bytes(),destination.read_bytes());self.assertFalse(destination.is_symlink())
                expected=list(argv);expected[expected.index("--out_dir")+1]=str(base/"out.full_ESM_unaccepted")
                self.assertEqual(called,[dict(args=expected,standalone_mode=False)])
                self.assertTrue((base/"out").is_dir());self.assertFalse((base/"out.full_ESM_unaccepted").exists())
            finally:os.chdir(prior)

    def test_post_CLI_corruption_cannot_publish_parent_acceptable_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);uid,path,argv=self.input(base,"A"*50);bank=base/"bank";bank.mkdir();self.ready(bank,uid,"A"*50)
            work=base/"work";work.mkdir();prior=os.getcwd()
            def corrupt(**kw):
                args=kw["args"];staged=Path(args[args.index("--out_dir")+1]);staged.mkdir()
                (staged/"apparently_valid.cif").write_text("A result must remain unpublished")
                torch.save(torch.ones(50,2560)*2,full.cache_path(work/"esm_embeddings",uid))
            env=dict(PPC_FULL_ESM_ROOT=str(bank),PPC_FULL_ESM_READY_SHA256=full.digest(bank/"ready.json"))
            try:
                os.chdir(work)
                with mock.patch.dict(os.environ,env),mock.patch.object(full,"runtime",return_value=torch), \
                        mock.patch.dict(sys.modules,{"runner.batch_inference":types.SimpleNamespace(protenix_cli=types.SimpleNamespace(main=corrupt))}):
                    with self.assertRaisesRegex(ValueError,"changed during stock"):full.fold(argv)
                self.assertFalse((base/"out").exists());self.assertTrue((base/"out.full_ESM_unaccepted").is_dir())
                self.assertTrue((work/"esm_embeddings/esm_error"/(uid+"_1.txt")).is_file())
            finally:os.chdir(prior)

    def test_changed_cache_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);uid,_,argv=self.input(base,"A"*50);bank=base/"bank";bank.mkdir();p=self.ready(bank,uid,"A"*50)
            torch.save(torch.ones(49,2560),p)
            env=dict(PPC_FULL_ESM_ROOT=str(bank),PPC_FULL_ESM_READY_SHA256=full.digest(bank/"ready.json"))
            with mock.patch.dict(os.environ,env),mock.patch.object(full,"runtime",return_value=torch):
                with self.assertRaisesRegex(ValueError,"Invalid full-query"):full.fold(argv)

    def prepare_fixture(self,base,sequence):
        uid="useq_"+hashlib.sha256(sequence.encode()).hexdigest();manifest=base/"queries.tsv"
        with manifest.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["unique_id","sequence","seq_sha256","n_residues"],delimiter="\t")
            w.writeheader();w.writerow(dict(unique_id=uid,sequence=sequence,seq_sha256=uid[5:],n_residues=len(sequence)))
        checkpoint=base/"runtime/checkpoint";checkpoint.mkdir(parents=True)
        for name in ("esm2_t36_3B_UR50D.pt","esm2_t36_3B_UR50D-contact-regression.pt"):(checkpoint/name).touch()
        return uid,manifest,dict(PROTENIX_ROOT_DIR=str(checkpoint.parent))

    def test_prepare_requests_full_length_and_reuses_only_valid_full_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);sequence="A"*4100;uid,manifest,env=self.prepare_fixture(base,sequence);calls=[]
            def compute(name,model,alphabet,labels,sequences,save_dir,**kw):
                calls.append(kw);self.assertEqual(sequences,[sequence]);self.assertEqual(labels,[uid+"_1"])
                torch.save(torch.ones(len(sequence),2560),Path(save_dir)/(labels[0]+".pt"))
            backend=types.SimpleNamespace(load_esm_model=lambda *a,**k:(object(),object()),compute_ESM_embeddings=compute)
            with mock.patch.dict(os.environ,env),mock.patch.object(full,"runtime",return_value=torch), \
                    mock.patch.dict(sys.modules,{"protenix.data.esm.compute_esm":backend}),mock.patch.object(torch.cuda,"empty_cache"):
                a=full.prepare(manifest,base/"bank");b=full.prepare(manifest,base/"bank")
            self.assertEqual(calls,[dict(toks_per_batch=16384,truncation_seq_length=4100)])
            self.assertEqual(a,b);self.assertEqual(a["records"][uid]["length"],4100)

    def test_prepare_rejects_unproven_preexisting_tensor_even_if_shape_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);uid,manifest,env=self.prepare_fixture(base,"A"*50);bank=base/"bank"
            p=full.cache_path(bank,uid);p.parent.mkdir(parents=True);torch.save(torch.ones(50,2560),p)
            backend=types.SimpleNamespace(load_esm_model=lambda *a,**k:(object(),object()),compute_ESM_embeddings=mock.Mock())
            with mock.patch.dict(os.environ,env),mock.patch.object(full,"runtime",return_value=torch), \
                    mock.patch.dict(sys.modules,{"protenix.data.esm.compute_esm":backend}):
                with self.assertRaises(FileNotFoundError):full.prepare(manifest,bank)
            self.assertFalse((bank/"ready.json").exists());backend.compute_ESM_embeddings.assert_not_called()

    def test_prepare_records_OOM_and_never_publishes_ready_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);uid,manifest,env=self.prepare_fixture(base,"A"*50)
            backend=types.SimpleNamespace(load_esm_model=lambda *a,**k:(object(),object()),
                                          compute_ESM_embeddings=mock.Mock(side_effect=RuntimeError("CUDA out of memory")))
            with mock.patch.dict(os.environ,env),mock.patch.object(full,"runtime",return_value=torch), \
                    mock.patch.dict(sys.modules,{"protenix.data.esm.compute_esm":backend}):
                with self.assertRaisesRegex(RuntimeError,"out of memory"):full.prepare(manifest,base/"bank")
            self.assertFalse((base/"bank/ready.json").exists());failures=list((base/"bank/failures").glob("*.json"))
            self.assertEqual(len(failures),1);self.assertTrue(json.loads(failures[0].read_text())["OOM"])


if __name__ == "__main__":unittest.main()
