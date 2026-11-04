import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"scripts/preparations/m18_union/pack_rcsb_sequence_inputs.py"
spec = importlib.util.spec_from_file_location("pack_rcsb",SCRIPT)
packing = importlib.util.module_from_spec(spec); spec.loader.exec_module(packing)
sys.path.insert(0,str(ROOT/"src"))
from ppcbind.data.esm_site_dataset import ESMProteinSiteDataset, collate_esm_site_features


class RCSBInputTests(unittest.TestCase):
    def fixture(self,base):
        union=base/"union";union.mkdir();bank=base/"bank";bank.mkdir()
        rows=[];labels=[];contexts=[]
        for i,split in enumerate(("train","val","test")):
            seq="ACD"*(17+i);digest=hashlib.sha256(seq.encode()).hexdigest();uid="useq_"+digest
            y=np.zeros(len(seq),dtype=np.uint8);y[i:i+3]=1
            rows.append(dict(unique_id=uid,sequence=seq,seq_sha256=digest,seq_sha1=hashlib.sha1(seq.encode()).hexdigest(),
                             length=len(seq),offset=len(labels),union_positive=3,union_label_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                             native_split=split,native_component="c"+str(i)))
            contexts.append(dict(unique_id=uid,pdb_id=str(i)+"abc"));labels.extend(y)
            p=bank/digest[:2]/(uid+".pt");p.parent.mkdir(exist_ok=True)
            torch.save(dict(unique_id=uid,sequence=seq,sequence_sha256=digest,n_residues=len(seq),
                            resolved_revision=packing.REVISION,repr_layers=[12,24,36],hidden_normalization="final_layernorm",
                            embeddings=torch.randn(len(seq),3456)),p)
        np.save(union/"labels.npy",np.asarray(labels,dtype=np.uint8))
        for name,data in (("queries.tsv",rows),("contexts.tsv",contexts)):
            with (union/name).open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]),delimiter="\t");writer.writeheader();writer.writerows(data)
        for name in ("source_maps.tsv","entry_audit.tsv"):
            (union/name).write_text("pdb_id\n0abc\n1abc\n2abc\n")
        summary=dict(status="complete",contract=packing.CONTRACT,queries=3,native_queries=3,residues=len(labels),
                     partner_length_or_feature_filter=False,final_split_changed=False,
                     files_sha256={p.name:packing.sha(p) for p in union.iterdir()})
        (union/"summary.json").write_text(json.dumps(summary))
        return argparse.Namespace(union=union,esmc_bank=bank,output=base/"out",expected_sequences=3,workers=1,smoke_count=0),rows

    def test_full_packing_existing_loader_collation_and_no_embedding_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,rows=self.fixture(Path(tmp));result=packing.pack(args)
            self.assertEqual(result["split_counts"],dict(train=1,val=1,test=1))
            self.assertTrue(result["independent_full_feature_content_audit"])
            self.assertTrue(result["embedding_bank_immutability_required_through_training"])
            ds=ESMProteinSiteDataset(args.output/"metadata",label_root=args.output/"labels",ids=[r["unique_id"] for r in rows],
                                    primary_embedding_root=args.output/"embeddings",strict_ids=True,require_labels=True,
                                    require_primary_embeddings=True,strict_label_metadata=True)
            batch=collate_esm_site_features([ds[i] for i in range(3)])
            self.assertEqual(tuple(batch["esm_embeddings"].shape),(3,57,3456))
            self.assertEqual(int((batch["labels"]==1).sum()),9)
            for r in rows:self.assertTrue((args.output/"embeddings"/(r["unique_id"]+".pt")).is_symlink())
            self.assertTrue((args.output/"ready.json").exists())
            with self.assertRaises(FileExistsError):packing.pack(args)

    def test_bad_feature_fails_closed_and_smoke_never_publishes_formal_ready(self):
        for change in ("sequence","shape","nonfinite","revision","smoke"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                args,rows=self.fixture(Path(tmp));r=rows[0];p=args.esmc_bank/r["seq_sha256"][:2]/(r["unique_id"]+".pt")
                x=torch.load(p,weights_only=False)
                if change=="sequence":x["sequence"]="X"*len(r["sequence"])
                elif change=="shape":x["embeddings"]=x["embeddings"][:-1]
                elif change=="nonfinite":x["embeddings"][0,0]=float("nan")
                elif change=="revision":x["resolved_revision"]="wrong"
                else:args.smoke_count=2
                torch.save(x,p)
                if change=="smoke":
                    self.assertEqual(packing.pack(args)["queries"],2)
                    self.assertTrue((args.output/"smoke_ready.json").exists())
                else:
                    with self.assertRaises(ValueError):packing.pack(args)
                    self.assertTrue((args.output/"failure.json").exists())
                self.assertFalse((args.output/"ready.json").exists())

    def test_changed_union_or_cross_split_component_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,rows=self.fixture(Path(tmp));(args.union/"queries.tsv").write_text("corrupt")
            with self.assertRaises(ValueError):packing.pack(args)
        with tempfile.TemporaryDirectory() as tmp:
            args,rows=self.fixture(Path(tmp));rows[1]["native_component"]=rows[0]["native_component"]
            with (args.union/"queries.tsv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
            s=json.loads((args.union/"summary.json").read_text());s["files_sha256"]["queries.tsv"]=packing.sha(args.union/"queries.tsv")
            (args.union/"summary.json").write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,"Cross-split"):packing.pack(args)

    def test_parallel_cli_keeps_all_alignment_and_manifest_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,rows=self.fixture(Path(tmp))
            env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
            subprocess.run([sys.executable,str(SCRIPT),"--union",str(args.union),"--esmc-bank",str(args.esmc_bank),
                            "--output",str(args.output),"--expected-sequences","3","--workers","2"],env=env,check=True,stdout=subprocess.PIPE)
            receipt=json.loads((args.output/"ready.json").read_text())
            self.assertEqual(receipt["queries"],3)
            self.assertEqual((args.output/"split/train_ids.txt").read_text().strip(),rows[0]["unique_id"])

    def test_changed_source_after_worker_audit_prevents_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,rows=self.fixture(Path(tmp));original=packing.pack_one;changed=False
            def mutate_after_audit(job):
                nonlocal changed
                result=original(job)
                if not changed:
                    path=Path(result[1]);x=torch.load(path,weights_only=False)
                    x["embeddings"][0,0]=float("nan");torch.save(x,path);changed=True
                return result
            with mock.patch.object(packing,"pack_one",side_effect=mutate_after_audit):
                with self.assertRaisesRegex(ValueError,"source/reference changed"):packing.pack(args)
            self.assertFalse((args.output/"ready.json").exists())
            self.assertTrue((args.output/"failure.json").exists())

    def test_missing_mandatory_fingerprint_cannot_disable_context_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,rows=self.fixture(Path(tmp));p=args.union/"summary.json"
            summary=json.loads(p.read_text());del summary["files_sha256"]["contexts.tsv"];p.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"mandatory"):packing.pack(args)
            self.assertFalse(args.output.exists())


if __name__=="__main__":unittest.main()
