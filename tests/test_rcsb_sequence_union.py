import argparse
import copy
import csv
import gzip
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preparations/m18_union/build_rcsb_sequence_union.py"
spec = importlib.util.spec_from_file_location("rcsb_union", SCRIPT)
union = importlib.util.module_from_spec(spec)
spec.loader.exec_module(union)


def uid(sequence):
    return "useq_" + hashlib.sha256(sequence.encode()).hexdigest()


def entry(pdb="1abc", position=1):
    a, b = "A" * 50, "C" * 3
    sources = {}
    for asym, sequence in (("A", a), ("B", b)):
        mapping = [dict(index=i, label_seq_id=i+1, auth_seq_id=i+10,
                        insertion_code="A" if asym=="A" and i==position else "",
                        resname="ALA" if asym=="A" else "CYS", auth_asym_id=asym)
                   for i in range(len(sequence))]
        sources[asym] = dict(sequence=sequence, seq_sha256=uid(sequence)[5:], residue_map=mapping)
    chains = [dict(chain_instance_id=f"{pdb}:asm1:{x}", assembly_id="1", source_label_asym_id=y)
              for x, y in (("A1", "A"), ("A2", "A"), ("B", "B"))]
    def pair(left,right,i,j):
        return dict(left=f"{pdb}:asm1:{left}",right=f"{pdb}:asm1:{right}",
                    left_positive=[i],right_positive=[j],residue_contacts=[[i,j]])
    value=dict(contract=union.CONTRACT,pdb_id=pdb,model_policy="first_model",coordinate_models=1,
               sources=sources,chains=chains,pairs=[pair("A1","B",position,0),pair("A2","B",2,1),pair("A1","A2",4,5)])
    expected={x["chain_instance_id"]:(uid(a if x["source_label_asym_id"]=="A" else b),"1",x["source_label_asym_id"]) for x in chains}
    return value,expected,{uid(a):50,uid(b):3},{uid(a)}


class RCSBUnionTests(unittest.TestCase):
    def test_short_partners_and_same_sequence_roles_union_including_inserted_residue(self):
        value,expected,lengths,native=entry()
        result=union.summarize_entry(value,expected,lengths,native)
        self.assertEqual(result["updates"][uid("A"*50)],[1,2,4,5])
        self.assertEqual(result["short_partner_sides"],2)
        self.assertEqual(result["outside_native_partner_sides"],2)
        self.assertEqual(len(result["contexts"]),2)
        self.assertEqual(len({x[5] for x in result["contexts"]}),2)
        self.assertEqual(result["source_maps"][0][4],1)

    def test_residue_indices_and_original_pair_masks_must_agree(self):
        for change in ("out_of_range","wrong_positive","duplicate_contact","missing_context","cross_assembly","wrong_hash","wrong_index"):
            value,expected,lengths,native=entry()
            if change=="out_of_range":value["pairs"][0]["residue_contacts"]=[[50,0]]
            elif change=="wrong_positive":value["pairs"][0]["left_positive"]=[10]
            elif change=="duplicate_contact":value["pairs"][0]["residue_contacts"]*=2
            elif change=="missing_context":expected.pop(next(iter(expected)))
            elif change=="cross_assembly":value["chains"][2]["assembly_id"]="2";k=value["chains"][2]["chain_instance_id"];expected[k]=(expected[k][0],"2","B")
            elif change=="wrong_hash":value["sources"]["A"]["seq_sha256"]="0"*64
            else:value["sources"]["A"]["residue_map"][1]["index"]=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                union.summarize_entry(value,expected,lengths,native)

    def fixture(self,base):
        raw=base/"raw.tsv";native=base/"native.tsv";mapping=base/"map.tsv";tags=base/"tags";tags.mkdir()
        def table(path,fields,rows):
            with path.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields,delimiter="\t");writer.writeheader();writer.writerows(rows)
        seqs=["A"*50,"C"*3]
        table(raw,["unique_id","seq_sha256","n_residues","sequence"],
              [dict(unique_id=uid(s),seq_sha256=uid(s)[5:],n_residues=len(s),sequence=s) for s in seqs])
        table(native,["unique_id","seq_sha256","length","split","component_id"],
              [dict(unique_id=uid(seqs[0]),seq_sha256=uid(seqs[0])[5:],length=50,split="train",component_id="component1")])
        rows=[]
        for pdb,position in (("1abc",1),("2abc",7)):
            value,expected,_,_=entry(pdb,position);path=tags/pdb[1:3]/f"{pdb}.json.gz";path.parent.mkdir(exist_ok=True)
            with gzip.open(path,"wt") as handle:json.dump(value,handle)
            for cid,(identifier,assembly,asym) in expected.items():
                rows.append(dict(pdb_id=pdb,chain_instance_id=cid,assembly_id=assembly,source_label_asym_id=asym,
                                 unique_id=identifier,seq_sha256=identifier[5:],tag_path=str(path)))
        table(mapping,["pdb_id","chain_instance_id","assembly_id","source_label_asym_id","unique_id","seq_sha256","tag_path"],rows)
        return argparse.Namespace(raw_sequences=raw,native_split=native,chain_map=mapping,tags_root=tags,
                                  output=base/"out",expected_sequences=1,workers=1,limit_entries=0)

    def test_full_output_exact_coverage_hashes_and_inverse_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));result=union.build(args)
            self.assertEqual((result["status"],result["queries"],result["residues"]),("complete",1,50))
            self.assertEqual(result["short_partner_directional_contributions"],4)
            y=np.load(args.output/"labels.npy",allow_pickle=False)
            self.assertEqual(np.flatnonzero(y).tolist(),[1,2,4,5,7])
            with (args.output/"queries.tsv").open() as f:rows=list(csv.DictReader(f,delimiter="\t"))
            self.assertEqual(rows[0]["contact_contexts"],"4")
            self.assertEqual(rows[0]["pdb_contexts"],"2")
            self.assertEqual(rows[0]["union_label_sha256"],hashlib.sha256(y.tobytes()).hexdigest())
            for name,digest in result["files_sha256"].items():self.assertEqual(union.sha(args.output/name),digest)
            self.assertFalse(result["partner_length_or_feature_filter"])
            self.assertFalse(result["normalization_fitted"])
            with self.assertRaises(FileExistsError):union.build(args)

    def test_smoke_never_publishes_formal_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));args.limit_entries=1
            result=union.build(args)
            self.assertEqual(result["status"],"smoke_complete")
            self.assertTrue((args.output/"smoke_summary.json").exists())
            self.assertFalse((args.output/"summary.json").exists())

    def test_snapshot_mismatch_fails_without_ready_summary_and_names_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));p=args.tags_root/"ab/1abc.json.gz"
            with gzip.open(p,"rt") as handle:value=json.load(handle)
            value["pairs"][0]["left_positive"]=[40]
            with gzip.open(p,"wt") as handle:json.dump(value,handle)
            with self.assertRaisesRegex(ValueError,"1abc"):union.build(args)
            self.assertFalse((args.output/"summary.json").exists())
            self.assertFalse(json.loads((args.output/"failure.json").read_text())["formal_ready"])

    def test_duplicate_map_or_filtered_native_universe_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            with args.chain_map.open() as f:lines=f.readlines()
            with args.chain_map.open("a") as f:f.write(lines[1])
            with self.assertRaises(ValueError):union.read_inputs(args.raw_sequences,args.native_split,args.chain_map,1)
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            with self.assertRaises(ValueError):union.read_inputs(args.raw_sequences,args.native_split,args.chain_map,2)

    def test_parallel_cli_matches_serial_label_and_provenance_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));union.build(args)
            command=[sys.executable,str(SCRIPT),"--raw-sequences",str(args.raw_sequences),"--native-split",str(args.native_split),
                     "--chain-map",str(args.chain_map),"--tags-root",str(args.tags_root),"--output",str(Path(tmp)/"parallel"),
                     "--expected-sequences","1","--workers","2"]
            env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
            subprocess.run(command,env=env,check=True,stdout=subprocess.PIPE,text=True)
            for name in ("labels.npy","queries.tsv","contexts.tsv","source_maps.tsv","entry_audit.tsv"):
                self.assertEqual(union.sha(args.output/name),union.sha(Path(tmp)/"parallel"/name),name)

    def test_input_symlink_is_pinned_before_hashing_and_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);args=self.fixture(base)
            original=args.native_split
            alternate=base/"alternate.tsv"
            alternate.write_text(original.read_text().replace("\ttrain\t","\tval\t"))
            link=base/"native_link.tsv";link.symlink_to(original)
            args.native_split=link
            original_sha=union.sha
            changed=False
            def sha_and_retarget(path):
                nonlocal changed
                value=original_sha(path)
                if Path(path)==original and not changed:
                    link.unlink();link.symlink_to(alternate);changed=True
                return value
            with mock.patch.object(union,"sha",side_effect=sha_and_retarget):
                result=union.build(args)
            self.assertTrue(changed)
            self.assertIn(str(original.resolve()),result["inputs_sha256"])
            with (args.output/"queries.tsv").open() as f:row=next(csv.DictReader(f,delimiter="\t"))
            self.assertEqual(row["native_split"],"train")

    def test_final_logging_failure_does_not_invalidate_published_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            with mock.patch("builtins.print",side_effect=BrokenPipeError):
                result=union.build(args)
            self.assertEqual(result["status"],"complete")
            self.assertEqual(json.loads((args.output/"summary.json").read_text()),result)
            self.assertFalse((args.output/"failure.json").exists())
            self.assertFalse((args.output/"summary.json.tmp").exists())

    def test_parallel_submission_window_is_bounded_and_ordered(self):
        class Future:
            def __init__(self,pool,value):self.pool,self.value=pool,value
            def result(self):self.pool.pending-=1;return self.value
        class Pool:
            pending=0;peak=0
            def submit(self,fn,job):
                self.pending+=1;self.peak=max(self.peak,self.pending)
                return Future(self,job)
        pool=Pool()
        self.assertEqual(list(union.bounded_results(pool,range(100),4)),list(range(100)))
        self.assertEqual(pool.peak,4)
        self.assertEqual(pool.pending,0)


if __name__=="__main__":unittest.main()
