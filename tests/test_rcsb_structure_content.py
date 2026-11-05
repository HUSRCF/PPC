import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gemmi
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"scripts/preparations/m18_union"
sys.path.insert(0,str(SOURCE))
SPEC=importlib.util.spec_from_file_location("structure_audit",SOURCE/"audit_rcsb_structure_content.py")
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


class StructureContentTests(unittest.TestCase):
    def fixture(self, root):
        seq="A"*50;uid="useq_"+hashlib.sha256(seq.encode()).hexdigest()
        directory=root/"snapshot/runs/one";directory.mkdir(parents=True)
        cif=directory/(uid+"_sample_0.cif");confidence=directory/(uid+"_summary_confidence_sample_0.json")
        s=gemmi.Structure();s.add_model(gemmi.Model(1));c=gemmi.Chain("A")
        names=[];coords=[]
        for i in range(50):
            r=gemmi.Residue();r.name="ALA";r.seqid=gemmi.SeqId(i+1," ")
            xyz=[]
            for j,name in enumerate(("N","CA","C","O")):
                a=gemmi.Atom();a.name=name;a.element=gemmi.Element("O"if name=="O"else"N"if name=="N"else"C")
                a.pos=gemmi.Position(i*3.8+j*.2,j*.1,0);a.b_iso=80+i*.1;r.add_atom(a);xyz.append([a.pos.x,a.pos.y,a.pos.z])
            c.add_residue(r);names.append(["N","CA","C","O"]);coords.append(np.array(xyz,dtype=np.float32))
        s[0].add_chain(c);s.setup_entities();s.make_mmcif_document().write_file(str(cif))
        confidence.write_text(json.dumps(dict(plddt=82)))
        bank=root/"raw";p=bank/uid[5:7]/(uid+".pt");p.parent.mkdir(parents=True)
        scalar=torch.zeros(50,89);scalar[:,0]=1
        feature_names=list(audit.RAW89_NAMES)
        data=dict(unique_id=uid,sequence=seq,sequence_sha256=uid[5:],source="protenix_mini_esm_v0.5.0",
                  normalization="raw; train-only fitting deferred",spatial_scalar_raw_features=scalar,
                  spatial_scalar_features=scalar.clone(),physchem_features=torch.zeros(50,62),
                  spatial_vector_features=torch.zeros(50,8,3),ca_coords=torch.tensor(np.stack([x[1]for x in coords])),
                  all_atom_names=names,all_atom_coords=coords,spatial_scalar_feature_names=feature_names)
        torch.save(data,p)
        row=dict(unique_id=uid,length="50",native_split="train",local_cif="",
                 star_candidates_json=json.dumps(["/star/runs/one/"+cif.name]),resolution="star_candidates")
        return uid,cif,confidence,p,bank,data,row

    def options(self, root, fixture, smoke):
        uid,cif,confidence,p,bank,data,row=fixture
        cat=root/"catalog";cat.mkdir()
        with (cat/"sources.tsv").open("w",newline="")as f:
            w=csv.DictWriter(f,fieldnames=list(row),delimiter="\t");w.writeheader();w.writerow(row)
        summary=dict(status="candidate_catalog_complete",unresolved_queries=[],claimed_existing_features=1,
                     native_queries=2,star_root="/star",outputs_sha256={"sources.tsv":audit.sha(cat/"sources.tsv")})
        (cat/"summary.json").write_text(json.dumps(summary));ids=root/"smoke_ids.txt";ids.write_text(uid+"\n")
        return argparse.Namespace(catalog=cat,catalog_sha256=audit.sha(cat/"summary.json"),expected_existing=1,
                                  expected_native=2,smoke_ids=ids if smoke else None,output=root/"out",
                                  snapshot=root/"snapshot",raw_bank=bank,workers=1)

    def test_exact_content_and_confidence_axis(self):
        with tempfile.TemporaryDirectory()as tmp:
            uid,cif,confidence,p,bank,data,row=self.fixture(Path(tmp))
            q,record=audit.validate_content(uid,50,cif,confidence,p)
            self.assertEqual(q.shape,(50,));self.assertAlmostEqual(float(q[0]),.8,places=6)
            self.assertLess(record["heavy_atom_max_abs_difference"],.0011)
            self.assertEqual(record["SASA_positive_residues"],50)

    def test_same_sequence_different_pose_is_rejected(self):
        for atom in ("CA","O"):
            with self.subTest(atom=atom),tempfile.TemporaryDirectory()as tmp:
                uid,cif,confidence,p,bank,data,row=self.fixture(Path(tmp))
                index=data["all_atom_names"][0].index(atom);data["all_atom_coords"][0][index,0]+=1
                if atom=="CA":data["ca_coords"][0,0]+=1
                torch.save(data,p)
                with self.assertRaisesRegex(ValueError,"pose differs"):
                    audit.validate_content(uid,50,cif,confidence,p)

    def test_non_anchor_scalar_column_permutation_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            uid,cif,confidence,p,bank,data,row=self.fixture(Path(tmp))
            for key in ("spatial_scalar_raw_features","spatial_scalar_features"):
                data[key][:,30]=1;data[key][:,31]=2
                data[key][:,[30,31]]=data[key][:,[31,30]]
            names=data["spatial_scalar_feature_names"];names[30],names[31]=names[31],names[30]
            torch.save(data,p)
            with self.assertRaisesRegex(ValueError,"scalar schema"):
                audit.validate_content(uid,50,cif,confidence,p)

    def test_duplicate_CIF_atom_cannot_hide_an_unchecked_pose(self):
        with tempfile.TemporaryDirectory()as tmp:
            uid,cif,confidence,p,bank,data,row=self.fixture(Path(tmp))
            s=gemmi.read_structure(str(cif));residue=s[0][0][0];original=residue.find_atom("O","*")
            matching=original.clone();matching.altloc="B";original.altloc="A";original.pos.x+=1
            residue.add_atom(matching);s.make_mmcif_document().write_file(str(cif))
            with self.assertRaisesRegex(ValueError,"Duplicate CIF heavy-atom"):
                audit.validate_content(uid,50,cif,confidence,p)

    def test_invalid_raw_features_confidence_or_sequence_are_rejected(self):
        for change in ("zero_sasa","nan_tensor","schema","sequence","confidence_scale"):
            with self.subTest(change=change),tempfile.TemporaryDirectory()as tmp:
                uid,cif,confidence,p,bank,data,row=self.fixture(Path(tmp))
                if change=="zero_sasa":data["spatial_scalar_raw_features"][:,0]=0;data["spatial_scalar_features"][:,0]=0
                elif change=="nan_tensor":data["physchem_features"][0,0]=float("nan")
                elif change=="schema":data["spatial_scalar_feature_names"][8]="wrong"
                elif change=="sequence":data["sequence"]="C"*50
                else:confidence.write_text(json.dumps(dict(plddt=101)))
                torch.save(data,p)
                with self.assertRaises(ValueError):audit.validate_content(uid,50,cif,confidence,p)

    def test_missing_transfer_is_pending_not_zero_confidence(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);uid,cif,confidence,p,bank,data,row=self.fixture(root);confidence.unlink()
            record,q=audit.worker((row,dict(star_root="/star"),str(root/"snapshot"),str(bank)))
            self.assertEqual(record["status"],"pending");self.assertIsNone(q)

    def test_ambiguous_sources_do_not_select_a_pose(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);uid,cif,confidence,p,bank,data,row=self.fixture(root)
            row["star_candidates_json"]=json.dumps(["/star/a/"+cif.name,"/star/b/"+cif.name])
            with self.assertRaisesRegex(ValueError,"Ambiguous"):
                audit.source_paths(row,dict(star_root="/star"),root/"snapshot")

    def test_smoke_never_publishes_full_existing_content_ready(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=self.options(root,self.fixture(root),True)
            result=audit.audit(args)
            self.assertEqual(result["status"],"smoke_complete")
            self.assertFalse((root/"out/existing_content_ready.json").exists())
            self.assertFalse(result["normalization_fitted"])

    def test_pending_full_scope_cannot_publish_ready_or_usable_zero_values(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root);args=self.options(root,fixture,False)
            fixture[2].unlink();result=audit.audit(args)
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["counts"],dict(pending=1))
            self.assertTrue(np.isnan(np.load(root/"out/absolute_CA_plddt_div100.npy")).all())
            self.assertFalse((root/"out/existing_content_ready.json").exists())

    def test_wrong_catalog_hash_stops_before_output_creation(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=self.options(root,self.fixture(root),False);args.catalog_sha256="wrong"
            with self.assertRaisesRegex(ValueError,"fingerprint"):audit.audit(args)
            self.assertFalse(args.output.exists())

    def test_real_spawned_worker_CLI_and_existing_only_readiness(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=self.options(root,self.fixture(root),False)
            cmd=[sys.executable,str(SOURCE/"audit_rcsb_structure_content.py"),"--catalog",str(args.catalog),
                 "--catalog-sha256",args.catalog_sha256,"--expected-existing","1","--expected-native","2",
                 "--snapshot",str(args.snapshot),"--raw-bank",str(args.raw_bank),"--output",str(args.output),"--workers","2"]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads((args.output/"existing_content_ready.json").read_text())
            self.assertEqual(receipt["status"],"existing_content_complete")
            self.assertEqual(receipt["existing_feature_scope"],1)
            self.assertEqual(receipt["native_queries"],2)
            self.assertFalse(receipt["full_native_structure_coverage"])


if __name__=="__main__":unittest.main()
