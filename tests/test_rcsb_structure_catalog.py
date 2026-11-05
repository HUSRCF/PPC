import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("catalog", ROOT/"scripts/preparations/m18_union/catalog_rcsb_structures.py")
catalog = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def fixture(self, root):
        rows=[]
        for i, seq in enumerate(("A"*50, "C"*51, "G"*52)):
            digest=hashlib.sha256(seq.encode()).hexdigest()
            rows.append(dict(unique_id="useq_"+digest, sequence=seq, length=len(seq), seq_sha256=digest,
                             native_split=("train", "val", "test")[i]))
        queries=root/"queries.tsv"
        with queries.open("w",newline="")as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
        missing=root/"missing.tsv"
        with missing.open("w",newline="")as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerow(rows[2])
        uid=rows[0]["unique_id"];local=root/"local.jsonl"
        local.write_text(json.dumps(dict(unique_id=uid,n_residues=50,source_cif="/hpc3/"+uid+"_sample_0.cif"))+"\n")
        uid=rows[1]["unique_id"];ledger=root/"ledger.jsonl";retained=root/"retained.jsonl"
        record=dict(unique_id=uid,n_residues=51,status="ok",cif="/star/runs/a/"+uid+"_sample_0.cif")
        ledger.write_text(json.dumps(record)+"\n");retained.write_text("")
        return queries,missing,local,ledger,retained,rows

    def run_catalog(self, root, fixture):
        q,m,l,s,r,rows=fixture
        queries=catalog.read_queries(q,catalog.sha(q),3)
        return catalog.build(queries,m,l,[s],[r],Path("/star"),root/"out",2)

    def test_catalog_is_not_content_acceptance_and_no_source_is_modified(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root)
            before={p:catalog.sha(p)for p in fixture[:5]}
            result=self.run_catalog(root,fixture)
            self.assertEqual(result["counts"],dict(local_index=1,star_candidates=1))
            self.assertEqual(result["transfer_files"],1)
            self.assertFalse(result["CIF_content_accepted"])
            self.assertFalse(result["file_existence_checked"])
            self.assertEqual(before,{p:catalog.sha(p)for p in fixture[:5]})
            self.assertNotIn("/star/",(root/"out/star_files.txt").read_text())

    def test_retained_index_resolves_missing_queue_source_without_score_selection(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root);q,m,l,s,r,rows=fixture
            record=json.loads(s.read_text());record.pop("n_residues");record.pop("status")
            r.write_text(json.dumps(record)+"\n");s.write_text("")
            result=self.run_catalog(root,fixture)
            self.assertEqual(result["status"],"candidate_catalog_complete")
            self.assertEqual(result["transfer_files"],1)
            self.assertEqual(result["retained_records_without_length"],1)

    def test_missing_catalog_entry_remains_explicit(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root);fixture[3].write_text("")
            result=self.run_catalog(root,fixture)
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(len(result["unresolved_queries"]),1)

    def test_multiple_structures_are_not_silently_chosen(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root);q,m,l,s,r,rows=fixture
            record=json.loads(s.read_text());record["cif"]=record["cif"].replace("/runs/a/","/runs/b/")
            r.write_text(json.dumps(record)+"\n")
            result=self.run_catalog(root,fixture)
            self.assertEqual(result["transfer_files"],2)
            self.assertEqual(result["additional_candidates"],1)

    def test_query_fingerprint_and_length_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root)
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                catalog.read_queries(fixture[0],"wrong",3)
            record=json.loads(fixture[3].read_text());record["n_residues"]=50
            fixture[3].write_text(json.dumps(record)+"\n")
            with self.assertRaisesRegex(ValueError,"length"):
                self.run_catalog(root,fixture)

    def test_paths_and_truncated_ledgers_fail_closed(self):
        for bad in ("/star/../secret/uid.cif","/elsewhere/uid.cif","/star/wrong.cif"):
            with self.subTest(path=bad),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);fixture=self.fixture(root);r=json.loads(fixture[3].read_text());r["cif"]=bad
                fixture[3].write_text(json.dumps(r)+"\n")
                with self.assertRaises(ValueError):self.run_catalog(root,fixture)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture=self.fixture(root);p=fixture[3];p.write_text(p.read_text().rstrip("\n"))
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                self.run_catalog(root,fixture)


if __name__ == "__main__":
    unittest.main()
