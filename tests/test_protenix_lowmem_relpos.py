import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];SCRIPTS=ROOT/"scripts/preparations/m18_union"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("lowmem",SCRIPTS/"protenix_lowmem_relpos.py")
low=importlib.util.module_from_spec(spec);spec.loader.exec_module(low)


def stock(self, d):
    with torch.no_grad():
        a,r,e,s,t=[d[k] for k in low.FIELDS]
        bc=(a[..., :,None]==a[...,None,:]).long();br=(r[..., :,None]==r[...,None,:]).long();be=(e[..., :,None]==e[...,None,:]).long()
        dr=torch.clip(r[..., :,None]-r[...,None,:]+self.r_max,0,2*self.r_max)*bc+(1-bc)*(2*self.r_max+1)
        dt=torch.clip(t[..., :,None]-t[...,None,:]+self.r_max,0,2*self.r_max)*bc*br+(1-bc*br)*(2*self.r_max+1)
        ds=torch.clip(s[..., :,None]-s[...,None,:]+self.s_max,0,2*self.s_max)*be+(1-be)*(2*self.s_max+1)
        d["relp"]=torch.cat([F.one_hot(dr,2*(self.r_max+1)),F.one_hot(dt,2*(self.r_max+1)),be[...,None],F.one_hot(ds,2*(self.s_max+1))],-1).float()
    return d


class LowMemoryRelativePositionTests(unittest.TestCase):
    def test_matches_full_reference_for_shapes_bounds_and_chunks(self):
        for n,batch in ((1,()),(7,()),(17,(2,)),(19,(2,3))):
            for r,s in ((0,0),(3,1),(32,2)):
                for dtype in (torch.int32,torch.int64):
                    fields=low.synthetic(n,batch,dtype,mixed=True);obj=SimpleNamespace(r_max=r,s_max=s)
                    expected=stock(obj,dict(fields))["relp"]
                    for chunk in (1,3,64):
                        with self.subTest(n=n,batch=batch,r=r,s=s,dtype=dtype,chunk=chunk):
                            d=dict(fields);result=low.generate_relp_chunked(obj,d,chunk)
                            self.assertIs(result,d);self.assertTrue(torch.equal(result["relp"],expected))
                            self.assertTrue(result["relp"].is_contiguous());self.assertFalse(result["relp"].requires_grad)
                            for key in low.FIELDS:self.assertIs(result[key],fields[key])

    def test_no_rng_consumption_and_noncontiguous_input(self):
        fields={k:v[::2] for k,v in low.synthetic(18,mixed=True).items()};state=torch.get_rng_state()
        obj=SimpleNamespace(r_max=32,s_max=2)
        a=stock(obj,dict(fields))["relp"];b=low.generate_relp_chunked(obj,dict(fields),4)["relp"]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_invalid_shape_dtype_or_chunk_fails(self):
        obj=SimpleNamespace(r_max=32,s_max=2)
        for changes,chunk in (({},0),({"sym_id":torch.zeros(4,dtype=torch.long)},3),({"sym_id":torch.zeros(7)},3)):
            with self.subTest(changes=list(changes),chunk=chunk):
                with self.assertRaises(ValueError):low.generate_relp_chunked(obj,{**low.synthetic(7),**changes},chunk)

    def test_proof_pin_and_full_scale_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"proof.json";p.write_text('{}')
            env=dict(PPC_LOWMEM_RELP_ACCEPTANCE=str(p),PPC_LOWMEM_RELP_ACCEPTANCE_SHA256="wrong")
            with mock.patch.dict(os.environ,env):
                with self.assertRaisesRegex(ValueError,"proof changed"):low.checked_proof()
            with self.assertRaisesRegex(ValueError,"complete historical maximum"):low.prove(p,50)

    def test_failed_postcheck_never_publishes_and_restores_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);out=base/"outputs";prior=os.getcwd();os.chdir(base)
            class Dummy:
                generate_relp=stock
            module=SimpleNamespace(RelativePositionEncoding=Dummy,__file__=inspect.getsourcefile(stock))
            original=Dummy.generate_relp;args=["pred","--out_dir",str(out)];uid="useq_test"
            def fake_fold(argv):
                self.assertIsNot(Dummy.generate_relp,original);Path(argv[-1]).mkdir()
            try:
                with mock.patch.object(low.full,"input_rows",return_value=(SimpleNamespace(out_dir=str(out)),{uid:"A"*50},"h")), \
                        mock.patch.object(low,"checked_proof",side_effect=[{},ValueError("proof replaced")]), \
                        mock.patch.object(low,"checked_runtime",return_value=module),mock.patch.object(low.full,"fold",side_effect=fake_fold):
                    with self.assertRaisesRegex(ValueError,"proof replaced"):low.predict(args)
                self.assertIs(Dummy.generate_relp,original);self.assertFalse(out.exists())
                self.assertTrue((base/"outputs.lowmem_unaccepted").is_dir())
                self.assertTrue((base/"esm_embeddings/esm_error"/(uid+"_1.txt")).is_file())
            finally:os.chdir(prior)

    def test_successful_postcheck_publishes_only_after_restoring_stock_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);out=base/"outputs";prior=os.getcwd();os.chdir(base)
            class Dummy:
                generate_relp=stock
            module=SimpleNamespace(RelativePositionEncoding=Dummy,__file__=inspect.getsourcefile(stock))
            original=Dummy.generate_relp;args=["pred","--out_dir",str(out)]
            def fake_fold(argv):
                self.assertIsNot(Dummy.generate_relp,original)
                self.assertFalse(out.exists());Path(argv[-1]).mkdir()
            def checked_runtime():
                self.assertIs(Dummy.generate_relp,original)
                return module
            try:
                with mock.patch.object(low.full,"input_rows",return_value=(SimpleNamespace(out_dir=str(out)),{"useq_test":"A"*50},"h")), \
                        mock.patch.object(low,"checked_proof",return_value={}), \
                        mock.patch.object(low,"checked_runtime",side_effect=checked_runtime),mock.patch.object(low.full,"fold",side_effect=fake_fold):
                    low.predict(args)
                self.assertTrue(out.is_dir());self.assertFalse((base/"outputs.lowmem_unaccepted").exists())
                self.assertIs(Dummy.generate_relp,original)
            finally:os.chdir(prior)


if __name__=="__main__":unittest.main()
