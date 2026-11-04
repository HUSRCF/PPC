import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppcbind.models.esm_site import ESMSiteClassifier


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/preparations/m18_union" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def small_config(**changes):
    return dict(
        d_esm=6, d_seq=4, d_prottrans=8, d_model=16, d_hidden=32,
        dropout=0., classifier_dropout=0., n_heads=4, geometry_heads=4,
        use_contact_graph=False, geometry_mode="edge_attention",
        use_prottrans_embeddings=True, prottrans_fusion_mode="zero_init_pre_contact_residual",
        prottrans_residual_dim=4, prottrans_residual_hidden=4, prottrans_residual_dropout=0.,
        **changes,
    )


def payload():
    return dict(
        esm_embeddings=torch.randn(2, 5, 6), seq_features=torch.randn(2, 5, 4),
        prottrans_embeddings=torch.randn(2, 5, 8),
        protein_mask=torch.tensor([[True] * 5, [True] * 3 + [False] * 2]),
        geometry_vectors=torch.randn(2, 5, 8, 3),
        geometry_edge_index=torch.tensor([[0, 1, 5, 6], [1, 0, 6, 5]]),
        geometry_edge_features=torch.randn(4, 30),
        geometry_edge_vectors=torch.randn(4, 1, 3),
        geometry_edge_scores=torch.ones(4),
    )


class GeometryMSATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(49)

    def test_position_changes_neither_initial_parameters_nor_rng(self):
        first = ESMSiteClassifier(small_config())
        rng = torch.get_rng_state()
        torch.manual_seed(49)
        second = ESMSiteClassifier(small_config(msa_residual_position="post_geometry"))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        x = payload()
        first.eval()
        second.eval()
        self.assertTrue(torch.equal(first(**x)["logits"], second(**x)["logits"]))

    def test_late_msa_does_not_modify_geometry_inputs(self):
        x = payload()
        changed = {**x, "prottrans_embeddings": torch.randn_like(x["prottrans_embeddings"])}
        for position in ("post_tcn", "post_geometry"):
            model = ESMSiteClassifier(small_config(msa_residual_position=position)).eval()
            with torch.no_grad():
                model.prottrans_residual_adapter[-1].weight.normal_(std=.2)
            seen = []
            hook = model.geometry_blocks[0].register_forward_pre_hook(
                lambda _, args: seen.append(args[0].detach().clone()))
            a, b = model(**x)["logits"], model(**changed)["logits"]
            hook.remove()
            self.assertEqual(torch.equal(*seen), position == "post_geometry")
            self.assertFalse(torch.equal(a[x["protein_mask"]], b[x["protein_mask"]]))
            self.assertTrue(torch.equal(a[~x["protein_mask"]], torch.zeros_like(a[~x["protein_mask"]])))

    def test_late_msa_requires_geometry_and_payload(self):
        with self.assertRaises(ValueError):
            ESMSiteClassifier({**small_config(), "geometry_mode": "none", "msa_residual_position": "post_geometry"})
        x = payload()
        x.pop("prottrans_embeddings")
        model = ESMSiteClassifier(small_config(msa_residual_position="post_geometry"))
        with self.assertRaises(ValueError):
            model(**x)

    def test_two_geometry_layers_and_msa_both_learn(self):
        x = payload()
        for position in ("post_tcn", "post_geometry"):
            model = ESMSiteClassifier(small_config(geometry_layers=2, msa_residual_position=position))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            for _ in range(2):
                optimizer.zero_grad()
                output = model(**x)["logits"]
                output[x["protein_mask"]].square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            self.assertGreater(model.prottrans_residual_adapter[-1].weight.abs().sum().item(), 0)
            for block in model.geometry_blocks:
                self.assertGreater(block.update[-1].weight.abs().sum().item(), 0)

    def test_matrix_contract_and_parent_immutability(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        before = copy.deepcopy(parents)
        outputs = set()
        for name in screen.VARIANTS:
            cfg = screen.build(*parents, Path("/tmp/geo_msa_test"), name)
            outputs.add(cfg["training"]["output_dir"])
            t = cfg["training"]
            self.assertEqual((t["seed"], t["epochs"], t["selection_metric"]), (49, 50, "pooled_ap"))
            self.assertTrue(t["validation_only"])
            self.assertFalse(t["eval_test_each_epoch"])
            self.assertIsNone(t["init_checkpoint"])
            self.assertEqual(t["max_residues"], 0)
            self.assertIn("pooled_ap", t["save_metric_checkpoints"].split(","))
            self.assertTrue(cfg["data"]["require_prottrans_embeddings"])
            self.assertEqual(cfg["model"]["d_prottrans"], 768)
            for key in screen.IDENTITY_KEYS:
                self.assertEqual(cfg["data"][key], parents[0]["data"][key])
            if name not in {"K7_G2_MSA_residue_mean", "K8_G2_MSA_pure_CE",
                            "K11_G2_MSA_twohop_pure_CE", "K13_G2_MSA_threehop_pure_CE"}:
                self.assertEqual((t["loss_reduction"], t["ratio_loss_weight"]), ("chain_mean", .75))
        self.assertEqual(parents, before)
        self.assertEqual(len(outputs), 14)
        base = screen.build(*parents, Path("/tmp/geo_msa_test"), "K1_G2_MSA")
        for name, axis in (("K7_G2_MSA_residue_mean", "loss_reduction"), ("K8_G2_MSA_pure_CE", "ratio_loss_weight")):
            variant = screen.build(*parents, Path("/tmp/geo_msa_test"), name)
            self.assertEqual(base["data"], variant["data"])
            self.assertEqual(base["model"], variant["model"])
            different = {key for key in base["training"] if base["training"][key] != variant["training"][key]}
            self.assertEqual(different, {"output_dir", axis})

    def test_depth_ratio_factorial_has_single_loss_axis(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        pairs = (
            ("K1_G2_MSA", "K8_G2_MSA_pure_CE", 1),
            ("K3_G2_MSA_twohop", "K11_G2_MSA_twohop_pure_CE", 2),
            ("K12_G2_MSA_threehop", "K13_G2_MSA_threehop_pure_CE", 3),
        )
        for regular, pure, depth in pairs:
            a = screen.build(*parents, Path("/tmp/factorial_test"), regular)
            b = screen.build(*parents, Path("/tmp/factorial_test"), pure)
            self.assertEqual(a["model"], b["model"])
            self.assertEqual(a["data"], b["data"])
            self.assertEqual(a["model"]["geometry_layers"], depth)
            diff = {k for k in a["training"] if a["training"][k] != b["training"][k]}
            self.assertEqual(diff, {"output_dir", "ratio_loss_weight"})
            self.assertEqual((a["training"]["ratio_loss_weight"], b["training"]["ratio_loss_weight"]), (.75, 0.))

    def test_incompatible_parent_split_is_rejected(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        b1 = geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q")
        g2 = geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention")
        encoder = msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49)
        encoder["data"]["split_dir"] = "/different/split"
        with self.assertRaises(ValueError):
            screen.build(b1, g2, encoder, Path("/tmp/geo_msa_test"), "K1_G2_MSA")

    def test_preparation_paths_are_normalized(self):
        screen = module("prepare_geometry_msa_screen")
        self.assertTrue(screen.absolute_path(".").is_absolute())


if __name__ == "__main__":
    unittest.main()
