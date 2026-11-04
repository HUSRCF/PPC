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
                            "K11_G2_MSA_twohop_pure_CE", "K13_G2_MSA_threehop_pure_CE",
                            "K26_G2_MSA_twohop_preTCN_pure_CE", "K28_G2_MSA_twohop_residue_mean",
                            "K29_G2_MSA_twohop_preTCN_residue_mean"}:
                self.assertEqual((t["loss_reduction"], t["ratio_loss_weight"]), ("chain_mean", .75))
        self.assertEqual(parents, before)
        self.assertEqual(len(outputs), 30)
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

    def test_msa_width_screen_changes_only_bottleneck_and_capacity(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        base = screen.build(*parents, Path("/tmp/width_test"), "K3_G2_MSA_twohop")
        counts = {}
        widths = {
            "K14_G2_MSA_twohop_width128": (128, 128),
            "K15_G2_MSA_twohop_width256": (256, 128),
            "K16_G2_MSA_twohop_capacity128": (64, 228),
            "K17_G2_MSA_twohop_capacity256": (64, 428),
        }
        for name, (projection, hidden) in widths.items():
            variant = screen.build(*parents, Path("/tmp/width_test"), name)
            self.assertEqual(base["data"], variant["data"])
            diff = {k for k in base["training"] if base["training"][k] != variant["training"][k]}
            self.assertEqual(diff, {"output_dir"})
            model_diff = {k for k in base["model"] if base["model"][k] != variant["model"][k]}
            expected_diff = {"prottrans_residual_hidden"}
            if projection != 64:
                expected_diff.add("prottrans_residual_dim")
            self.assertEqual(model_diff, expected_diff)
            model = ESMSiteClassifier(variant["model"])
            self.assertEqual(model.prottrans_proj.proj[0].out_features, projection)
            self.assertEqual(model.prottrans_residual_adapter[0].out_features, hidden)
            self.assertEqual(model.prottrans_residual_adapter[-1].out_features, 256)
            self.assertEqual(torch.count_nonzero(model.prottrans_residual_adapter[-1].weight), 0)
            self.assertEqual(len(model.geometry_blocks), 2)
            counts[name] = sum(p.numel() for p in model.parameters())
        self.assertEqual(abs(counts["K14_G2_MSA_twohop_width128"]
                             - counts["K16_G2_MSA_twohop_capacity128"]), 36)
        self.assertEqual(abs(counts["K15_G2_MSA_twohop_width256"]
                             - counts["K17_G2_MSA_twohop_capacity256"]), 108)

    def test_preparation_paths_are_normalized(self):
        screen = module("prepare_geometry_msa_screen")
        self.assertTrue(screen.absolute_path(".").is_absolute())

    def test_regularization_screen_changes_only_declared_knobs(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        base = screen.build(*parents, Path("/tmp/regularization"), "K3_G2_MSA_twohop")
        axes = {
            "K19_G2_MSA_twohop_dropout_low": ({"dropout": .15, "classifier_dropout": .30}, {}),
            "K20_G2_MSA_twohop_dropout_high": ({"dropout": .35, "classifier_dropout": .55}, {}),
            "K21_G2_MSA_twohop_wd003": ({}, {"weight_decay": .03}),
            "K22_G2_MSA_twohop_wd020": ({}, {"weight_decay": .20}),
            "K23_G2_MSA_twohop_ls002": ({}, {"label_smoothing": .02}),
            "K24_G2_MSA_twohop_msa_drop010": ({"prottrans_residual_dropout": .10}, {}),
        }
        torch.manual_seed(49)
        reference = ESMSiteClassifier(base["model"])
        state, rng = reference.state_dict(), torch.get_rng_state()
        for name, (model_axis, training_axis) in axes.items():
            cfg = screen.build(*parents, Path("/tmp/regularization"), name)
            self.assertEqual(cfg["data"], base["data"])
            self.assertEqual({k for k in base["model"] if cfg["model"][k] != base["model"][k]}, set(model_axis))
            self.assertEqual({k for k in base["training"] if cfg["training"][k] != base["training"][k]},
                             {"output_dir", *training_axis})
            for key, value in model_axis.items():
                self.assertEqual(cfg["model"][key], value)
            for key, value in training_axis.items():
                self.assertEqual(cfg["training"][key], value)
            torch.manual_seed(49)
            model = ESMSiteClassifier(cfg["model"])
            self.assertEqual(sum(p.numel() for p in model.parameters()), 3693034)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
            if name == "K24_G2_MSA_twohop_msa_drop010":
                self.assertEqual(model.prottrans_proj.proj[3].p, .10)
                self.assertEqual(model.prottrans_residual_adapter[2].p, .10)

    def test_fusion_position_depth_factorial_changes_no_other_inputs(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        for post_name, pre_name, depth in (
            ("K1_G2_MSA", "K2_G2_MSA_preTCN", 1),
            ("K3_G2_MSA_twohop", "K18_G2_MSA_twohop_preTCN", 2),
        ):
            post = screen.build(*parents, Path("/tmp/position_depth"), post_name)
            pre = screen.build(*parents, Path("/tmp/position_depth"), pre_name)
            self.assertEqual(post["data"], pre["data"])
            self.assertEqual({k for k in post["training"] if post["training"][k] != pre["training"][k]},
                             {"output_dir"})
            self.assertEqual({k for k in post["model"] if post["model"][k] != pre["model"][k]},
                             {"msa_residual_position"})
            self.assertEqual(pre["model"]["geometry_layers"], depth)
        x = payload()
        torch.manual_seed(49)
        post_model = ESMSiteClassifier(small_config(geometry_layers=2)).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(49)
        pre_model = ESMSiteClassifier(small_config(geometry_layers=2, msa_residual_position="pre_tcn")).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in post_model.state_dict().items():
            self.assertTrue(torch.equal(value, pre_model.state_dict()[name]), name)
        self.assertTrue(torch.equal(post_model(**x)["logits"], pre_model(**x)["logits"]))

    def test_pre_tcn_interaction_cells_are_single_axis_controls(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        for post_name, pre_name in (
            ("K23_G2_MSA_twohop_ls002", "K25_G2_MSA_twohop_preTCN_ls002"),
            ("K11_G2_MSA_twohop_pure_CE", "K26_G2_MSA_twohop_preTCN_pure_CE"),
            ("K15_G2_MSA_twohop_width256", "K27_G2_MSA_twohop_preTCN_width256"),
        ):
            post = screen.build(*parents, Path("/tmp/interactions"), post_name)
            pre = screen.build(*parents, Path("/tmp/interactions"), pre_name)
            self.assertEqual(pre["data"], post["data"])
            self.assertEqual({k for k in pre["training"] if pre["training"][k] != post["training"][k]},
                             {"output_dir"})
            self.assertEqual({k for k in pre["model"] if pre["model"][k] != post["model"][k]},
                             {"msa_residual_position"})
        self.assertEqual(screen.build(*parents, Path("/tmp/interactions"),
                                     "K25_G2_MSA_twohop_preTCN_ls002")["training"]["label_smoothing"], .02)
        self.assertEqual(screen.build(*parents, Path("/tmp/interactions"),
                                     "K26_G2_MSA_twohop_preTCN_pure_CE")["training"]["ratio_loss_weight"], 0.)

    def test_twohop_aggregation_cells_keep_models_and_inputs(self):
        geometry = module("prepare_geometry_screen")
        msa = module("prepare_msa_factorial")
        screen = module("prepare_geometry_msa_screen")
        parents = (
            geometry.build(ROOT, Path("/tmp/geo_parent"), "B1_fixed91q"),
            geometry.build(ROOT, Path("/tmp/geo_parent"), "G2_edge_attention"),
            msa.build_config(ROOT, Path("/tmp/msa_parent"), "D", "encoder", 49),
        )
        for reference, changed in (("K3_G2_MSA_twohop", "K28_G2_MSA_twohop_residue_mean"),
                                   ("K18_G2_MSA_twohop_preTCN", "K29_G2_MSA_twohop_preTCN_residue_mean")):
            first = screen.build(*parents, Path("/tmp/aggregation"), reference)
            second = screen.build(*parents, Path("/tmp/aggregation"), changed)
            self.assertEqual(first["data"], second["data"])
            self.assertEqual(first["model"], second["model"])
            self.assertEqual({k for k in first["training"] if first["training"][k] != second["training"][k]},
                             {"output_dir", "loss_reduction"})
            self.assertEqual(second["training"]["loss_reduction"], "residue_mean")
            self.assertEqual(second["training"]["ratio_loss_weight"], .75)
            self.assertEqual(second["training"]["label_smoothing"], 0.)

    def test_wide_msa_paths_receive_task_gradient_after_zero_init(self):
        x = payload()
        targets = torch.arange(int(x["protein_mask"].sum())) % 2
        for projection, hidden in ((128, 128), (256, 128), (64, 228), (64, 428)):
            cfg = {**small_config(geometry_layers=2), "prottrans_residual_dim": projection,
                   "prottrans_residual_hidden": hidden}
            model = ESMSiteClassifier(cfg)
            weight = model.prottrans_proj.proj[0].weight
            initial = weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
            for _ in range(2):
                optimizer.zero_grad()
                logits = model(**x)["logits"]
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.equal(logits[~x["protein_mask"]],
                                            torch.zeros_like(logits[~x["protein_mask"]])))
                torch.nn.functional.cross_entropy(logits[x["protein_mask"]], targets).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                optimizer.step()
            self.assertFalse(torch.equal(initial, weight), (projection, hidden))


if __name__ == "__main__":
    unittest.main()
