# Strong backbone reruns on MMseq cov0 cluster-mode1 split

Immediate experiments use the built-in `ESMSiteClassifier` Transformer path.

Mamba was not submitted because `mamba_ssm` is not installed in the hpc2 `af3` environment (`importlib.util.find_spec("mamba_ssm") is None`).

Common settings:

- split: `features/contact_labels/splits_mmseq30_cov0_clust1_tmk_no_len_limit_predstruct_relaxed`
- ESM: MLC root, `features/esm2_t33_650M_UR50D_mlc/pt`
- PredStruct scalar root: `features/pred_struct_sequence_scalar_relaxed/pt`
- backbone change: `n_transformer_layers=2`, `n_heads=8`, `transformer_ff_mult=4`
- batch size: 8, AMP enabled, lr `3e-5`

Configs:

- `train_contact_site_esm_mlc_predstruct_full_transformer2_seed42_cov0clust1_20260619.yaml`
- `train_contact_site_esm_mlc_predstruct_nograph_transformer2_seed42_cov0clust1_20260619.yaml`
