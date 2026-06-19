# MMseq30 cov0 cluster-mode1 reruns

This experiment repeats the main predicted-structure sequence models on a regenerated MMseq split.

MMseq command:

```bash
mmseqs easy-cluster features/mmseq30/chains.fasta features/mmseq30_cov0_clust1/mmseq30_cov0_clust1 features/mmseq30_cov0_clust1/tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1 --threads 32
```

Split used by training: `features/contact_labels/splits_mmseq30_cov0_clust1_tmk_no_len_limit_predstruct_relaxed`.

Models:

- `train_contact_site_esm_mlc_predstruct_nograph_seed42_cov0clust1_20260619.yaml`: MLC + predicted-structure scalar, no ESM contact graph.
- `train_contact_site_esm_mlc_predstruct_full_seed42_cov0clust1_20260619.yaml`: MLC + predicted-structure scalar + ESM contact graph.
