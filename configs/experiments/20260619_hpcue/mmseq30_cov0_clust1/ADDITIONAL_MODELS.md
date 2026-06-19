# Additional cov0 cluster-mode1 model reruns

These configs complete the representative model matrix on split `features/contact_labels/splits_mmseq30_cov0_clust1_tmk_no_len_limit_predstruct_relaxed`.

Already submitted separately:

- MLC + PS + no graph
- MLC + PS + contact

Additional configs in `additional_models/`:

- MLC + PS + contact weak-reg
- MLC + no scalar + contact
- Final + PS + contact, using `features/esm2_t33_650M_UR50D_contact/pt` to avoid empty graph
- Final + no scalar + contact, using `features/esm2_t33_650M_UR50D_contact/pt`
- Final + PS no-context
- Final + no scalar no-context
- Final + PS + context no graph
