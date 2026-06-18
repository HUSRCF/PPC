# 2026-06-19 HPC UE experiment archive

Archived copies of the configs submitted to `i64m1tga800ue` on 2026-06-19. The original submitted paths under `configs/` remain in place because pending Slurm jobs reference those exact paths via `CONFIG=...`.

## Groups

- `sequence_only/no_context/`: ESM embeddings only, no external predicted-structure scalar features, no context modules.
- `sequence_only/contact_prior/`: sequence-only models with ESM-derived contact graph/contact prior; no predicted-structure scalar features.
- `predicted_structure_sequence/no_context/`: predicted-structure-derived scalar residue features without chain/global/contact context.
- `predicted_structure_sequence/contact_prior/`: predicted-structure-derived scalar features plus final/multilayer ESM contact prior.
- `predicted_structure_sequence/ablations/`: predicted-structure full-model ablations, including no contact graph and weak regularization.
- `manifests/`: submitted config-to-output-dir manifests.

## Job sets

- `hpcue_predstruct_batch_20260619_002109.tsv`: jobs `9887660`-`9887670`.
- `hpcue_predstruct_moreseeds_20260619_002328.tsv`: jobs `9887677`-`9887686`.

All configs use split `features/contact_labels/splits_mmseq30_tmk_no_len_limit_predstruct_relaxed`. Predicted-structure scalar configs use `features/pred_struct_sequence_scalar_relaxed/pt`.
