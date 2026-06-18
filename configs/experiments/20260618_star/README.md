# 2026-06-18 star experiment archive

Archived config for the extra star GPU0 run launched in tmux session `ppc_final_ps_ctx_ng_20260618_171453`.

## Purpose

This run tests final-layer ESM embeddings plus predicted-structure-derived scalar residue features with chain/global/position context, but without the ESM contact graph. It separates the effect of predicted-structure scalar features from contact-prior and multilayer ESM effects.

## Runtime paths on star

- Config: `~/Code/PPC/configs/train_contact_site_esm_final_predstruct_context_nograph_seed42_star_20260618_171453.yaml`
- Log: `~/Code/PPC/logs/ppc_final_predstruct_context_nograph_star_20260618_171453.log`
- Output: `~/Code/PPC/runs/contact_site_esm_final_predstruct_context_nograph_reg_seed42_e30_lr5e5_b96_star_20260618_171453`
- Tmux: `ppc_final_ps_ctx_ng_20260618_171453`
