# Dataset and Evaluation Leakage Audit (2026-07-12)

## Conclusion

No train/validation/test leakage was found in the chain-filtered global-SI30 training universe used by M7/M8. The audit covers identifiers, source complexes, sequence components, exact sequences, and the actual resolved feature/cache files, not only the three split text files.

The training loop also has no validation/test gradient path. `_evaluate` uses `model.eval()` under `torch.no_grad()` and contains no backward or optimizer/scaler step. M8 additionally enforces this at runtime by checking that evaluation starts and ends with no parameter gradients, that logits do not require gradients, and that no parameter version changes during evaluation.

## Data Evidence

- Chains: train `14,052`, validation `3,796`, test `2,392`; total `20,240`.
- Source PDBs: train `4,192`, validation `1,028`, test `630`.
- Global-SI components: train `240`, validation `55`, test `43`.
- Unique sequence SHA1 values: train `4,868`, validation `1,075`, test `724`.
- Duplicate chain IDs within a split: `0`.
- Cross-split overlap for chain ID, source PDB, component ID, exact sequence SHA1, exact group, and representative ID: `0` in every pair of splits.
- Both chain manifests contain exactly the same `20,240` chain IDs as the split union, with no split-field mismatch.

## Resolved File Evidence

Every configured input was resolved with the same candidate order used by the dataset loader. Missing files and cross-split resolved-path overlaps were both zero for:

- source compact ESM payloads;
- PDB-level label payloads;
- ESMFold-derived per-chain sequence/structure scalar features;
- ESM-C per-chain embeddings, including resolution of all symlink targets;
- ESM2 contact graphs, including resolution of all symlink targets.

The ESM-C and contact roots contain `20,240` chain aliases pointing to `6,667` exact-sequence payloads. Resolving those aliases produced `4,868 / 1,075 / 724` unique targets in train/validation/test, respectively, and no target was shared across split boundaries.

## Gradient Evidence

Before this change, a parameter's `.grad` could remain populated after the final training batch because gradients were cleared at the start of the next training step. Observing such a tensor during validation would therefore show a **stale training gradient**, not a validation gradient. Validation still ran under `torch.no_grad()` and did not call backward or update weights.

The training script now clears gradients immediately before validation and performs strict runtime checks. A deterministic positive-control test showed:

- intentionally retained training gradients are detected and cause evaluation to fail;
- after `zero_grad(set_to_none=True)`, evaluation leaves `0` parameter gradients;
- evaluation causes `0` parameter-version changes;
- validation logits have `requires_grad=False`;
- with `ratio_loss_weight=0`, CE, CE+Dice, Focal+Dice, and chain-mean CE remain bitwise equal to the previous implementation (`torch.equal`).

When `eval_test_each_epoch=false`, the training script now does not read test IDs and does not construct a test loader. M8 uses this setting and selects checkpoints only by validation chain-macro AP.

## SI30 Evidence

The existing high-sensitivity DeepTMInter-style audit reports:

- `6,667` unique exact sequences;
- `252,542` recalled cross-split candidate pairs globally realigned with BLOSUM62, gap-open `-10`, gap-extension `-0.5`;
- maximum cross-split global identity `0.2826855`;
- SI >= `0.30` violations: `0`.

A fresh 110-thread rerun was launched in tmux session `ppc_si30_audit_0712`. It regenerates and hash-checks the sequence universe, then recomputes both the high-sensitivity and exhaustive-profile candidate alignments from scratch. It intentionally reuses the SHA-recorded MMseqs candidate files; therefore the remaining limitation is candidate recall rather than alignment reproducibility. This is not a mathematical all-pairs dynamic-programming proof.

## Validation Overfitting Control

Repeated model selection on validation is not train/test leakage, but it can overfit the validation set. The control policy is:

1. screen seed 42 using validation chain-macro AP only;
2. require at least `+0.005` macro AP over M7c and no more than `0.003` pooled-AP regression;
3. confirm advancing variants with seeds 43 and 44;
4. evaluate the frozen winner on test once.

## Artifacts

- Machine-readable audit: `benchmark/audits/esm_site_data_isolation_20260712/audit.json`
- Human-readable audit: `benchmark/audits/esm_site_data_isolation_20260712/README.md`
- Reusable audit script: `scripts/analysis/audit_esm_site_data_isolation.py`
- Training runtime checks: `scripts/training/train_esm_site.py`
