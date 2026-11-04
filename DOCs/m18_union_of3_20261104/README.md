# M18 Exact-Sequence Union and OpenFold3 Integration

## Scope and Contracts

This directory indexes the completed single-chain experiments, not TMINTER3,
whole-RCSB pairwise training, protein-ligand prediction or a new model release.
The original M18 models and their historical test results are retained unchanged.

- 20,240 retained chain contexts are grouped by exact full observed-structure sequence.
- 6,667 unique sequences: train 4,868 / val 1,075 / test 724.
- A position is positive if any retained assessed context labels it positive;
 it is negative if assessed but never positive; never-assessed positions stay-100.
- This is an ever-observed union target, not an assembly-specific probability.
- Original 5.5 A heavy-atom all-partner labels are aggregated; geometry is not relabeled.
- Author residue numbering and insertion codes remain in the reverse chain mapping.
- Exact sequences, PDBs and original SI30 components do not cross splits.
- No UniProt/TmAlphaFold bridge or full-canonical-sequence replacement is used.

See `union_dataset_summary.json` and the external `chain_to_union.tsv` for the
frozen source audit and mapping. Split membership was not recomputed in this work.

## Structure and Feature Pipeline

1. Build the exact-sequence label union and retain per-position support counts.
2. Predict all 6,667 monomers with OF3 v0.5.0/OpenBind-0, five seed-42 samples each,
   three recycles, 200 diffusion steps, unpaired UniRef90 MSA, no templates.
3. Validate exact sequence, finite coordinates and confidence dimensions.
4. Select maximum official `sample_ranking_score`; ties choose smallest sample ID.
   No labels are used. Preserve the exact query through CIF/PDB conversion.
5. Extract the existing `protein_features_v4` 89D scalar descriptors. Fit normalization
   on 4,868 train sequences only and append one availability channel, giving 90D.
6. Reuse exact-string-validated ESM-C MLC 12/24/36 embeddings (3456D) and ESM2
   contact graphs (top16, score>=0.05, sequence separation>=6, bidirectional).
7. Pack identity metadata, union labels, normalized features, embeddings, contact
   graphs, split lists, manifest, normalization and hash receipts.

Preparation entry points:

```text
scripts/preparations/m18_union/build_exact_sequence_union.py
scripts/preparations/m18_union/select_openfold_structures.py
scripts/preparations/m18_union/prepare_union_training.py extract|pack
scripts/preparations/complete/protein_features_v4.py
```

The preparation tools intentionally enforce this frozen 6,667-sequence contract.
They are not a general whole-RCSB dataset builder. Use `--help` for required input
paths. For extract/pack, retain the same `PREP_RUN_ID`; source fingerprint, selection
receipt, sequence/label hashes and train-only normalization are checked. Embedding
links must be dereferenced when copying the prepared bundle to another host.

## Completed Loss Matrix

All variants started from scratch at seed 49 for 50 epochs, with the same M18 architecture
and frozen inputs. A is a matched 50-epoch baseline; the older 30-epoch run is not used
as the budget-matched control. Batch 64, token budget 65,536, 8 workers, prefetch 4,
LR 8e-5, weight decay 0.1, 10% warmup then constant LR, AMP. No early stopping or per-epoch test.

| Run | Positive weight | Smoothing | Residue mass term | Coefficient | AP-best epoch | Val macroAP |
|---|---:|---:|---|---:|---:|---:|
| A | 3.0159315 |0.05| Ratio MSE |0.75|21|0.729028|
| B | 3.0159315 |0.05| Ratio MSE |3.00|27|0.726168|
| C |1|0.05| Ratio MSE |0.75|19|0.728832|
| D |1|0| Ratio MSE |0.75|20|0.730250|
| E |1|0| Bernoulli-rate KL |1.00|26|0.730004|

The mass ratio is the mean positive probability over valid residues; every exact
sequence has equal mass-loss weight. Rate KL uses log-mean probabilities and exact
target entropy limits for 0/1 rates. It is not a reproduction of AlphaGenome's
Poisson/multinomial loss. E's coefficient is not claimed to be gradient-scale matched.

`loss_matrix_completion.json` and `asset_index.json` identify jobs 797200_0--4 and
checkpoints. Each run completed 50 epochs with 50 checkpoints/validation predictions.

## Completed Auxiliary-Head Matrix

Base D is unweighted/unsmoothed CE plus 0.75 MSE(s,r). Add a separate pooled ratio
head with MSE(q,r), preserving the original residue mass constraint:

```text
s = mean valid residue probabilities
r = known-positive labels / known labels
q = sigmoid(Linear1(SiLU(Linear128(LayerNorm(mean/max h)))))
loss = CE + 0.75*MSE(s,r) + lambda_aux*MSE(q,r)
```

| Run | Head | Head-to-backbone gradient | Lambda aux | AP-best epoch | MacroAP | AUPRC | Residue ratioMAE | Head ratioMAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| R0 |none|none|0|20|0.728637|0.755418|0.089913|-|
| R1 |mean/max MLP|detached|0.75|14|0.730595|0.755295|0.101082|0.069271|
| R2 |mean/max MLP|joint|0.75|26|0.727868|0.752437|0.055597|0.068686|

All are validation-only seed-49 point estimates at each run's macroAP-best epoch.
The head did not show a clear ranking gain. R2 changes the calibration tradeoff,
but ratio improvement is not uniform: at epoch 50 R0/R2 ratioMAE is 0.058100/0.062439.
R2's independent head is not better than its residue mean at its AP-best epoch.
R1 cannot establish an auxiliary-supervision benefit because its h is detached.

The CPU detached control was bitwise equal, but full GPU trajectories diverged.
Each run had 2 AMP gradient-protection skipped steps; no nonfinite forward losses
were skipped. No statistical superiority or automatic model promotion is claimed.
Full matched-epoch results are in `auxiliary_head_results.json`.

## Metric and Selection Definitions

- `chain_ap_macro`: equal weight per exact sequence; all-negative AP=0.
- `pooled_ap`: sklearn average precision. Legacy `pr_auc` is an AP alias.
- `trapezoid_auprc`: trapezoidal area under pooled precision-recall curve.
- F1/MCC@0.5 uses literal 0.5, not the nearest searched threshold.
- Macro threshold sweeps use a shared gate, never each chain's oracle gate.
- Residue-mean ratioMAE and independent head ratioMAE are distinct outputs.
- Brier/NLL are unweighted probability metrics, separately macro and pooled.
- `chain_mean` epoch loss is chain-weighted; legacy batch-mean is retained separately.
- All epoch checkpoints/validation predictions, metric-specific best and Pareto
 references are retained. `best.pt` remains a macroAP compatibility alias, not a
 final balanced selection. The existing test set is already inspected, not sealed.

## Compatibility and Replay

Existing defaults remain compatible: ratio_loss_source=auto uses a supplied head
as the legacy mass target; no head means residue mean. Default head construction
retains its old RNG consumption and optimizer/clipping stays one global group.
New auxiliary supervision requires explicit ratio_loss_source=residue_mean.

The integrated replay configs explicitly set validation-only, no-overwrite, per-epoch
retention and residue mass. R0/R1/R2 also opt into separate head/backbone optimizer
and clipping groups, preserved constructor RNG and zero head dropout. Frozen original
source on HPC3 is not relabeled as this integration's historical training commit.

The replay configs target a NEW `ppc_repo_replay_20261104` output root. They have not
been retrained as part of this Git integration. They reuse the actual verified HPC3
inputs; edit paths for another machine and validate the full bundle before training.
Never point their output back at completed historical runs.

Example on HPC3, only when a new replay is explicitly desired:

```bash
export PPC_REPO_ROOT=/absolute/path/to/this/PPC
export PPC_PYTHON=/data/user/shuang886/.conda/envs/openfold/bin/python
export PPC_MATRIX=ratio-head
sbatch "$PPC_REPO_ROOT/configs/experiments/20261104_m18_union_of3/replay_hpc3.slurm"
# For A--E instead: PPC_MATRIX=loss and sbatch --array=0-4%4 ...
```

## Artifact Index

`asset_index.json` contains real HPC3 checkpoint paths/SHA-256 hashes, selected
epochs, metrics and per-residue validation prediction paths. No model/feature tensor
is committed. `input_contract.json`, `input_preflight.json`, and original source
hash maps preserve the frozen audit. These are data-integrity provenance, not a
claim that read access to split IDs is model training on test.

Only trusted owned checkpoints should be loaded with `weights_only=False`.
