# AP0.80 Milestone and Post-Milestone Contract

User instruction: after PDBTM validation pooled AP reaches0.80, evaluate test
without using test to select a model, then attempt RCSB whole-set training.

## 1. Validation Target

An eligible candidate must complete its declared50epochs successfully, retain
all1,075 validation queries /404,267 residues without skipped validation batches,
pass source/label/score provenance checks, and satisfy the existing same-epoch
macro AP guardrail >=0.7417948466002535. Pooled AP must be >=0.80 at full numeric
precision, not merely round to0.80.

Candidates include the already-declared single models and fixed ensembles.
Model/checkpoint selection uses validation only. Ensemble membership/weights
must follow the policy recorded before calculation; no test-derived change is
permitted. Every displayed metric comes from the same selected checkpoint or
the same frozen ensemble output.

## 2. Lock Before Test

Once an eligible candidate reaches the target, lock the validation-selected
system before reading its test outputs. Record:

- Architecture, feature/normalization contracts and code versions.
- Exact checkpoint epochs and SHA-256 hashes; strict state-dict compatibility.
- Ensemble membership, weights and clipped-logit epsilon where applicable.
- Validation prediction hashes, AP values, and the validation-selected F1/MCC
  thresholds. Fixed0.50/0.60 thresholds may be reported separately.
- Frozen union-v2 test cohort and residue mapping; no new length cutoff,
  structure fallback or label changes.

The current campaign uses724 exact-sequence union test samples, not the old
2,392 context-specific test-chain benchmark. Counts/mappings must be audited
at the actual evaluation stage. This historically inspected universe must not
be called a newly sealed independent holdout.

Evaluate the locked system in eval/no-gradient mode. Report full coverage,
macro AP, pooled AP, AUROC, F1/MCC at validation-frozen gates and fixed defaults,
Top-L metrics, and diagnostic calibration/ratio errors. Do not search a test
threshold grid or use test-oracle F1/MCC to select anything. A disappointing
test result must still be retained and reported.

Test outcomes cannot alter checkpoints, ensemble weights, gates, feature
normalization, architecture or hyperparameters for this locked result. Other
already-running trials can finish, but they remain validation analyses rather
than replacements chosen to improve this test table. Any later evaluation
requires a separately timestamped validation lock, not a test-guided rerun.

## 3. Then Attempt Whole-RCSB Training

Continue the current **single-chain / all-partner-union PPIS** task. Do not
switch to pairwise prediction or the PDBbind/BioLiP/COACH420/Holo4k ligand
datasets. Inspect the current RCSB assets before assuming compatibility:

- Observed-structure sequence and insertion-code mapping; heavy-atom5.5A,
  biological-assembly all-partner labels; exact-sequence union construction.
- SI-controlled splits and overlap with the PDBTM evaluation cohort. Do not
  train on evaluation sequences/components and then claim an independent
  comparison on those same examples.
- ESM-C layer/dimension compatibility, predicted-monomer structures, corrected
  geometry descriptors, confidence-field semantics and train-only normalization.
- Direct MSA Transformer feature coverage. If absent at whole-set scale,
  explicitly prepare it or use a documented geometry/ESM-C pretraining stage
  followed by MSA adaptation; do not silently zero/fabricate required features.

Choose the transferred architecture/configuration from validation evidence and
asset readiness, never from the just-observed test ranking. RCSB preparation
and training use independent output directories and preserve shared raw data.
The test evaluation and an actual RCSB training attempt remain required work
after the AP milestone; reaching0.80 alone does not close the full user request.
