# Validation Lock and Fixed Test Scoring Core

## Scope and Current Status

The user requires test evaluation only after a full-precision eligible
PDBTM validation pooled AP>=0.80. The best completed point is currently
0.7997048805648456. **No real test inference or test scoring has run.**

`scripts/analysis/locked_union_evaluation.py` implements the lock/scoring
core, not an end-to-end inference pipeline. A dedicated strict GPU inference
producer and a full validation replay remain prerequisites before test use.
Do not reuse the training script's relaxed warm-start or threshold-sweep path
for the locked test evaluation.

## Lock Contract

Before opening shared split metadata, checkpoint tensors or completion
receipts, recompute AP from the validation-only candidate NPZ and enforce
pooled AP>=0.80 and macro AP>=0.7417948466002535. After passing that gate:

- Verify all1,075 validation IDs,404,267 residues, sequence lengths/order,
  residue offsets and union-label hashes against the frozen mapping.
- Require50consecutive full-coverage validation epochs, no validation skips,
  no test training records, and successful terminal receipts for all members.
- Reproduce each source from its original saved validation predictions.
- Load every checkpoint using `strict=True`; reject missing/extra keys,
  incompatible shapes and nonfinite floating-point state tensors.
- Verify the exact declared single model or fixed equal-logit blend,
  epsilon1e-7, without alpha or membership search.
- Freeze checkpoint/model-state/validation hashes, epochs, configurations,
  cohort metadata and validation-only F1/MCC gates in a new exclusive lock.

The full shared cohort table is accessed only after the validation gate.
Recording test IDs at that point is not reading test prediction tensors or
raw labels, and does not make this historically inspected universe sealed.

## Fixed Scorer

The scorer verifies the lock SHA and immutable assets, requires exactly the
locked members and inference receipts, and verifies model-state invariance,
no gradients/optimizer steps, and validation reproduction before accepting
test prediction files. Predictions carry lock/member/checkpoint provenance.
All724 test queries must align exactly; no truncation or structure fallback
is permitted. Per-member labels and residue ordering must agree.

Metrics use the frozen validation F1 and MCC gates plus fixed0.50/0.60
defaults. AP/AUROC, macro AP, ratio diagnostics and ceil-based Top-L/20,
Top-L/10 and Top-L/5 are reported. No test-oracle threshold search is called.

Existing output paths, including dangling symlinks, are refused. The scorer
atomically reserves a new output directory before reading test data; failures
leave that directory for audit rather than silently repeating evaluation.
An incomplete directory is not a completed result. Any recovery requires
review of the failure and must retain the same frozen model/gates.

## Validation Performed

Six synthetic focused tests cover valid locks, strict loading, nonfinite and
out-of-range contracts, below-target closure, terminal failures, fixed gates,
ceil TopK, provenance/receipt requirements, overwrite/symlink refusal and
atomic output reservation. The full local suite passed91tests with2optional
snapshot tests skipped. Narrow read-only review used gpt-6-luna max and found
no remaining blockers after two boundary fixes.

The negative probe on the actual best **validation-only** NPZ must fail before
shared cohort metadata or any checkpoint/test tensor is opened. Its receipt
is stored alongside this document; no real test data is needed for that probe.

## Remaining Before Real Test

1. Reach0.80 at full precision under the existing eligibility rules.
2. Implement/review a dedicated no-optimizer GPU producer, with strict loading,
   model.eval/no-grad, state hashes and source/config/feature provenance.
3. Replay every member on validation and reproduce locked source predictions
   within predeclared numerical tolerances; reproduce the combined milestone.
4. Verify feature/normalization compatibility and full test-cohort readiness.
5. Freeze the final system, run fixed test inference/scoring, and retain its
   result regardless of whether it looks better or worse than validation.
6. Attempt whole-RCSB single-chain all-partner-union training, as specified in
   the milestone contract. Do not choose that configuration from test results.
