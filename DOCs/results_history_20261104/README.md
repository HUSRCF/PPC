# Recovered Single-Chain Results History

This directory repairs a documentation/asset-index omission in the 2026-11-04
PPC integration. It does not retrain, rescore, change checkpoint selection or
promote a new model. Start at [the main results index](../RESULTS_INDEX.md).

## Coverage

1. M0-M6 historical test progression, original M18 final five-seed result and
   M21C equal-three result, copied from the local paper's frozen result bundle.
2. M7 paired-seed confirmation and M8 validation-only screen, recovered from HPC2.
3. M9-M21: 55 training runs plus 7 eval-only exports, recovered from existing
   Star config.json/metrics.jsonl files. All 62 selected records have 3,796
   validation chains; all 55 training runs reached their configured epoch budget
   in the retained logs. This is a log fact, not a current scheduler audit.
4. M14's offline conditional-gain/stacking summary, which is separate from its
   two frozen-residual training runs in the raw run catalog.
5. August matched-five-seed controlled feature ablation, with uncertainty and
   alignment evidence.
6. November union/OpenFold3 is already in [its own directory](../m18_union_of3_20261104/README.md).

The catalog covers retained artifacts found at collection time. It is not a claim
that every attempted/failed experiment on every machine has been recovered.

## Contracts

The historical split is the chain-filtered global pairwise-SI30 split named
`splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708`.
"Pairwise" here names the sequence-identity calculation; these results are still
**single-chain all-partner-union PPIS**, not partner-conditioned pairwise PPIS.
The original context-specific labels are not the later across-context exact-sequence
union. Historical chain contexts total 14,052 train / 3,796 val / 2,392 test.

No UniProt/PDBTM/TmAlphaFold bridge is introduced. These historical experiments use
the exact-sequence ESMFold feature channel, while the November experiments use OF3.
No partner/assembly pose is supplied as a model input.

## Metric Semantics

- Historical trainer `pr_auc` is sklearn average precision. It is labeled pooled
  AP in the run catalog, not trapezoidal PR area.
- Downstream frozen M18/M21C summaries explicitly separate AP and trapezoidal
  AUPRC. Preserve the distinction even when the values are numerically close.
- F1/MCC in the historical run catalog use the same per-record validation-F1 gate.
  They are validation summaries, not held-out claims.
- The selected record follows the configured validation metric, with the old
  f1 fallback if a metric is unavailable. Ties select the first record.
- For M21C head-only calibration, selection is negative head MAE. Do not substitute
  the epoch with maximum residue AP.
- Eval-only epoch 0 is an export convention; it is not the source checkpoint epoch.
- Residue-mean ratio MAE and independent ratio-head MAE are separate quantities.
- Archived reports may retain original historical wording. The current interpretation,
  validation/test distinction and uncertainty caveats are in the main index.

## Provenance and Reuse

- [source_manifest.json](source_manifest.json) lists the imported source paths,
  host and SHA-256 receipts. Copied source files are retained verbatim.
- [star_m9_m21_runs.json](star_m9_m21_runs.json) contains compact configs/model
  settings, original config/metrics file hashes, selected and last records, and
  source paths for every recovered run. It is an export, not the full raw log.
- [star_m9_m21_validation.csv](star_m9_m21_validation.csv) and
  [STAR_RUNS.md](STAR_RUNS.md) are views derived only from that JSON.
- [M18 checkpoint index](sources/m18/CHECKPOINTS.tsv) retains five checkpoint
  locations, epochs, weights and frozen SHA-256 values. These are external paths,
  not repository downloads.
- Historical runtime source commits and the current integration commit are not
  interchangeable. The results index does not claim that today's trainer generated
  every historical record.
- Weights, structures, embedding/feature tensors and per-residue binary predictions
  remain outside Git. No heavy data or environment change is needed to read this bundle.
