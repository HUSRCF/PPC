# Controlled Ablation Result-to-Claim Gate

## Verdict

- `claim_supported`: partial
- `confidence`: high
- Contract: global pairwise DeepTMInter SI30 chain-filtered split, seeds 46--50,
  equal-five logit ensemble, validation-only threshold selection, and one frozen
  test evaluation.
- Alignment audit: PASS for 3,796 validation chains, 2,392 test chains,
  1,036,256 test residues, and 43 test SI30 components.

## Supported

1. The full M18 model materially outperforms the no-ESMFold variant. A minus B
   is positive for chain-macro AP (`+0.0727`, 95% CI `[+0.0535,+0.0955]`), F1
   (`+0.0691`, `[+0.0433,+0.0901]`), and MCC (`+0.0917`,
   `[+0.0600,+0.1166]`).
2. The full model materially outperforms ESMC-only. A minus C is positive for
   chain-macro AP (`+0.0702`, `[+0.0517,+0.0930]`), F1 (`+0.0530`,
   `[+0.0355,+0.0707]`), and MCC (`+0.0879`, `[+0.0634,+0.1120]`).
3. The dominant supported gain is associated with the ESMFold-derived 90D
   residue channel. B minus C does not exclude zero on any tested metric.
4. The TCN has a smaller supported contribution: A minus E excludes zero for
   chain-macro AP (`+0.0138`, `[+0.0075,+0.0210]`) and MCC (`+0.0159`,
   `[+0.0049,+0.0268]`), but not for F1 or ratio MAE.

## Not Supported

1. A reliable contribution from the frozen ESM2 contact prior is not detected.
   A minus D and B minus C both cross zero for all four bootstrap metrics.
2. A reliable ratio-MAE improvement is not detected for any planned pairwise
   comparison; every ratio-MAE interval crosses zero.
3. This single 43-component test split does not establish broad cross-dataset
   generalization.
4. AUPRC and AUROC are point estimates in this analysis; the component
   bootstrap covers chain AP, frozen-threshold F1/MCC, and raw ratio MAE.

## Paper-Safe Wording

> Under the matched SI30 chain-filtered split, seeds 46--50, equal-five logit
> ensemble, and validation-only threshold selection, the full M18 model
> outperformed the no-ESMFold and ESMC-only variants. The strongest and most
> reliable contribution came from the ESMFold-derived 90D residue channel,
> while the TCN provided a smaller gain. We did not detect a reliable benefit
> from the frozen ESM2 contact prior in this setting.

## Exporter Equivalence Audit

The prediction-export reruns used the same locked checkpoints as the prior
scalar evaluations. For D, all audited critical scalar metrics were exactly
equal. For B, the maximum absolute difference across validation/test AUPRC,
AUROC, chain AP, selected-gate metrics, and ratio metrics was `2.4e-7`; all five
validation-selected gates were identical. This is floating-point-level drift
and does not alter any operating-point or conclusion.
