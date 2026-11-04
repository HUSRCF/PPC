# DeltaNet feasibility for single-chain PPIS

Status: literature/dependency assessment only. No DeltaNet package was installed and no DeltaNet training was submitted.

Update2026-11-05: [Offline0.6.0 dependency dry run passed](deltanet_20261105/README.md);19 proposed dependency changes, eight core packages preserved, installation still awaits explicit approval. This does not yet establish import/kernel compatibility.

## Sources and applicability

- Yang et al., [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), NeurIPS2024: delta-rule associative memory plus hardware-efficient sequence-parallel training.
- Yang, Kautz and Hatamizadeh, [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), ICLR2025: learned forgetting and targeted delta updates; [official implementation](https://github.com/NVlabs/GatedDeltaNet).
- [FLA GatedDeltaNet layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py) provides chunk-mode training, padding/unpadding and cu_seqlens support. It is a causal language-model component by default, not an already validated bidirectional PPIS model.

These results concern language modeling/retrieval, not evidence of improved membrane-protein PPIS AP. Engineering feasibility is good; predictive gain is unproven.

## Proposed adaptation

Prefer a one-layer bidirectional Gated DeltaNet as a post-fusion context arm, not a replacement for the successful G2 geometry graph.

1. Keep the same frozen ESM-C/MSA/structure inputs, split, D loss and50-epoch contract.
2. Insert after the MSA residual and before contact/geometry propagation, matching the BiGRU placement.
3. Use independent forward/backward states and direction-specific processing, reverse only each valid chain, and restore residue order before combining to256D. Do not reverse a padded whole tensor naively.
4. Reset all memory/short-convolution states at every chain boundary and every batch; no persistent inference cache. Correct cu_seqlens must prevent cross-protein contamination.
5. Retain the original causal short convolution inside each directional branch, unless a separately labeled ablation justifies removing it. The outer TCN is not automatically equivalent to these q/k/v local convolutions.
6. Pre-normalize, use output dropout0.1 and initial fixed residual scale0.1. Measure the actual added parameters and add a matching MLP control rather than assuming parity with the296960-parameter BiGRU module.
7. Validate long mixed-precision forward/backward and FP32 state/accumulation boundaries explicitly; the failed C1 ELU-kernel implementation is not evidence against delta-rule memory.

The delta rule writes a prediction error into a matrix-valued memory instead of simply accumulating key-value outer products. Gating also controls forgetting. With fixed head dimensions it supports linear scaling in sequence length, but that does not guarantee a speed/AP gain at the sequence lengths and batch shapes used here.

## Trial order

| Candidate | Main purpose |
|---|---|
| Existing BiGRU/MLP/fusion-position screen | Establish whether post-fusion sequence processing helps at all |
| B1+MSA and G2+MSA | Complete the previously proposed common-feature geometry x MSA2x2; strongest current empirical priority |
| Old90D+MSA+Bi-GatedDeltaNet | Compare to the same-feature BiGRU/MLP control without mixing G2 changes |
| G2+MSA+Bi-GatedDeltaNet | Test geometry/sequence-memory complementarity only after the G2+MSA control exists |

Do not immediately stack several memory and full-attention blocks. A one-layer controlled arm is easier to interpret. Initial language-model evidence supports trying the operator, not presuming superiority over GRU or G2.

## Existing HPC3 environment

Read-only audit of /data/user/shuang886/.conda/envs/openfold/bin/python:

- PyTorch2.7.1+cu128 and Triton3.3.1 exist.
- FLA, einops, transformers and causal_conv1d are absent.
- Current [FLA pyproject](https://github.com/fla-org/flash-linear-attention/blob/main/pyproject.toml) declares Python>=3.10; its CUDA extra requires torch>=2.7 and triton>=3.3. These two version minima are satisfied, but this is not an end-to-end compatibility test.
- The full package also declares transformers>=5.6.2,<6 and einops. Optional convolution backends may have separate requirements; do not assume a compiled causal_conv1d extension is mandatory for every supported backend.

Do not install latest dependencies into fold/openfold or replace Torch/Triton. A future implementation should clone the existing environment, pin an audited FLA version/commit, run an installation dry run and obtain approval before adding dependencies. A tiny plain-PyTorch recurrence can serve as a numerical reference, but formal training should use the proven parallel kernel rather than a Python loop over residues.

## Acceptance and interpretation

Required tests: full residue coverage/order; prefix padding invariance; independent state across chains; valid-only reversal; gradient isolation during evaluation; finite long-sequence AMP; matched initial backbone RNG; actual throughput/VRAM. Rank candidates using validation macro AP and pooled AP jointly; ratio MAE remains diagnostic and test stays untouched.
