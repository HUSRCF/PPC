# Single-chain union OF3 architecture screen

Status: implementation and validation; launch IDs are recorded separately once submitted.

## Frozen contract

- Task: single-chain, exact-sequence ever-observed all-partner union, 5.5 A heavy-atom contact labels.
- Universe: 6,667 exact sequences; train 4,868 / validation 1,075 / test 724.
- Split, labels, ESM-C MLC layers 12/24/36, OF3 selected monomer and optimizer contract are unchanged.
- Each new run starts from scratch at seed49. No additional seed controls in this screen.
- D loss: chain-mean unweighted CE plus 0.75 residue-mean ratio MSE; no separate ratio head.
- Budget: 50 epochs, 12 hours, one allocated GPU per run, HPC3 acd_ue, array concurrency two.
- Validation-only; test loader and per-epoch test evaluation are disabled.
- Promotion considers validation chain-macro AP and pooled AP jointly. Ratio MAE is diagnostic.

## Matrix

| ID | Input change | Architecture change | Primary comparator |
|---|---|---|---|
| B0_fixed90 | Correct two descriptor channels | Existing TCN + ESM2 mean graph + global mean/max | Historical D49, same target |
| B1_fixed91q | B0 + absolute pLDDT/100 | Same as B0 | B0 |
| G0_of3_mean | Same as B1 | Replace contact edges with OF3 graph, same mean aggregator | B1 |
| G1_edge_mlp | Same as B1 | Query/source-conditioned geometry edge MLP | G0 and B1 |
| G2_edge_attention | Same as B1 | Sparse Q/K attention with geometry edge bias | G1 |
| G3_gvp | Same as B1 | Scalar/vector GVP-style message, invariant vector-norm readout | G1 |
| C0_full_attention | Same as B1 | Replace global mean/max broadcast with one full-attention block | B1 |
| C1_linear_full | Same as B1 | Positive-kernel linear attention plus one full-attention block | C0 |

All arms preserve two local MLP blocks and the two-layer TCN. G1--G3 replace, rather than stack on top of, the simple ESM2 graph. C0/C1 retain the ESM2 graph. No assembly pose, partner coordinates, template search, or new folding is introduced.

## Corrected descriptors

The original 89D scalar cache used raw index8 for a coordinate-origin-dependent XYZ spread and raw index51 for an RSE term whose alleged SASA input was actually surface curvature. B0 replaces them with:

1. Local centered C-alpha spread in a seven-residue window, normalized by chain maximum. This is named explicitly as a spread descriptor, not as a validated physical surface roughness.
2. Actual SASA divided by chain maximum, multiplied by the existing normalized B-factor term.

Original SASA[0] and Bnorm[49] are recovered by inverting the parent z score, which had no clipping. Float32 roundoff is recorded. All other 87 scalar channels and the availability flag are retained exactly. Only the two corrected columns are refit on training sequences. Absolute C-alpha atom pLDDT is bounded by division by100 and is not fitted on validation/test. It is not the all-atom global avg_plddt; these two averages differ legitimately.

## Geometry inputs

- Frozen maximum-confidence OF3 sample; exact residue sequence must equal the query.
- kNN16 within16 A, symmetrized, unioned with sequence neighbors at offsets one and two. No self edges.
- Eight covariant vector channels from backbone, sidechain, local chain directions and backbone-plane normal; missing vectors are zero.
- Thirty invariant edge channels:16 distance RBFs,4 sequence-separation descriptors,2 endpoint pLDDT values,6 local-frame direction components and2 frame-valid flags.
- Edge weight: exp(-(distance/10)^2) times sqrt(max(q_src,.05)*max(q_dst,.05)).
- GVP-style implementation uses pure PyTorch and scalar/vector channel mixing, not an official pretrained GVP checkpoint. Proper-rotation invariance/equivariance and translation-invariant edge preparation are tested.

## Acceptance gates

CPU unit tests cover rotations, padding, invalid edges, crop/edge-attribute alignment, finite gradients and branch updates. The formal feature receipt requires full-length coverage of all6,667 sequences, finite values, train-only normalization and no label reads. Each GPU arm then performs a real-payload two-update smoke, validates two chains, confirms the backbone and zero-initialized branch updated, and checks zero test samples before starting its independent50-epoch run.

This is a single-seed augmented-system screen. Added modules change parameter counts and constructor RNG consumption, so same seed does not imply parameter-matched initial backbones. No architecture or statistical superiority is claimed before results are available.
