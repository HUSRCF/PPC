# PPC: Single-Chain PPIS Prediction

PPC contains the single-chain M18 residue-level protein-protein interaction-site
model and its benchmark tools. The separate partner-conditioned/pairwise project
is [TMINTER3](https://github.com/sunnylab-cb/TMINTER3); it is not this model.

## Current Union/OpenFold3 Recipe

```text
Frozen ESM-C 600M layers 12/24/36 -> 3456D -> 256D ---+
OpenFold3 descriptors89D + availability -> 90D -> 256D +-> sum
Chain-index embedding + relative-position encoding -+
  -> 2 residual MLP blocks
  -> 2 gated-depthwise TCN blocks (kernel7, dilation1/2)
  -> 1 weighted message-passing block on frozen ESM2 contact edges
  -> mean/max global context broadcast + LayerNorm
  -> residue classifier256->512->2 -> positive softmax probability
```

ESM2 supplies contact edges, not another residue embedding branch. ESM-C and
structure prediction are offline/frozen. No partner identity, partner structure or
experimental assembly pose is input. The optional pooled ratio head remains an
ablation, not an adopted model release.

## Entry Points

| Purpose | Location |
|---|---|
| Model | `src/ppcbind/models/esm_site.py` |
| Training and eval-only prediction | `scripts/training/train_esm_site.py` |
| Validation metrics/loss accounting | `src/ppcbind/site_metrics.py` |
| Exact-sequence union and OF3 input preparation | `scripts/preparations/m18_union/` |
| Completed experiment replay configurations | `configs/experiments/20261104_m18_union_of3/` |
| Methods, results and artifact index | `DOCs/m18_union_of3_20261104/README.md` |
| CPU regression tests | `tests/` |

The latest dataset has 6,667 exact observed-structure sequences: train 4,868, val 1,075,
test 724. Labels are 5.5 A heavy-atom all-partner contacts, OR-merged across retained
contexts of an identical full sequence. Original SI30 split membership is retained.
This is not a per-chain binary classification task or a pairwise interface task.

## Tests

Use an existing verified environment; do not reinstall PyTorch/vision/audio.
Training/metrics use PyTorch, NumPy, scikit-learn, SciPy and PyYAML. Union preparation
also needs Biopython; structure selection needs Gemmi; complete features need
mkdssp and the existing preparation modules.

```bash
export PYTHONPATH="$PWD/src"
/path/to/verified/python -m unittest discover -s tests -p 'test_*.py'
```

An optional trusted-HPC3 acceptance test is enabled by `PPC_M18_SNAPSHOT_ROOT`;
it strict-loads original checkpoints and checks CPU forward/export equivalence.
It does not run training or score the test set.

## Version and Asset Policy

This integration starts from Star's committed PPC version 6ad6f95 and preserves its
22 commits after the previously published master. Existing local/Star dirty files
are not imported or overwritten. Historical experiment source remains frozen on
HPC3; the integrated commit is not misidentified as the historical training commit.

Weights, raw feature tensors, predicted structures and per-residue tensor outputs
stay outside Git. `DOCs/m18_union_of3_20261104/asset_index.json` records their actual
paths, checkpoint SHA-256 hashes, selected epochs and validation metrics. Original
M18 checkpoints/results remain unchanged. No new training or model promotion is
performed by checking out this repository.
