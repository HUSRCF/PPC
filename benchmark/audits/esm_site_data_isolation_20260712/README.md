# ESM Site Data and Gradient Isolation Audit

- Overall status: **PASS**
- Config: `/media/990Pro/ProtBind/PPC/configs/experiments/20260712_m8_screen_hpc2/train_m8d_tcn2_ratio01_seed42.yaml`
- Training script: `/media/990Pro/ProtBind/PPC/scripts/training/train_esm_site.py`
- Loaded chain counts: `{'train': 14052, 'val': 3796, 'test': 2392}`
- Test loaded by training config: `False`

## Split Isolation

- `seq_id` cross-split overlap: train-val=0, train-test=0, val-test=0
- `pdb_id` cross-split overlap: train-val=0, train-test=0, val-test=0
- `component_id` cross-split overlap: train-val=0, train-test=0, val-test=0
- `seq_sha1` cross-split overlap: train-val=0, train-test=0, val-test=0
- `exact_group_id` cross-split overlap: train-val=0, train-test=0, val-test=0
- `representative_id` cross-split overlap: train-val=0, train-test=0, val-test=0

## Resolved File Isolation

- `source_esm`: missing=0; cross-split resolved-path overlaps=0
- `labels`: missing=0; cross-split resolved-path overlaps=0
- `sequence_features`: missing=0; cross-split resolved-path overlaps=0
- `primary_embeddings`: missing=0; cross-split resolved-path overlaps=0
- `contact_graphs`: missing=0; cross-split resolved-path overlaps=0

## Gradient Isolation

- `_evaluate` calls `model.eval()`: `True`
- `_evaluate` runs under `torch.no_grad()`: `True`
- `_evaluate` contains backward: `False`
- `_evaluate` contains optimizer/scaler step: `False`
- Runtime logits/parameter-version assertions present: `True`

## SI30 Evidence

- Unique exact sequences: `6667`
- Recalled cross-split pairs: `252542`
- SI >= 0.30 violations: `0`
- Maximum recalled cross-split global SI: `0.2826855123674912`
- Scope limitation: Zero violations proves zero only among MMseqs2-recalled candidates; it is a high-sensitivity audit, not a mathematical exhaustive-all-pairs proof.

## Failures

- None.
