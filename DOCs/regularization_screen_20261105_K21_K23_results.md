# Completed Weight-Decay and Smoothing Controls

K21/K22/K23 completed50epochs with Slurm800313_2/3/4 COMPLETED exit0:0.
Every epoch retained all1,075 validation queries/404,267 residues without
validation skips. None evaluated test. Training recorded1/1/2AMP-skipped
updates and1/1/2sanitized gradient elements respectively.

| System | Epoch(s) | Macro AP | Pooled AP |
|---|---:|---:|---:|
| K21 weight decay0.03 | 27 | 0.756780 | 0.785465 |
| K22 weight decay0.20 | 25 | 0.754599 | 0.784199 |
| K23 label smoothing0.02 | 32 | **0.760476** | **0.790364** |
| G2+MSA_D+K21 | 27+13+27 | 0.766525 | 0.797343 |
| G2+MSA_D+K22 | 27+13+25 | 0.766189 | 0.796674 |
| G2+MSA_D+K23 | 27+13+32 | 0.768027 | 0.799379 |

K23 is the strongest completed single model by pooled AP, exceeding K3 by
0.00131057 at the selected validation points. Its fixed ensemble nevertheless
remains below G2+MSA_D+K18 pooled AP0.7997048805648456. No candidate here
reaches0.80. All ensemble weights and original G2/MSA_D checkpoints were fixed
before calculation;there was no alpha or extra-combination search.

These are adaptive selected-validation,single-seed results, not statistical
confirmation that smoothing improves the underlying population. K24's MSA
dropout arm remains running. The next controlled cells are described in
`preTCN_interactions_20261105.md`. See the resolved/source receipt and full
result JSON with the same filename stem for reproducibility.
