# Completed MSA-Dropout Control

K24 completed50epochs in1h28m05s, Slurm800313_5 COMPLETED exit0:0.
All epochs retained1075validation queries/404267residues without skips or
test evaluation. Training recorded1AMP-skipped update and3sanitized gradient
elements. Only elementwise MSA projection/adapter dropout changed to0.10.

Pooled-selected epoch21 gives macro AP0.7579491948842205 and pooled
AP0.7894735820148974. Its predeclared G2e27+MSA_De13+K24e21 equal-thirds
ensemble gives macro AP0.7664211725701363/pooled AP0.7987994557072695.
It does not improve the completed leader0.7997048805648456 or reach0.80.

The six-arm regularization screen is now complete. The strongest single is
still K23(label smoothing0.02), pooled AP0.7903642797571038;the strongest
completed fixed ensemble remains G2+MSA_D+K18. No weights or additional
combinations were searched, and test remains closed. See same-stem resolved
and result JSON for source hashes, full metrics and completed-run evidence.
