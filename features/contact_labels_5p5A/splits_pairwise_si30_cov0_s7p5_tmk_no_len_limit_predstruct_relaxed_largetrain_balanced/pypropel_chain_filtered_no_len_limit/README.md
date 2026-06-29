# PyPropel Chain-Filtered Split

This directory keeps the existing PDB-level pairwise-SI split and adds
chain-level allowlists from the PyPropel/TMKit no-length-limit QC dataset.

Use `*_chain_ids.txt` or `*_chain_manifest.csv` for residue-level benchmarks.
The original `train_ids.txt`, `val_ids.txt`, and `test_ids.txt` remain PDB ids.

See `chain_filter_summary.json` for counts and the strict pairwise SI audit.
