# Full RCSB Single-Chain Union Label Build

This is the post-target RCSB preparation stage. The PDBTM validation target
and fixed test readout remain frozen; neither is used to choose RCSB labels,
architectures, thresholds, or data exclusions.

## Exact Universe

- Original observed-sequence snapshot: 271397 exact sequences, all lengths.
- Native ge50 universe: 251267 exact sequences and 74646793 observed residues.
- Existing split: 201013 train / 25127 val / 25127 test.
- Input chain map: the original full contact-participating snapshot, including
  partners below 50 residues, not the old feature-ready or pairwise-ready table.
- The builder requires the native universe to equal every raw snapshot
  sequence of length >=50, with recomputed sequence SHA-256 and unique IDs.

This is a snapshot of contact-participating protein sequences, not every
current RCSB sequence. Contact-free contexts would add zero masks and cannot
change a binary OR; context counts do not estimate all experimental contexts.

## Label Contract

The original tag contract is
`observed-polymer-heavy-atom-5p5-assembly-v2`, first coordinate model.
Original pairs were generated within biological assemblies using protein
heavy atoms and a 5.5 Angstrom contact criterion. No threshold or assembly
definition is changed here. No ligand label or external P-L dataset is used.

For each original chain instance, take the union over all its partners.
Then take the union over all instances with exactly the same observed
sequence. Sequence positions remain the original observed-residue indices;
insertion-coded residues remain present, and internal coordinate gaps are
not filled with an expected polymer or UniProt sequence.

The builder checks the original residue-contact list against both stored
positive masks. It rejects duplicate or out-of-range indices, duplicate
pairs, cross-assembly pairs, source/chain-map mismatches and snapshot drift.
Short or feature-missing partners still contribute to a qualified query.
All-positive query unions are retained and reported, not silently removed.

## Outputs and Readiness

Independent root:
`/data/user/shuang886/PPC_RCSB_single_union/20261105/label_build/`.

| Artifact | Meaning |
|---|---|
| `labels.npy` | Flat uint8 binary labels, CSR-style concatenation |
| `queries.tsv` | Exact sequence, hashes, offset, length, positive count, original split/component |
| `contexts.tsv` | Original PDB, assembly, chain instance and per-context mask hash |
| `source_maps.tsv` | Observed-residue mapping hash, insertion/gap counts, source tag hash |
| `entry_audit.tsv` | Read source path, compressed file hash and raw pair/model counts |
| `summary.json` | Only a successful full build can publish this completion receipt |
| `smoke_summary.json` | Limited-entry smoke receipt; never formal-ready |
| `failure.json` | Failed build diagnostics; partial artifacts are not ready data |

Source tables are hashed before and after processing. Each raw tag's inode,
size and timestamp are checked around reading; its compressed content hash
is archived. No raw CIF, original tag, existing split or feature bank is
modified. A fresh output directory is required, rather than overwriting a
previous result. Full-dataset materialization includes held-out labels as
dataset creation only; it does not expose them to training or model selection.

## Execution and Verification

New implementation:
`scripts/preparations/m18_union/build_rcsb_sequence_union.py`.
Focused tests cover short partners, different assembly roles of an identical
sequence, insertion-coded residues, binary union, inverse provenance,
corrupt masks, duplicate mappings, full-universe enforcement, smoke isolation
and serial/parallel byte-identical outputs.

Final local verification: 10 focused tests passed; full suite 130 passed,
2 optional snapshot tests skipped. A narrow gpt-6-luna max review found three
engineering issues, now fixed and regression-tested: input symlink pinning,
atomic completion publication despite logging errors, and bounded ordered
parallel submission. Re-review verdict PASS, with no new concrete issue.
Full-scale peak memory was not exercised by these tests. Real-input smoke
and full output audit remain separate acceptance stages.

The ACD scheduler rejects CPU-only submissions without a GPU request. This
CPU label build must not consume an otherwise unused GPU to bypass that
policy. The user has allowed direct CPU processing when resources are
available; check load/memory, use a modest 16 workers with low priority and
single-threaded numerical libraries, and retain a PID/log/launch receipt.

Feature completeness, train-only normalization, loader acceptance and an
actual RCSB model training attempt are still required. This label build
alone is not a completed training experiment.
