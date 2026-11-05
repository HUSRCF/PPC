# RCSB Structure-Branch Readiness

This preparation continues the whole-RCSB single-chain union stage. It does
not change the frozen PDBTM target/test system,read RCSB test predictions,
or claim that an ESM-C-only pilot already transfers the complete M18 model.

## Actual Feature Probe

The existing `tminter3_esmc` environment on HPC3 can import the frozen PPC
complete-feature implementation without installing anything or changing
Torch. It provides BioPython,Biotite,SciPy,Gemmi and pyDSSP.

Four previously accepted Protenix structures were converted to temporary
PDBs and processed on CPU. Query lengths99,50,68 and54 were all retained
exactly,including the68all-X and54partial-X queries. Outputs have62D
physicochemical,89D raw scalar and8x3vector features. All tensors were finite.
Respectively92,50,68and54residues had positive SASA. Secondary-structure
outputs contained non-coil assignments rather than a uniform fallback.

The actual SASA function uses `Bio.PDB.SASA.ShrakeRupley`;the module's
missing-FreeSASA warning is misleading for that implementation. FreeSASA
was not used or installed. Exact source hashes and the four query checks
are archived in `RCSB_structure_readiness_20261105/feature_probe.json`.

This is a four-query functionality/alignment probe,not full-bank acceptance
or biological validation of X-rich predictions. The probe used pyDSSP's
three-state assignments mapped into the eight-channel interface. The
existing production extractor required mkdssp. Those backends are not
declared numerically equivalent;formal short-feature extraction must
explicitly resolve this backend contract rather than silently mix them.

## Existing Inputs And Missing Work

| Scope | Count | Meaning |
|---|---:|---|
| Native ge50 sequences |251267|Full single-chain sequence universe|
| Existing raw89D structure feature records |224276|Existing source bank;not re-folded|
| Local HPC3 CIF index |154876|Source candidates already local|
| Existing-feature sequences without local indexed CIF |69400|Need source resolution/transfer for absolute confidence and coordinate QC|
| New50--99aa folding requests |26906|Two queued acd_u tasks800847_[0-1]|
| Historical longer folding failures |85|Tracked separately;not counted as repaired|

The local indexed CIFs are all within the existing-feature scope. The69400
nonlocal candidates split into55287train,6968val,and7145testqueries.
This is an input-location inventory,not test evaluation or a reason to
change the native split. Six production ledgers on Star account for69134
of these UIDs. The other266must be checked against the two retained-source
indices;they are not automatically new folding failures.

## Required Acceptance Before Geometry Training

1. Keep a frozen candidate-source table and hashes of native queries,
   missing requests,local index,production ledgers and retained indices.
   Preserve multiple candidates instead of choosing one by model score.
2. Transfer only Star candidates absent from the local source index. A path
   in a ledger is not proof that its file exists or has correct contents.
3. Check the exact unchanged query,C-alpha coverage,finite coordinates,
   confidence scale,and agreement with coordinates used for the cached
   raw descriptors. The same sequence alone does not prove the same pose.
4. Correct descriptor columns8and51 using the declared geometry recipe;
   fit any new normalization only on the actual native training partition.
   Add absolute confidence and geometry edges only from validated monomers.
5. Keep85long failures and any newly discovered invalid inputs explicit.
   Do not fabricate required structure/MSA features or silently reduce the
   ESM-C-only full-universe pilot to the old feature-ready subset.

No package installation,normalization fit,full geometry-training launch,
or full CIF-content acceptance is established by this document.
