# Whole-RCSB Single-Chain Pilot Execution Contract

## Relation to the PDBTM Target

The PDBTM validation pooled-AP target is already met by the frozen
G2+MSA_D+K26 equal-logit ensemble (0.8005468731). The fixed test readout is
0.7693981218 pooled AP. Its members,epochs,weights and validation gates remain
unchanged. Remaining PDBTM validation-only jobs cannot replace this system
using test feedback,and no additional PDBTM test run is authorized here.

This next stage implements the user's request to attempt whole-RCSB
single-chain union training. It does not return to pairwise prediction or
introduce P-L datasets. The ESM-C-only pilot is a first whole-data training
and throughput baseline,not the final transfer of the complete geometry/MSA
system. Those input additions remain separate subsequent work.

## Native Split Decision

The run uses the existing native RCSB ge50 split without protecting old
PDBTM holdouts:201013train/25127val/25127test exact observed sequences.
This default follows the explicit whole-RCSB scope;no reply was received to
the optional protection-preference question. It is not represented as an
explicit user selection of that option,and no test score informed the choice.
The alternative would remove most of the native training pool.

Old PDBTM test results stay archived as pre-RCSB results. A model trained on
the native RCSB training pool cannot claim independent generalization on
that old PDBTM test. Conversely,a PDBTM-supervised checkpoint cannot
initialize this run because old PDBTM train overlaps native RCSB val/test.
The downstream network therefore starts from scratch.

Native RCSB val/test are held out from this run's optimizer. They are not
automatically a newly sealed research holdout:historical research cohorts
overlap parts of this universe. Future test claims must disclose that
history and the detected-SI30/cov80 plus same-PDB component contract;this
preparation does not supply an exhaustive global Biopython SI certificate.
The pilot does not read or evaluate native RCSB test at all.

## Frozen Pilot

| Item | Contract |
|---|---|
| Input | Frozen ESM-C600M layers12/24/36,final pretrained LayerNorm,3456D |
| Labels | Actual observed-sequence 5.5A heavy-atom biological-assembly all-partner ever-observed binary union |
| Model | 2379010parameters;256D projection,two gated-depthwise TCN layers,local MLP/global pooling/classifier |
| Missing inputs | No structure,contact or MSA branch;none fabricated |
| Initialization | Scratch downstream network;no checkpoint or smoke-weight reuse |
| Loss | Unweighted chain-mean CE,no smoothing,ratio/auxiliary weights0 |
| Budget | Seed49,3full epochs,no cropping,no train/val subsampling |
| Selection | Native validation pooled AP;macro AP and probability diagnostics retained |
| Loader | Batch64,token budget65536,8workers,prefetch4,pinned/persistent workers |
| Scheduler | One acd_u GPU,12CPUs,128GiB,12h;no array or extra seed controls |
| Test | No loader,no threshold search,no test-driven model selection |

Configuration is in
`configs/experiments/20261105_rcsb_single_union/R0_esmc_only_pilot_seed49.json`.
This three-epoch pilot is not a full-convergence result or an isolated
dataset-expansion ablation. Its AP must not be numerically compared to the
different PDBTM validation cohort as proof of data-expansion benefit.

## Readiness and One-Time Continuation

`submit_rcsb_union_pilot.py` waits on the already launched content-audit
producer. It never allocates a GPU just to wait for data. A failure marker,
missing/wrong producer handle,missing full acceptance,modified source/config,
wrong cohort or unresolved input-integrity issue prevents submission.
An observation timeout does not cancel the producer or declare it failed.

Before submission,it rechecks full accepted files,split IDs,original union
assignments,model/source hashes and the pinned real H100 smoke acceptance.
All compact metadata/label payload hashes and their filesystem identities
are audited. Feature values remain references to the original immutable
bank rather than a second approximately1.03TB copy. A submission-attempt
record is written before calling the scheduler;ambiguous acceptance requires
manual reconciliation,never automatic duplicate submission.

The scheduled GPU wrapper again verifies the frozen plan,code,input receipt,
source/reference inventory and compact payload identities. Approved CUDA
Torch is mandatory;there is no silent CPU fallback in the wrapper. It checks
runtime train/val counts,no-test settings,full validation coverage and
gradient isolation,then rechecks inputs/source after successful training.
Completion is not inferred merely from job submission or a checkpoint file.

## Cross-Node Filesystem Probe

A read-only CPU step within the existing800559_1allocation measured one
identical feature on ACD-Login-3 and ACD1-21:

| Field | Login | GPU Node |
|---|---:|---:|
| Device | 73 | 49 |
| Inode | 11011613304 | 11011613304 |
| Bytes | 2879262 | 2879262 |
| mtime_ns | 1792187872546403000 | 1792187872546403000 |
| ctime_ns | 1792187872546404000 | 1792187872546404000 |

Device numbers are client-local. The controller compares all identity fields
on the recorded login host;on other nodes,it excludes only device and keeps
global inode,size,mtime,ctime and resolved-link checks. Content SHA-256 was
audited when reading the payload. The source bank and compact inputs must
remain immutable throughout use. This is not a hostile-root-proof snapshot.

Controller review:gpt-6-luna max PASS within that filesystem/immutability
contract. Six focused tests cover one-time submission,no-test/scratch/full
cohort guards,timeout without producer cancellation,original native split
checks,changed payloads/plans,and cross-node device-number handling. The
reviewer also exercised a mocked GPU lifecycle with one trainer invocation.
Full local suite:142passed,2optional snapshot tests skipped.

## Remaining Full-Model Inputs

Existing assets include224276raw Protenix feature records retaining C-alpha
and all-atom coordinates,and154876validated local HPC3 CIFs. A local CIF
sample has absolute confidence in B_iso (45.14--97.11),rather than assuming
the chain-normalized descriptor is absolute pLDDT.

The unfilled structure requests remain26991sequences:26906short chains and
85longer historical failures. Of the short inputs,347contain ambiguous or
unknown letters;their exact sequence/structure representation needs explicit
QC rather than silent residue replacement. The original production queue
filters at100aa and must not be reused unchanged for the short-chain repair.
Whole-RCSB MSA coverage is not yet established. None of these gaps are declared
complete by the ESM-C-only pilot.
