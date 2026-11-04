# Gated DeltaNet dependency preparation

Status: offline dependency dry run passed; nothing installed and no environment cloned. Await explicit user approval before installing anything.

## Environment and frozen components

Source environment: /data/user/shuang886/.conda/envs/openfold. Proposed clone: /data/user/shuang886/.conda/envs/ppc_deltanet.

Keep torch2.7.1+cu128, torchvision0.22.1+cu128, torchaudio2.7.1+cu128, Triton3.3.1, NumPy2.1.2, SciPy1.18.1, scikit-learn1.9.1 and pandas2.3.1 unchanged. These are pinned in constraints_hpc3.txt and none is in the resolver's planned installation list. The original fold/openfold environments and all existing flash_attn packages must remain untouched.

## Assets and dry run

HPC3 could not query PyPI directly, so the fixed wheels were downloaded on the local machine and transferred offline. Wheelhouse: /data/user/shuang886/PPC_M18/dependency_plans/deltanet_20261105/wheels,30files totaling41603854bytes. Includes flash-linear-attention0.6.0, fla-core0.6.0 and Transformers5.6.2 plus dependencies. No optional compiled causal-conv1d or flash-attn package was requested.

Dry run used the existing environment's pip25.2 with --dry-run --no-index --find-links, explicit CUDA extras, the eight frozen constraints and JSON --report. It proposes18 new packages plus click8.1.7->8.5.0, only in the future clone. The precise19-package list, prior versions, wheel hashes and raw-report SHA-256 are in dry_run_summary.json. The eight core versions were rechecked unchanged; no package installation was performed.

Official resources: [FLA package metadata](https://pypi.org/project/flash-linear-attention/0.6.0/), [implementation](https://github.com/fla-org/flash-linear-attention), [GatedDeltaNet layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py). These establish the published package/operator, not PPIS efficacy or actual kernel compatibility on this environment.

## After approval only

Clone with copy semantics and offline package reuse, preserving the original environment. Install from the prepared offline wheels using the exact19-package install_lock.txt plus constraints_hpc3.txt. requirements.txt records the two top-level requirements. Verify the eight frozen versions, run pip check, and compare source/clone inventories. Do not solve dependency conflicts by upgrading Torch or Triton.

Before any formal experiment, use an allocated H100 to validate import, ragged/prefix masks, independent state at chain boundaries, valid-only reverse order, noncausal forward/backward combination, finite mixed-precision outputs/gradients and agreement with a small numerical reference. Do not share language-model inference caches between proteins. A successful dependency resolver is not a successful numerical/runtime acceptance.

Initial architecture proposal: one bidirectional Gated DeltaNet after MSA fusion and before geometry propagation. Keep G2 as the spatial path. Start with a narrow direction width, measure actual parameter count and add a matching capacity control. No DeltaNet training has been submitted yet.
