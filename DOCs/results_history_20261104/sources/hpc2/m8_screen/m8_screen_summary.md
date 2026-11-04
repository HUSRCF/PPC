# M8 Validation-Only Screen

- Baseline: `M7c`
- Test visibility: hidden; checkpoints are selected only by validation chain-macro AP.
- Advance gate: macro AP delta >= `0.005` and pooled AP delta >= `-0.003`.

| Variant | Epoch | Macro AP | Delta | Pooled AP | Delta | Raw ratio MAE | Advance |
|---|---:|---:|---:|---:|---:|---:|:---:|
| M7c | 20 | 0.739431 | - | 0.753759 | - | None | control |
| M8d | 22 | 0.742653 | +0.003223 | 0.757350 | +0.003591 | 0.109789 | False |
| M8a | 20 | 0.739577 | +0.000147 | 0.761357 | +0.007598 | 0.075661 | False |
| M8e | 20 | 0.739384 | -0.000046 | 0.757965 | +0.004206 | 0.072695 | False |
| M8c | 15 | 0.737561 | -0.001869 | 0.760627 | +0.006867 | 0.121674 | False |
| M8b | 22 | 0.736752 | -0.002679 | 0.749055 | -0.004705 | 0.144457 | False |
