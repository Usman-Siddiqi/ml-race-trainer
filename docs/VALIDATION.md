# Rebuild validation

No training, backpropagation, optimizer update, GPU tensor workload, or GPU benchmark was run.

## Checks performed

- CPU regression suite: 44 checks passed on Windows with Python 3.12.
- Ruff lint and formatting checks passed.
- Gymnasium's environment checker passed for every track.
- A fixed arithmetic steering controller physically finished all three tracks on CPU.
  This validates drivability and finish rules; it is not a learned policy.
- Raycasts were checked against the actual collision boundaries.
- Checkpoint round trips restored networks, replay, RNG, counters, and manually populated
  optimizer metadata without an optimizer update. A failed save preserved the old file.
- UI controls, startup, rendering, and stop behavior were checked offscreen. The monitor's
  training subprocess was replaced with a fake child in tests.
- Menu, setup, and driving screenshots were rendered on CPU and visually inspected.
- CUDA metadata reports PyTorch 2.11.0+cu128 and NVIDIA GeForce RTX 4050 Laptop GPU.
  Windows reported external power, so CPU remains the explicit project default.

## Preserved original state

Recovered local edits are committed at `c940c81`. The old checkpoint remains ignored by Git:

```text
best_race_model.pth
SHA256 F7FFE7DB207283656068BFF3B2E8E1D5EE9EC284ABAAB31F3AE8B9A6530F5539
Architecture 7 -> 128 -> 128 -> 128 -> 9
```

Only its tensor names/shapes were read, with `weights_only=True` and `map_location="cpu"`.
It was not changed or treated as a version 2 policy.

## Intentionally unverified

End-to-end training, convergence, learned lap times, GPU kernel execution, and CPU/GPU
throughput comparisons remain unverified because training was deferred. CUDA availability
confirms device discovery; it is not an execution benchmark. The CI workflow is committed
locally and has not been run on hosted runners.
