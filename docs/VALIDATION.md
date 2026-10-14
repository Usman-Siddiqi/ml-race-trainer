# Validation record

## Completion audit and GPU verification - 2026-10-13

The planned implementation is complete in the original project checkout. This audit checked
the current modules, entry points, runtime, and tests against the rebuild scope:

| Planned change | Implemented behavior and evidence |
| --- | --- |
| Consistent track geometry | Walls, raycasts, collision, projection, and directed gates share cached geometry. All three layouts finish with a fixed arithmetic controller. |
| Correct episode endings | Ordered gates, circuit laps, sprint finish, crash termination, and time/stall truncation have regression coverage. |
| Better driving controls | Braking/coasting/steering and bicycle-style dynamics; 13 normalized observations and nine discrete actions. |
| One learning pipeline | Double DQN, Huber loss, gradient clipping, step-based exploration, balanced tracks, and fixed-seed evaluation. |
| Explicit GPU support | CPU default; explicit CUDA and power-aware auto. Actual CUDA inference and optimizer updates now pass. |
| Recoverable learning state | Atomic versioned checkpoints include networks, optimizer, replay, RNG, counters, and configuration. CPU/CUDA restore and partial stop/resume now pass with real updates. |
| Unified UI and commands | Manual driving, replay, explicit training setup, responsive monitor, and compatibility adapters. A real bounded CUDA child process now completes through the desktop monitor. |
| Reproducible installation and checks | Locked dependencies, runtime outside OneDrive, Windows launcher, CPU CI workflow, documentation, and optional GPU tests. |

### Results

- Default regression suite: **45 passed, 6 GPU checks skipped**, in 9.00 seconds.
- Explicit CUDA suite: **6 passed**, in 18.10 seconds.
- Ruff lint, formatting, `git diff --check`, and `uv lock --check` passed.
- Machine: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM, driver 617.42;
  Python 3.12.12, PyTorch 2.11.0+cu128. Windows reported external power.
- CPU and CUDA outputs/actions agreed. The restored CUDA optimizer's next update matched
  the uninterrupted CUDA update exactly; its CPU counterpart agreed within float tolerances.
- Target synchronization, finite loss/gradients, checkpoint policies, all-track greedy
  evaluation, and actual optimizer moments on CUDA were verified.
- An optimizer restore defect was found and fixed: Adam's CPU step counters could alias
  between agents restored from the same checkpoint. Each restore now owns independent state.
- All learning runs used disposable pytest directories. Integration coverage totals only
  **216 simulation steps and 51 optimizer updates**, across CLI resume, partial stop/resume,
  and the desktop subprocess; the remaining agent tests use small synthetic batches.
- Peak PyTorch tensor allocation recorded in the parent GPU tests was about 19 MiB. This
  excludes the CUDA context, other applications, and the desktop child process; it is not
  total system VRAM usage or a performance benchmark.

Reproduce the checks from the project root with the installed runtime:

```powershell
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -m pytest -q
& $racePython -m pytest -q -m gpu --run-gpu --junitxml=artifacts\gpu-validation.xml
& $racePython -m ruff check .
& $racePython -m ruff format --check .
uv lock --check
```

Local machine-readable results are in the ignored `artifacts/cpu-validation.xml`,
`artifacts/gpu-validation.xml`, and `artifacts/validation-summary.json` files.

### Remaining limits

Full training, convergence, learned lap times, and CPU/GPU throughput comparisons remain
unmeasured. Passing CUDA tests establishes that the learning pipeline executes correctly;
it does not establish that a policy learns to finish or that GPU execution is faster.
The CI workflow remains locally committed and has not been run on hosted runners.

## Initial rebuild validation - 2026-09-21

During the initial rebuild, no training, backpropagation, optimizer update, GPU tensor
workload, or GPU benchmark was run. The record below describes that earlier validation.

### Checks performed

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

### Preserved original state

Recovered local edits are committed at `c940c81`. The old checkpoint remains ignored by Git:

```text
best_race_model.pth
SHA256 F7FFE7DB207283656068BFF3B2E8E1D5EE9EC284ABAAB31F3AE8B9A6530F5539
Architecture 7 -> 128 -> 128 -> 128 -> 9
```

Only its tensor names/shapes were read, with `weights_only=True` and `map_location="cpu"`.
It was not changed or treated as a version 2 policy.

### Initially unverified

End-to-end training, convergence, learned lap times, GPU kernel execution, and CPU/GPU
throughput comparisons were unverified at that point because training was deferred.
The 2026-10-13 checks above now cover bounded learning and GPU kernel execution.
