# Validation record

## Version 3: training, unseen-track results, and speed - 2026-11-10

Version 3 was built, trained, and measured on 2026-11-09 and 2026-11-10. Unlike the earlier
records below, this one includes real training and real driving results.

Machine: Ryzen 7 8845HS (8 cores/16 threads), 16 GiB RAM, NVIDIA GeForce RTX 4050 Laptop
GPU (6141 MiB, driver 621.01), Windows 11, on external power. Python 3.12.12, PyTorch
2.11.0+cu128, NumPy 2.5.3, Gymnasium 1.3.0, pygame-ce 2.5.8.

### What "unseen" means here

Training draws only from `random:0` to `random:511`. Results below are on the test set:
the three hand-built layouts plus `random:2000000000` onward. None of these are used for
training or for choosing `best.pt`, which uses a separate validation set
(`random:1000000000` onward). A held-out run is one greedy episode from the start line
with a seeded start offset. The large test is 203 tracks with 2 start variations: 406 runs.

### Default run

`python -m race_trainer train --output runs\v3-default`, all defaults, seed 42, with the
final code. Auto selected CUDA.

| Measure | Result |
| --- | --- |
| Training time | 1,000,000 steps in 1.9 minutes, including 10 evaluations and saves |
| Training speed | Median 9,494 steps/s over 50 progress windows (6,327 to 10,016) |
| Validation (16 unseen tracks, scored during training) | 88% finished at 100,000 steps; 100% at every evaluation from 200,000 to 1,000,000 |
| Default test (`evaluate`: 3 hand-built + 20 generated) | 23 of 23 runs finished |
| Large test (`evaluate --generated 200 --episodes 2`) | 406 of 406 runs finished, for `best.pt` and for the final weights |
| Hand-built layouts, never trained on | Oval, S-curve sprint, and rounded square all finished |
| Pace on the large test | Mean speed 157 units/s (top speed 180); mean lap 9.8 s |

An earlier run of the same configuration the same day took 2.6 minutes (median 6,784
steps/s), so expect two to three minutes depending on what else the laptop is doing.

A trained car brakes for corners rather than holding one speed: on one test circuit its
speed ranged from 108 to 180 units/s, with full throttle on 52% of steps, coasting on
39%, and braking on 9%. That lap came from an earlier model of the same design.

CPU-only: `train --device cpu --steps 400000` ran at a median 3,815 steps/s and took 1.8
minutes. Its model also finished 406 of 406 runs on the large test, at a lower mean speed
of 142 units/s; it had 40% of the default step budget.


### Seeds and the two fixes that came from measuring

Final weights after 1,000,000 steps, scored on the 406 held-out runs. Each row is one
training run; rows differ only in seed and the setting named.

| Training setting | Seeds | Runs finished per seed | Total |
| --- | --- | --- | --- |
| Constant learning rate, no steering noise | 42 | 305 | 305 / 406 (75.1%) |
| Annealed learning rate, no steering noise | 42, 1, 2 | 404, 406, 392 | 1,202 / 1,218 (98.7%) |
| Annealed, steering noise 0.1 | 42, 1, 2, 3 | 406, 405, 405, 405 | 1,621 / 1,624 (99.8%) |
| Annealed, steering noise 0.2 (default) | 42, 1, 2, 3, 4 | 406, 406, 406, 406, 402 | 2,026 / 2,030 (99.8%) |

Mean speed on the test set was between 155 and 161 units/s in every run, against a top
speed of 180. Steering noise did not cost pace.

What these comparisons do and do not show:

- **Learning-rate annealing.** The constant-rate run finished 100% of its 16 validation
  tracks at every evaluation from 200,000 to 900,000 steps, then 62% at the final one, and
  its final weights finished 305 of 406 held-out runs while its `best.pt` finished 405. No
  annealed run ended that way, but this is one constant-rate run at this length, so it
  shows the failure exists rather than how often it happens.
- **Steering noise.** Without it one seed in three was clearly weaker (392). With it, nine
  runs across two noise levels each finished at least 402. The two levels are not
  distinguishable on this evidence; 0.2 is the default.
- **Crash penalty.** Raising the crash penalty from 5 to 20 at a constant learning rate
  made validation results less steady (as low as 75% of 48 tracks late in training), so
  it was not adopted.
- **`best.pt` selection.** Ranking tied checkpoints by speed chose, for one seed, a model
  from 50,048 steps that finished 391 of 406 held-out runs while that run's final weights
  finished 406. Selection now keeps the newest checkpoint among those that finish the most
  validation tracks. With the default settings above, `best.pt` and the final weights were
  the same checkpoint in every run.

An untrained network finishes 0 of 43 runs on the default test set: the car does not move.

### Speed

`python -m race_trainer benchmark`, measured with nothing else training:

| Measurement | Result |
| --- | --- |
| Simulator alone, 1 car | 1,826 steps/s |
| Simulator alone, 64 cars | 12,237 steps/s |
| Simulator alone, 128 cars | 11,319 steps/s |
| Learner update, CUDA, batch 512 / 2,048 / 4,096 | 3.7 / 3.7 / 4.4 ms |
| Learner update, CPU (8 threads), batch 512 / 2,048 / 4,096 | 5.8 / 15.9 / 29.8 ms |
| Training loop, short test, `--device cuda` | 9,392 steps/s |
| Training loop, short test, `--device cpu` | 3,589 steps/s |

The short test overstates a long run, which slows as replay outgrows the CPU cache and
also stops to evaluate and save; see the full-run figures above.

How the speed was reached, in order, each measured on this machine:

1. One car per Python step ran at about 2,900 steps/s. Sixty-four cars in the batched
   simulator ran at about 5,700, limited by memory traffic in the raycast.
2. Processing the raycast in cache-sized slices and reusing temporaries: about 10,000.
3. With learning on, the first GPU learner (batch 512, replay on the GPU, actions chosen
   on the GPU) trained at only about 1,400 steps/s. Each update took about 9 ms whatever
   the batch size, because each CUDA call costs tens of microseconds here.
4. CPU replay sampling, one packed transfer per update, a combined forward pass, fused
   Adam, and choosing actions on a CPU copy of the network: 5.4 ms per update, flat up to
   2,048 transitions. With batch 2,048 every 64 steps, training ran at about 5,000 steps/s.
5. Running each update on a worker thread while the simulator advances: about 7,000 to
   9,500, the range of the two full runs above.

Two identical 40,000-step CUDA runs produced identical network weights and replay contents.

### Checks

- CPU suite: **112 passed, 9 GPU checks skipped**. It covers the generator, geometry,
  one-car/many-car equality, n-step replay, checkpoints, the headless loop with learning
  switched off, the CLI, and the app with a fake child process.
- CUDA suite (`-m gpu --run-gpu`): **9 passed**. Tiny disposable runs of a few hundred
  steps: updates, target and driving-copy synchronization, fitting a predictable reward,
  CPU/CUDA checkpoint restore, CLI train and resume, stop and resume, repeatability with
  the worker thread, a real desktop training subprocess, and the benchmark command.
- Ruff lint and format checks passed; `uv lock --check` passed.
- GitHub Actions ran the CPU checks on Windows and Linux for the pushed commits.

Reproduce:

```powershell
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -m pytest -q
& $racePython -m pytest -q -m gpu --run-gpu
& $racePython -m ruff check .
& $racePython -m race_trainer train --output runs\v3-default
& $racePython -m race_trainer evaluate --model runs\v3-default\best.pt --generated 200 --episodes 2
& $racePython -m race_trainer benchmark
```

### Limits of this evidence

- Generalization was measured on new layouts from the same generator and on the three
  hand-built layouts. Hairpins that double back, crossings, and roads narrower than a
  half-width of 28 are untested.
- All full-length results are from CUDA training on one machine. CPU-only training was
  measured once, at 400,000 steps.
- Seeds differ in outcome. The table above is the evidence; a single run is not.
- Laps beyond one, resumed runs of full length, and the desktop app's training monitor at
  full length were not measured. Resume and the monitor are covered by bounded tests.
- No comparison against a hand-written or optimal racing line was made, so "fast" is
  relative to the car's top speed, not to the best possible lap.

## Completion audit and GPU verification (version 2) - 2026-10-13

This and the following record describe version 2, which version 3 replaced. They are kept
as history. Version 2's observation included centerline guidance, and no full training run
was ever measured for it.

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
- An optimizer restore defect was found and fixed: Adam's CPU step counters could alias
  between agents restored from the same checkpoint. Each restore now owns independent state.
- All learning runs used disposable pytest directories. Integration coverage totalled only
  216 simulation steps and 51 optimizer updates.

Full training, convergence, learned lap times, and CPU/GPU throughput comparisons were
unmeasured at that point.

## Initial rebuild validation (version 2) - 2026-09-21

During the initial rebuild, no training, backpropagation, optimizer update, GPU tensor
workload, or GPU benchmark was run.

- CPU regression suite: 44 checks passed on Windows with Python 3.12; Ruff checks passed.
- Gymnasium's environment checker passed for every track.
- A fixed arithmetic steering controller physically finished all three tracks on CPU.
- Raycasts were checked against the actual collision boundaries.
- Checkpoint round trips, UI controls, startup, rendering, and stop behavior were checked
  offscreen, with the training subprocess replaced by a fake child.

### Preserved original state

Recovered local edits are committed at `c940c81`. The old checkpoint remains ignored by Git:

```text
best_race_model.pth
SHA256 F7FFE7DB207283656068BFF3B2E8E1D5EE9EC284ABAAB31F3AE8B9A6530F5539
Architecture 7 -> 128 -> 128 -> 128 -> 9
```

Only its tensor names/shapes were read, with `weights_only=True` and `map_location="cpu"`.
It was not changed, and no later version treats it as a usable policy.
