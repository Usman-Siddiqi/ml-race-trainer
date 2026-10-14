# Development and troubleshooting

Read [HANDOFF.md](HANDOFF.md) for the current state and [REFERENCE.md](REFERENCE.md) for
the contracts. This guide describes how to extend the project without duplicating its
simulator or learning pipeline.

## Make a change

1. Inspect Git status and the affected source. Keep any user changes and local history.
2. Use the installed Python 3.12 environment or recreate it from the lockfile.
3. Change the package module responsible for the behavior; extend the unified CLI/UI only
   when the feature needs another way to access that behavior.
4. Check the relevant contracts: seeds, observations/actions, endings, checkpoints, and logs.
5. Run meaningful checks for the affected subsystem, then update the corresponding docs.
6. Commit significant changes separately with a message describing the resulting behavior.

Documentation changes can be checked for valid local links, current defaults, and consistent
commands without rerunning GPU learning. Do not treat a fixed-controller lap or a smoke test
as evidence that a learned policy is good.

## Test map

| File | Coverage |
| --- | --- |
| [test_environment.py](../tests/test_environment.py) | Gymnasium contracts/seeds, raycast/collision agreement, directed ordered gates, endings, all-track drivability, invalid physics/actions |
| [test_agent.py](../tests/test_agent.py) | Double DQN target arithmetic, environment-step epsilon, immutable inference, replay wraparound/RNG restoration |
| [test_checkpoint.py](../tests/test_checkpoint.py) | Safe checkpoint round trips, independent optimizer restores, atomic failure protection, legacy rejection, resume dry-run validation |
| [test_device_and_evaluation.py](../tests/test_device_and_evaluation.py) | Power-aware selection, explicit CUDA errors, deterministic evaluation and truthful summaries |
| [test_cli_and_ui.py](../tests/test_cli_and_ui.py) | Menu-only startup, dry-run/no-agent guarantee, invalid arguments, rendering, navigation, mocked monitor lifecycle, viewer close |
| [test_gpu.py](../tests/test_gpu.py) | Actual CPU/CUDA agreement, gradients/updates, target copies, trained state restore, CLI resume, partial stop/resume, real desktop child |
| [conftest.py](../tests/conftest.py) | CPU protections, offscreen SDL, explicit GPU opt-in, allocation reporting |

From the project root:

```powershell
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -m pytest -q tests/test_environment.py
& $racePython -m pytest -q
& $racePython -m ruff check .
& $racePython -m ruff format --check .
uv lock --check
```

The default suite sets `CUDA_VISIBLE_DEVICES` to an empty string before collection and
blocks backward/Adam steps. Its CPU checkpoint fixtures populate optimizer metadata
without training. Keep those checks inexpensive and independent of GPU hardware.

Explicit CUDA checks are a separate invocation:

```powershell
& $racePython -m pytest -q -m gpu --run-gpu --junitxml=artifacts\gpu-validation.xml
```

GPU tests require a usable CUDA build and fail if CUDA is unavailable. They use small
synthetic batches and bounded temporary runs; do not increase them to production training
length. A new learning test must carry the `gpu` marker and respect those bounds.
The desktop child test caps its subprocess configuration and has a finite wait deadline.
JUnit records parent-process peak PyTorch allocation; it does not measure total VRAM or
child-process allocations. CI runs the CPU path on Windows and Linux, not CUDA.

## Add a track

Implement a layout through `Track` in `tracks.py`; define its centerline, half-width,
closed/open rule, and gate count. Add its name to `TRACK_NAMES` and its UI label to
`TRACK_LABELS`. Decide whether it belongs in the default training rotation.

The offset polygons assume consistent circuit winding, nonzero segments, and usable bend
angles. Avoid duplicate consecutive points, reversals, and self-intersecting offsets.
Keep spawn and finish clear of walls. An open sprint must end inside the road so the car
can cross the final gate before hitting the end boundary. Consider the current S-specific
lap handling when extending sprint semantics to another track name.

Use the same Track methods for rendering, sensors, collision, projection, and crossings.
Extend contracts/seed, raycast agreement, gate order, and fixed-controller drivability
coverage. Inspect an offscreen preview, then manually drive when useful. Do not introduce
a rendering-only road definition.

## Change observations, controls, or physics

Preserve the indexed input and action definitions in REFERENCE.md. If their meaning or
shape changes, update `OBSERVATION_VERSION`/`ACTION_VERSION`, network dimensions, replay
arrays, environment spaces, compatibility handling, and tests together.

Substantial physics/reward changes can invalidate comparisons with old policies even when
tensor shapes match. Decide whether the checkpoint contract needs a new version and state
the compatibility decision explicitly. Keep the finite-input checks and bounded movement.

The environment's continuous action pair and the agent's discrete action index are distinct.
Only terminal states suppress bootstrap. The checkpoint counter in info is cumulative
across laps, while the current gate resets per lap; preserve these meanings in reports.

## Change learning or persistence

Keep `run_training` as the sole coordinator. Inference/evaluation must disable exploration
and avoid consuming the agent's exploration RNG. Target networks are separate, have no
gradients, and synchronize by optimizer updates. Epsilon advances by environment steps.

For a new hyperparameter, update its dataclass, validation, serialization, CLI exposure
if wanted, and documentation. Ensure the saved checkpoint reconstructs the original
configuration; CLI resume currently restricts overrides. Check older checkpoints when
introducing required fields or changing defaults.

Preserve atomic file replacement and weights-only loading. Test actual next-update
behavior after restore in the bounded GPU suite, not just matching saved weight tensors.
Optimizer state must not share mutable tensors with another loaded agent. Keep replay
capacity/cursor and RNG restoration consistent after ring-buffer wraparound.

## Change the desktop app

Use the existing shared viewer/renderer for manual driving and model replay. Keep startup
and setup free of training side effects. Launch learning as the existing subprocess and
consume JSON stdout through the queue; never put learning on the UI event thread.

Test stop requests issued before the run directory exists, close-while-busy behavior,
process failures, and completion events. Stop is cooperative; the window can remain open
until the child saves. Update telemetry consumers when changing log fields. Offscreen tests
use dummy SDL, while real interactive driving still needs a normal display.

## Investigate training results

When full training is requested, retain a run's configuration, seeds, and logs. Look at
per-track completion/crash rates and checkpoint fraction alongside mean reward. A higher
reward does not by itself show reliable finishes. The best checkpoint ranking prioritizes
completion, then checkpoint progress, then reward. Report finish time only for successes.

Compare CPU/CUDA with identical settings, sufficient warm-up, and synchronized timing.
Profile simulation, transfer, inference, and optimization separately before changing the
network or adding parallel environments. Greedy single-state inference and host reads can
limit GPU gains for this small model. Current results do not establish arbitrary-track
generalization or a GPU speedup.

## Troubleshooting

| Symptom | Check / response |
| --- | --- |
| Import missing or wrong Python version | Use the LocalAppData interpreter; rerun `setup.ps1` with Python 3.12/locked dependencies. Global Python is independent. |
| Import resolves a different checkout | Print `race_trainer.__file__`; recreate the editable install through setup after moving folders. |
| OneDrive says cloud provider unavailable | Restore sync/local availability and choose Always keep on this device. Preserve local edits before using another checkout; do not assume placeholders contain committed source. |
| CUDA unavailable | Run doctor, verify the NVIDIA GPU is enabled and `nvidia-smi` sees it, and install the CUDA extra. A CPU PyTorch wheel cannot execute CUDA. |
| CUDA tests fail but doctor sees the GPU | Check for inherited `CUDA_VISIBLE_DEVICES` restrictions and run the explicit GPU test invocation. Doctor confirms discovery, not kernel execution. |
| Auto chooses CPU | Battery/unknown power or unavailable CUDA is expected. Windows can count USB power as external; choose the device explicitly when needed. |
| Backpropagation blocked in a test | This is the CPU fixture's protection. Put bounded CUDA learning coverage under the GPU marker and opt-in flag. |
| Legacy model error | The original seven-input model cannot be used with v2 observations/actions/physics. Preserve it; a v2 model requires new training. |
| Resume rejects arguments | Only episodes/device/CPU threads/output/rendering may override saved settings. The episode target must exceed the completed count. |
| Output directory exists | Choose a new path; every training/resume run gets a separate directory. |
| Stop seems delayed | The marker is checked every 32 environment steps, and evaluation/checkpoint writes are not interruptible through that marker. Wait for the stopped event and final save. |
| Runtime failure has no final checkpoint | Arbitrary exceptions do not follow the cooperative stop path; use the latest periodic checkpoint. Save/evaluation intervals determine its age. |
| Blank/noninteractive display | Do not use dummy SDL for manual play. Tests intentionally use offscreen surfaces. |
| Pygame package conflict | Use pygame-ce; do not install both distributions that provide the same `pygame` import. |
| Remote clone lacks the rebuild or models | Source commits are still local; ignored model/run files also require separate transfer. Use the original checkout for continuity. |

## Update the record

Record the date, exact commands, device/runtime, results, and workload bounds in
[VALIDATION.md](VALIDATION.md) after meaningful verification. Distinguish implementation
checks from convergence, lap quality, and speed measurements. Refresh HANDOFF.md after
significant progress so the next editor can continue without chat history.
