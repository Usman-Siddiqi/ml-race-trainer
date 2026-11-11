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
as evidence that a learned policy is good; use `evaluate` on unseen tracks for that.

## Test map

| File | Coverage |
| --- | --- |
| [test_tracks.py](../tests/test_tracks.py) | Generator reproducibility, variety, corner limits, canvas bounds; seed ranges; raycast/collision agreement; a mixed batch equals each track alone; fixed-controller drivability of built-in and generated layouts |
| [test_environment.py](../tests/test_environment.py) | Gymnasium contracts/seeds, sensors-only observation, ordered directed gates, endings, rewards, grip limit, many cars equal one car exactly, random start positions |
| [test_agent.py](../tests/test_agent.py) | Double DQN targets, action table, exploration bias/hold/floors, n-step returns across endings, ring wrap, and resume breaks, replay restore, update schedule |
| [test_checkpoint.py](../tests/test_checkpoint.py) | Checkpoint round trips, independent optimizer restores, atomic failure protection, legacy and version 2 rejection, resume dry-run validation |
| [test_device_and_evaluation.py](../tests/test_device_and_evaluation.py) | Power-aware selection, explicit CUDA errors, deterministic evaluation, batched evaluation equals driving each episode alone |
| [test_training.py](../tests/test_training.py) | Vector environment restarts and true successors; the headless loop, logs, evaluation, stop, and resume with learning switched off; CLI train then evaluate; bounded simulator benchmark |
| [test_cli_and_ui.py](../tests/test_cli_and_ui.py) | Menu-only startup, dry-run/no-agent guarantee, default device, invalid arguments, rendering, navigation, mocked monitor lifecycle, viewer track stepping |
| [test_gpu.py](../tests/test_gpu.py) | CUDA learner and its CPU driving copy, gradients/updates, target and actor synchronization, fitting a predictable reward, trained state restore, CLI resume, stop/resume, real desktop child, benchmark |
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
blocks backward/Adam steps. The headless loop is still exercised on CPU by setting the
warm-up longer than the run, so no update is ever due. Keep those checks inexpensive and
independent of GPU hardware.

Explicit CUDA checks are a separate invocation:

```powershell
& $racePython -m pytest -q -m gpu --run-gpu --junitxml=artifacts\gpu-validation.xml
```

GPU tests require a usable CUDA build and fail if CUDA is unavailable. They use small
synthetic batches and bounded temporary runs of a few hundred steps; do not grow them into
real training. A new learning test must carry the `gpu` marker and respect those bounds.
CI runs the CPU path on Windows and Linux, not CUDA.

## Add or change tracks

Generated tracks come from `generate_track` in `tracks.py`. To change the kind of layouts
produced, edit `_random_loop` (shape) or `_loop_is_drivable` (what is rejected), then bump
`GENERATOR_VERSION`: every seed then produces a new track, so old evaluation numbers no
longer compare. Render a contact sheet of a few dozen seeds and look at it before trusting
a change, and keep the drivability test passing.

A hand-built layout goes in `_builtin` and `TRACK_NAMES`, with a label in `Track.label`.
Decide whether it belongs in default training; the current three are held out on purpose.

Rules any layout must satisfy:

- Corner radius comfortably above the half-width, or the inner boundary folds over itself.
- No two stretches close enough for their roads to overlap. Projection onto the centerline
  assumes the nearest segment is the one the car is on.
- Circuits may wind either way; the larger offset loop is taken as the outside.
- A sprint needs room for its run-off: the finish is `SPRINT_RUNOFF` before the end wall.

Use the same `Track` data for rendering, sensors, collision, projection, and crossings. Do
not introduce a rendering-only road definition or a second geometry routine: add batched
functions next to the existing ones and let `Track` call them with a batch of one.

## Change observations, controls, or physics

The inputs and actions are indexed in REFERENCE.md. If their meaning or shape changes,
update `OBSERVATION_VERSION`/`ACTION_VERSION`, the network, replay arrays, environment
spaces, compatibility handling, and tests together.

Keep the observation to things a car could measure. Anything derived from the centerline,
gates, or lap progress lets the policy follow the map instead of the road, which is what
version 3 removed. The observation test moves that hidden state and expects no change.

Substantial physics or reward changes invalidate comparisons with old policies even when
tensor shapes match; reward and physics settings are saved in every checkpoint so a model
is always evaluated under the rules it trained with. Keep the finite-input checks and the
bounded movement per step.

All physics lives in `RaceSim.step`. It advances every car at once and freezes finished
ones, so write changes as array operations and keep the one-car/many-car equality test.

## Change learning or persistence

Keep `run_training` as the sole coordinator. Evaluation must stay greedy and must not
consume the agent's exploration RNG. The target network has no gradients and synchronizes
by update count. Epsilon advances by environment steps.

Replay is time-ordered per car because n-step returns are built at sample time. Anything
that breaks that order needs a break marker, as resume does with `mark_break`.

For a new hyperparameter, update its dataclass, validation, CLI exposure if wanted, and
documentation. A new required field makes older version 3 checkpoints fail to load their
configuration, so either give it a default or bump `FORMAT_VERSION`.

Preserve atomic file replacement and weights-only loading. Test actual next-update behavior
after restore in the GPU suite, not just matching weights. Optimizer state must not share
mutable tensors with another loaded agent; keep the Adam alias regression test.

## Change the desktop app

Use the shared viewer/renderer for manual driving and model replay. Keep startup and setup
free of training side effects. Launch learning as the existing subprocess with `--json`
and consume its events through the queue; never put learning on the UI thread.

Test stop requests issued before the run directory exists, close-while-busy behavior,
process failures, and completion events. Update the monitor when changing event fields.
Offscreen tests use dummy SDL; real interactive driving needs a normal display.

The README GIFs in `docs/media` come from `tools/readme_gifs.py`, which drives a saved
model through the same simulator and renderer offscreen. It needs Pillow, which is not a
project dependency; install it to a separate folder rather than into the environment:

```powershell
uv pip install --target $env:TEMP\race-pillow --python $racePython pillow
$env:PYTHONPATH = "$env:TEMP\race-pillow"
& $racePython tools\readme_gifs.py --model runs\v3-default\best.pt
```

Regenerate them after visible changes to rendering or a new default model.

## Make it faster

Run `benchmark` first; it measures the simulator alone, the learner alone, and both
together. The two run concurrently, so training speed is set by the slower one plus the
work that stays on the main thread: choosing actions, storing steps, and drawing batches.
On the recorded machine the simulator is the slower one.

- **Simulator:** time is mostly the raycast, then the on-road test. Both scale with the
  number of wall segments per track, so coarser centerline spacing is the cheapest lever.
  Array temporaries that outgrow the CPU cache slow it sharply; that is why the raycast
  works in slices.
- **Learner on CUDA:** cost per update is dominated by the number of CUDA calls, not the
  batch size, up to a few thousand transitions. Prefer fewer calls and larger batches.
- **Learner on CPU:** cost grows with batch size; more threads help large batches.

Changing the batch size or update schedule changes learning, not just speed. Re-check on
unseen tracks, over more than one seed, before adopting a faster setting.

## Investigate training results

Keep a run's directory: it has the configuration, seeds, and logs. Read the evaluation
lines, not the progress lines, to judge the policy: progress lines describe exploring cars
starting at random positions, which crash far more than the greedy policy does.

`evaluate` with `--generated 200 --episodes 2` gives a much tighter estimate than the 16
validation tracks scored during training. Check `seen_in_training` before calling a result
generalization. Report finish time only for finishes.

Compare settings with identical seeds and step budgets, and repeat across seeds: single
runs of DQN vary.

## Troubleshooting

| Symptom | Check / response |
| --- | --- |
| Import missing or wrong Python version | Use the LocalAppData interpreter; rerun `setup.ps1` with Python 3.12/locked dependencies. Global Python is independent. |
| Import resolves a different checkout | Print `race_trainer.__file__`; recreate the editable install through setup after moving folders. |
| CUDA unavailable | Run doctor, verify the NVIDIA GPU is enabled and `nvidia-smi` sees it, and install the CUDA extra. A CPU PyTorch wheel cannot execute CUDA. |
| Auto chooses CPU | Battery/unknown power or unavailable CUDA is expected. Choose `--device cuda` explicitly when needed. |
| Training is slower than expected | Run `benchmark`. Laptops throttle on battery and when hot; other programs using the GPU also slow the learner. |
| Progress lines show mostly crashes | Expected while exploring. Judge the model by the "Unseen tracks" lines and by `evaluate`. |
| Backpropagation blocked in a test | This is the CPU fixture's protection. Put bounded CUDA learning coverage under the GPU marker. |
| "Unsupported checkpoint contract" | The model is from version 2 or earlier. Its inputs no longer exist; train a version 3 model. |
| Resume rejects arguments | Only steps, device, CPU threads, and output may change. The step target must exceed the completed count. |
| Output directory exists | Choose a new path; every training/resume run gets a separate directory. |
| Stop seems delayed | The marker is checked every 16 simulator steps; an evaluation or checkpoint write in progress finishes first. |
| Runtime failure has no final checkpoint | Unexpected exceptions do not follow the stop path; use the checkpoint from the last evaluation. |
| Unknown track | Use a built-in name or `random:<seed>`; `random` alone is only valid for `train --tracks`. |
| Blank/noninteractive display | Do not use dummy SDL for manual play. Tests intentionally use offscreen surfaces. |
| Pygame package conflict | Use pygame-ce; do not install both distributions that provide the same `pygame` import. |

## Update the record

Record the date, exact commands, device/runtime, results, and workload bounds in
[VALIDATION.md](VALIDATION.md) after meaningful verification. Distinguish implementation
checks from policy quality and speed measurements. Refresh HANDOFF.md after significant
progress so the next editor can continue without chat history.
