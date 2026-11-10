# Project instructions

This repository is Race Lab / ML Race Trainer v3, a Python racing simulator and Double DQN
learning lab. Cars learn from car-mounted sensors on generated tracks and are tested on
tracks they have not seen. These instructions apply to any coding tool working in this checkout.

## Read first

1. [Handoff](docs/HANDOFF.md): current state, local runtime, preserved files, and next steps.
2. [README](README.md): installation, controls, and user commands.
3. [Architecture](docs/ARCHITECTURE.md): module boundaries and design decisions.
4. [Reference](docs/REFERENCE.md): configuration, observations, actions, checkpoints, and logs.
5. [Development](docs/DEVELOPMENT.md): extension workflow, tests, and troubleshooting.
6. [Validation](docs/VALIDATION.md): dated evidence and unmeasured outcomes.

## Preserve these behaviors

- Active implementation lives in `race_trainer/`. Top-level Python scripts are adapters;
  add features to the package rather than creating another training loop.
- Opening the app or its setup screen must not start training. Training starts through
  an explicit command or the Start training button, and runs headless.
- The policy's inputs are limited to what the car can measure: wall-distance rays, speed,
  and its previous controls. Do not feed it centerline, gate, or lap-progress information.
- Default training uses generated tracks only. The validation seeds, the test seeds, and
  the three built-in layouts stay out of the default training pool, and `evaluate` reports
  any evaluated track that a model trained on.
- There is one simulator, `RaceSim`. One car and many cars must agree exactly, and geometry
  for drawing, sensors, collision, progress, and finish gates must stay consistent.
- Crash/finish are terminal. Time limits/stalls truncate and retain the Bellman bootstrap.
- Training defaults to `--device auto`: CUDA on external power when available, otherwise
  CPU. Explicit `cpu` and `cuda` are supported. Watching and evaluating default to CPU.
- Preserve the original `best_race_model.pth`. Its seven-input weights, and version 2
  checkpoints, are incompatible with v3 and are intentionally rejected.
- Checkpoints must preserve the learning configuration, replay, RNG, and optimizer state.
  Each optimizer restore must own independent tensors; keep the Adam alias regression test.
- Observation/action changes need explicit contract versioning and compatibility handling.
- Resume creates a new run directory and preserves saved parameters. Cars restart on
  resume while their earlier experience stays in replay behind a break marker.

## Runtime and checks

On this Windows machine, use the isolated environment outside OneDrive:

```powershell
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -m pytest -q
& $racePython -m ruff check .
& $racePython -m ruff format --check .
```

GPU checks are explicit: `& $racePython -m pytest -q -m gpu --run-gpu`.
The default suite hides CUDA and prohibits backpropagation/Adam updates. Do not disable
those protections to make a CPU test pass. Run checks appropriate to code changes;
documentation-only changes need link/content checks rather than rerunning learning tests.

A default training run takes a few minutes on this machine, so training to check a change
to learning, rewards, physics, or tracks is expected: train, then score the result with
`evaluate` on unseen tracks. Compare settings over more than one seed before changing
defaults, and record what was measured in the validation record.

Use Python 3.12 and `uv.lock` for the recorded setup. Reinstall through `setup.ps1` after
moving the source folder so the editable install resolves the correct package. If the
LocalAppData environment is missing, `setup.ps1 -Device cuda` recreates it.

## Working conventions

- Preserve user edits and significant local history. Commit significant changes separately.
- Keep weights, runs, artifacts, and environments out of Git, as configured in `.gitignore`.
- Update the relevant docs when behavior, defaults, compatibility, or validation changes.
- Keep implementation status separate from measured policy quality and performance.
- Report what changed and what was actually verified. Publishing or pushing is a separate
  action from making local commits.

No coding-tool memory, cloud service, API key, or proprietary workspace state is required
to understand or run this project.
