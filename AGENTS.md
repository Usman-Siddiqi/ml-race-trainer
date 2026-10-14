# Project instructions

This repository is Race Lab / ML Race Trainer v2, a Python racing simulator and Double DQN
learning lab. These instructions apply to any coding tool working in this checkout.

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
  an explicit command or the Start training button.
- CPU is the default. Explicit CUDA and power-aware auto selection are supported.
  Bounded GPU tests may be run with `--run-gpu`; full policy training and sustained
  benchmarks need a user request. Earlier bounded GPU checks are already authorized.
- Geometry for drawing, sensors, collision, progress, and finish gates must stay consistent.
- Crash/finish are terminal. Time limits/stalls truncate and retain the Bellman bootstrap.
- Preserve the original `best_race_model.pth`. Its seven-input weights are incompatible
  with v2 and are intentionally rejected.
- Checkpoints must preserve the learning configuration, replay, RNG, and optimizer state.
  Each optimizer restore must own independent tensors; keep the Adam alias regression test.
- Observation/action changes need explicit contract versioning and compatibility handling.
- Resume creates a new run directory and preserves saved parameters. A partially stopped
  episode restarts while retaining its experience in replay.

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

Use Python 3.12 and `uv.lock` for the recorded setup. Reinstall through `setup.ps1` after
moving the source folder so the editable install resolves the correct package.

## Working conventions

- Preserve user edits and significant local history. Commit significant changes separately.
- Keep weights, runs, artifacts, and environments out of Git, as configured in `.gitignore`.
- Update the relevant docs when behavior, defaults, compatibility, or validation changes.
- Keep implementation status separate from measured policy quality and performance.
- Report what changed and what was actually verified. Publishing or pushing is a separate
  action from making local commits.

No coding-tool memory, cloud service, API key, or proprietary workspace state is required
to understand or run this project.
