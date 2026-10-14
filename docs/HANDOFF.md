# Project handoff

Updated 2026-10-13. This guide is the starting point for continuing the project in T3 Code
or another editor. Everything needed for development is in this repository's source and
Markdown documentation.

## What this project is

Race Lab is a local Python/Gymnasium racing simulator with a Pygame desktop interface and
a PyTorch Double DQN learner. You can drive manually, inspect sensors, train a policy,
replay a v2 checkpoint, and evaluate it on fixed tracks and seeds.

There are three layouts: an oval circuit, a rounded square circuit, and an S-curve sprint.
The policy sees wall distances plus speed and centerline guidance. This is a driving/control
experiment; there is no camera perception system or external dataset.

## Current state

The v2 implementation and bounded CUDA validation are complete. The latest code/validation
commit before this documentation handoff is `862a8a3`. The working branch is `main`.
At that baseline it contained nine local commits beyond `origin/main`, including the
integration merge. The documentation handoff adds another local commit. Use `git status`
and `git log` for the current state; nothing has been pushed as part of this work.

Last recorded checks on 2026-10-13:

| Check | Result |
| --- | --- |
| Default CPU suite | 45 passed; six GPU tests skipped |
| Explicit CUDA suite | Six passed on RTX 4050 |
| Lint, formatting, Git whitespace, dependency lock | Passed |
| All three tracks physically drivable | Passed with a fixed arithmetic controller |
| Actual learning/checkpoint/stop/resume/desktop child | Passed in tiny disposable CUDA runs |

Full training, convergence, learned lap times, and GPU speedup have not been measured.
There is no validated trained v2 driving policy from this work. GPU test models are
disposable fixtures, not candidates for normal replay or further training.

See [VALIDATION.md](VALIDATION.md) for dates, scope, resources, and limitations.

## Open the right folder and runtime

The active project folder on this computer is:

```text
C:\Users\usman\Documents\Code\ml-race-trainer
```

Open that folder directly in T3 Code. It contains the full Git history, unpublished local
commits, ignored original model, validation artifacts, and project docs. The former OneDrive
copy is still present because Windows reports that it is in use; use the Documents copy as
the active checkout.

The Windows runtime is installed outside OneDrive:

```text
%LOCALAPPDATA%\RaceTrainer\venv
```

From the repository root in PowerShell:

```powershell
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -c "import sys, race_trainer; print(sys.executable); print(race_trainer.__file__)"
& $racePython -m race_trainer doctor
& $racePython -m race_trainer train --dry-run --device cpu
```

The import path should point to this checkout's `race_trainer` directory. These commands
inspect the runtime/configuration without starting training. Launch the menu through
`Launch Race Lab.cmd` or `& $racePython -m race_trainer app`.

For a new machine or a moved checkout, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Device cuda
```

Use `-Device cpu` for the CPU build. Setup requires `uv`, uses Python 3.12 and the lockfile,
and installs the development tools. Reinstall after moving the checkout because the
environment contains an editable package install. The launcher falls back to a project
`.venv` if the LocalAppData runtime is absent. See [README](../README.md) for other platforms.

The recorded machine has a Ryzen 7 8845HS (8 cores/16 threads), 16 GiB RAM, and an RTX 4050
Laptop GPU with about 6 GiB VRAM. The validated runtime was Python 3.12.12 and PyTorch
2.11.0+cu128 with driver 617.42. Recheck metadata after hardware/runtime changes. CPU stays
the default; auto uses CUDA only when Windows reports external power and CUDA is available.
No AMD NPU execution path is implemented.

## Files that Git does not carry

| Local item | How to preserve or recreate it |
| --- | --- |
| `best_race_model.pth` | Preserve separately when copying/cloning. This is the original incompatible seven-input policy. Its SHA-256 is recorded in VALIDATION.md. |
| `runs/` and `*.pt` | Copy any wanted v2 checkpoints and run logs separately; these are ignored. No long training run was created during the rebuild. |
| `artifacts/` | Optional local previews and test reports; ignored and reproducible. The dated Markdown validation record is tracked. |
| `%LOCALAPPDATA%\RaceTrainer\venv` | Recreate with setup; avoid copying environment binaries between machines. |

Keep the `.git` directory/history when transferring the current checkout. A source ZIP or
remote clone alone omits local commits and ignored models. No runtime model is bundled
with the repository.

## Design and file map

The active package is `race_trainer/`; old root scripts forward to it. The implementation
has one simulator, one learning loop, and shared manual/replay rendering. Start with
[ARCHITECTURE.md](ARCHITECTURE.md), then use [REFERENCE.md](REFERENCE.md) for module APIs,
exact input/action indices, all configuration defaults, run logs, and checkpoint fields.

[DEVELOPMENT.md](DEVELOPMENT.md) explains where to make changes, how to add tracks or
observations, which tests cover each subsystem, and how to diagnose installation, CUDA,
OneDrive, and checkpoint problems. [Documentation index](README.md) links the full set.

## Significant history

| Commit | Purpose |
| --- | --- |
| `c940c81` | Preserved recovered original edits; historical v1 source baseline |
| `4910b49` | Shared geometry and correct episode completion |
| `761a71b` | Double DQN, CUDA selection, and resumable learning pipeline |
| `029a0a9` | Unified CLI, Race Lab UI, and explicit training controls |
| `cc50d74` | Resume validation and CPU startup guarantees |
| `21b8a09` | Windows runtime setup, CPU CI, and initial documentation |
| `626b7fc` | Independent optimizer state on checkpoint restore |
| `862a8a3` | Bounded CUDA tests and completion validation record |

The history also includes an integration merge. The commit preserving original edits
was integrated before the final v2 checkout; the table groups changes by purpose rather
than providing a strict chronological log.

## Work that can come next

These are experiments or future features, not unfinished promises from the rebuild:

1. When the user requests training, run a controlled v2 baseline and inspect per-track
   finish rates, crashes, checkpoint progress, and loss. Save the configuration and seeds.
2. Compare CPU and CUDA using matched settings and a bounded workload. This small network
   may spend considerable time in simulation and transfers, so measure before optimizing.
3. Investigate learning failures before increasing network size or training duration.
   Keep terminal/truncation handling, exploration pacing, and evaluation seeds intact.
4. Add varied track geometry and held-out layouts if arbitrary-track generalization becomes
   a goal. Current evaluation covers three known layouts and seeded starting variations.
5. Consider curriculum, parallel environments, or richer UI configuration after a measured
   baseline establishes a useful target. These features are not currently implemented.

Bounded GPU tests are authorized. Continuing this documentation work is not a request to
start a full training run or sustained benchmark.

## First prompt for the next coding session

```text
Continue this Race Lab / ML Race Trainer v2 repository.
Read AGENTS.md, docs/HANDOFF.md, README.md, docs/ARCHITECTURE.md,
docs/REFERENCE.md, docs/DEVELOPMENT.md, and docs/VALIDATION.md first.
Inspect git status/log and use the existing isolated Python 3.12 runtime.
Preserve the original model and local commits. The implementation is complete;
45 CPU and 6 bounded GPU tests passed on 2026-10-13. Full policy training and
GPU speedup remain unmeasured. Do not start a long training run unless requested.
Commit significant changes and update the relevant docs.
My next task is: [describe the next change or experiment].
```
