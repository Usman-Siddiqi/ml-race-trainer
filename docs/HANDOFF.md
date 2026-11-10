# Project handoff

Updated 2026-11-10. This guide is the starting point for continuing the project in any
editor or coding tool. Everything needed for development is in this repository's source and
Markdown documentation.

## What this project is

Race Lab is a local Python racing simulator with a Pygame desktop interface and a PyTorch
Double DQN learner. Cars drive from car-mounted sensors only. They train headlessly on
generated tracks, 64 at a time, and are then tested on tracks they have never seen. You can
drive manually, watch a trained model on any track, and score it with one command.

Tracks are either one of three hand-built layouts (oval, rounded square, S-curve sprint) or
`random:<seed>`, a reproducible generated circuit or sprint. There is no camera perception
and no external dataset.

## Current state

Version 3 is implemented, trained, and measured. It replaced version 2 on 2026-11-09/10
because version 2 trained poorly and its policy was fed centerline guidance from the map;
[ARCHITECTURE.md](ARCHITECTURE.md) explains what was wrong and what changed.

A default training run takes under three minutes on the recorded laptop and produces a
model that finishes nearly every unseen track it is given, including the three hand-built
layouts that are no longer trained on. Exact figures, seeds, and commands are in
[VALIDATION.md](VALIDATION.md); do not quote results from anywhere else.

The working branch is `main` and is pushed to `origin`. Use `git status` and `git log` for
the current state.

## Open the right folder and runtime

The active project folder on this computer is:

```text
C:\Users\usman\Documents\Code\ml-race-trainer
```

The Windows runtime is installed outside OneDrive:

```text
%LOCALAPPDATA%\RaceTrainer\venv
```

From the repository root in PowerShell:

```powershell
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -c "import sys, race_trainer; print(sys.executable); print(race_trainer.__file__)"
& $racePython -m race_trainer doctor
& $racePython -m race_trainer train --dry-run
```

The import path should point to this checkout's `race_trainer` directory. These commands
inspect the runtime/configuration without starting training. Launch the menu through
`Launch Race Lab.cmd` or `& $racePython -m race_trainer app`.

If that environment is missing (it was on 2026-11-09 and had to be recreated), or on a new
machine or a moved checkout, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Device cuda
```

Use `-Device cpu` for the CPU build. Setup requires `uv`, uses Python 3.12 and the lockfile,
and installs the development tools. Reinstall after moving the checkout because the
environment contains an editable package install. The launcher falls back to a project
`.venv` if the LocalAppData runtime is absent. See [README](../README.md) for other platforms.

The recorded machine has a Ryzen 7 8845HS (8 cores/16 threads), 16 GiB RAM, and an RTX 4050
Laptop GPU with about 6 GiB VRAM; Python 3.12.12, PyTorch 2.11.0+cu128, driver 621.01.
Training defaults to `--device auto`: CUDA when Windows reports external power and CUDA is
available, otherwise CPU. No AMD NPU execution path is implemented.

## The everyday loop

`race.cmd` in the repository root runs any command with the installed environment, so
these work in a fresh terminal opened in this folder:

```powershell
.\race train --output runs\my-run
.\race evaluate --model runs\my-run\best.pt
.\race watch --model runs\my-run\best.pt
.\race benchmark
```

`runs\v3-default\best.pt` is an already trained model on this machine.

## Files that Git does not carry

| Local item | How to preserve or recreate it |
| --- | --- |
| `best_race_model.pth` | Preserve separately when copying/cloning. This is the original incompatible seven-input policy. Its SHA-256 is recorded in VALIDATION.md. |
| `runs/` and `*.pt` | Ignored. `runs\v3-default` holds the model behind the validation record; any model can be retrained in minutes. Version 2 checkpoints no longer load. |
| `artifacts/` | Optional local previews and evaluation reports; ignored and reproducible. |
| `%LOCALAPPDATA%\RaceTrainer\venv` | Recreate with setup; avoid copying environment binaries between machines. |

No runtime model is bundled with the repository.

## Design and file map

The active package is `race_trainer/`; old root scripts forward to it. There is one
simulator (`RaceSim`), used for one car or many, one learning loop, and shared manual/replay
rendering. Start with [ARCHITECTURE.md](ARCHITECTURE.md), then use
[REFERENCE.md](REFERENCE.md) for module APIs, exact input/action indices, all configuration
defaults, run logs, and checkpoint fields.

[DEVELOPMENT.md](DEVELOPMENT.md) explains where to make changes, how to change the track
generator or observations, which tests cover each subsystem, how to make training faster,
and how to diagnose problems. [Documentation index](README.md) links the full set.

## Significant history

| Commit | Purpose |
| --- | --- |
| `c940c81` | Preserved recovered original edits; historical v1 source baseline |
| `4910b49` to `862a8a3` | Version 2: shared geometry, Double DQN, unified CLI/UI, checkpoints, bounded CUDA tests |
| `8b513bb` | Version 3 simulator: sensors-only observations, generated tracks, batched `RaceSim` |
| `b9220d9` | Version 3 learner: multi-car headless training, n-step replay, unseen-track evaluation |
| `337b76d` | Steadier and faster training: annealed learning rate, steering noise, newest-leader `best.pt`, concurrent learner, clearer `evaluate` and `benchmark` |
| Following commit | Version 3 documentation and the measured validation record |

## Work that can come next

These are possibilities, not unfinished promises:

1. **Harder tracks.** The generator makes smooth single loops. Hairpins that double back,
   chicanes, crossings, and much narrower roads are untested; adding a second shape family
   would show how far sensors-only driving generalizes.
2. **Faster laps.** The policy averages well under top speed through corners. Compare it
   against a hand-written racing line, or try a longer run, a larger network, or continuous
   control, measuring lap time on the test tracks.
3. **Simulator speed.** The simulator now limits training speed. Coarser centerline spacing
   or several simulator processes are the next levers; see DEVELOPMENT.md.
4. **Multiple laps and traffic.** Laps beyond one are implemented but unmeasured. Other
   cars on track would need new sensors and a contract version bump.
5. **Desktop app.** Training setup exposes steps and device only; track choice, car count,
   and an in-app evaluation table could be added.

## First prompt for the next coding session

```text
Continue this Race Lab / ML Race Trainer v3 repository.
Read AGENTS.md, docs/HANDOFF.md, README.md, docs/ARCHITECTURE.md,
docs/REFERENCE.md, docs/DEVELOPMENT.md, and docs/VALIDATION.md first.
Inspect git status/log and use the isolated Python 3.12 runtime
(recreate it with setup.ps1 if it is missing).
Cars drive from sensors only and train on generated tracks; keep the
validation, test, and built-in tracks out of default training.
Check changes to learning by training (minutes) and running evaluate
on unseen tracks. Commit significant changes and update the relevant docs.
My next task is: [describe the next change or experiment].
```
