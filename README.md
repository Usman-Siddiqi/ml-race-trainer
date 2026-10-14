# Race Lab · ML Race Trainer

A racing and reinforcement-learning lab for manual driving, Double DQN training, and
repeatable evaluation. Version 2 shares one geometry across walls, sensors, and finish gates.
Opening the app shows a menu. Training starts only through **Start training** or an explicit
`train` command.

For continuing development in a new editor or coding tool, start with the
[project handoff](docs/HANDOFF.md), [project instructions](AGENTS.md), and
[documentation index](docs/README.md).

![Race Lab menu](docs/ui-preview.png)

## Start on Windows

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. From this folder:

```powershell
# Install NVIDIA CUDA support. Setup runs no GPU workload or training.
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Device cuda

# Double-click Launch Race Lab.cmd, or:
$racePython = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv\Scripts\python.exe'
& $racePython -m race_trainer app
```

Use `-Device cpu` for a smaller CPU-only installation. Setup uses Python 3.12 and `uv.lock`.
Its environment lives under `%LOCALAPPDATA%\RaceTrainer\venv`, so large Python/CUDA files
stay outside OneDrive. It does not modify global Python. The launcher also supports a
conventional project `.venv` when the local runtime is absent.

On other platforms, run `uv sync --locked --python 3.12 --extra cpu`, then use the Python
executable inside `.venv`. Do not install `pygame` alongside `pygame-ce`; both provide the
same `pygame` import. Pip fallback files are provided, but uv uses the full dependency lock.

## Drive and inspect

```powershell
& $racePython -m race_trainer doctor
& $racePython -m race_trainer tracks
& $racePython -m race_trainer manual --track s_track
& $racePython -m race_trainer render --track square --output artifacts\square.png
```

`doctor` reads package, power, and CUDA metadata without running a benchmark or GPU tensor
operation. `render` produces a CPU-rendered PNG without a policy or training.

| Control | Action |
| --- | --- |
| W / Up | Accelerate |
| S / Down | Brake |
| A D / Left Right | Steer |
| Release throttle | Coast |
| Space | Pause |
| R | Restart with the same episode seed |
| Tab | Toggle sensors |
| Esc | Return to the menu |
| Enter, after a model episode | Next replay episode / finish replay |

Oval and rounded-square circuits have 12 ordered gates per lap. The S-curve is a sprint
with six gates and a finish before the road ends. Gates count only in sequence and in the
correct direction. Speed is measured in simulation units per second.

## Devices

**CPU is the default for training and replay.**

| Option | Behavior |
| --- | --- |
| `--device cpu` | CPU execution; two training threads by default |
| `--device cuda` | Networks and minibatches use the NVIDIA GPU; errors clearly if unavailable |
| `--device auto` | CUDA only when available and Windows reports external power; otherwise CPU |

Automatic selection uses the operating system's power report. USB power may count as
external power; select `cpu` for an unambiguous choice. Explicit `cuda` overrides power
detection. Automatic selection uses CPU when power status is unknown. Simulation and replay
storage remain on CPU. No GPU speedup has been measured for this small network.

## Training, when ready

Inspect the configuration without constructing an agent or starting a run:

```powershell
& $racePython -m race_trainer train --dry-run --device cpu
```

The following command **starts training** when invoked:

```powershell
& $racePython -m race_trainer train --episodes 1000 --tracks oval s_track square --device cuda
```

Desktop setup trains on the selected track. The CLI defaults to all three in a balanced
rotation. Tracks and physics are saved with the run. Options include `--seed`, `--max-steps`,
`--batch-size`, `--learning-rate`, `--learning-starts`, `--exploration-steps`, `--eval-every`,
`--eval-episodes`, `--save-every`, and `--cpu-threads`. `--render-every 25` shows one episode
every 25; the default is headless.

Each run gets a new directory under `runs/`. An explicit `--output` must also name a new directory.

| File | Contents |
| --- | --- |
| `config.json` | Configuration, chosen device, and resume source |
| `metrics.jsonl` | Episode rewards, loss, exploration, steps, and finish reasons |
| `evaluations.jsonl` | Per-track evaluation summaries and individual episode results |
| `best.pt` | Best fixed-seed evaluation result |
| `latest.pt` | Periodic and final resume state |

Use **Stop & save**, close the training monitor, or press Ctrl+C in a training terminal to
save `latest.pt`. A `stop.request` file is checked between environment steps. If stopped
mid-episode, resume restarts that episode and retains experience already in replay.

```powershell
& $racePython -m race_trainer train --resume runs\YOUR_RUN\latest.pt --episodes 2000 --device cuda
```

`--episodes` is the total target, including completed episodes. Resume preserves seed, tracks,
physics, and learning settings and creates a new run directory. Checkpoints contain both
networks, Adam state, replay, counters, random state, configuration, and observation/action
versions. Saves replace files atomically.

## Watch and evaluate

```powershell
& $racePython -m race_trainer watch --model runs\YOUR_RUN\best.pt
& $racePython -m race_trainer watch --model runs\YOUR_RUN\best.pt --track square
& $racePython -m race_trainer evaluate --model runs\YOUR_RUN\best.pt --tracks oval s_track square --episodes 10 --output artifacts\evaluation.json
```

Replay defaults to the saved track and physics. The menu uses its visibly selected track
with the checkpoint's physics. Greedy evaluation reports completion/crash rates, checkpoint
fraction, reward mean/deviation, and successful finish time. These three fixed layouts and
seeded starting variations do not establish generalization to arbitrary unseen circuits.

## What changed

- Nine wall sensors, speed, relative lookahead heading, and lateral offset: 13 normalized inputs.
- Nine actions: accelerate/coast/brake crossed with left/straight/right.
- Bicycle-style turning, drag, fixed simulation steps, directed gates, and progress rewards.
- Vectorized raycasts and cached geometry shared by rendering, sensors, and collisions.
- Double DQN, Huber loss, gradient clipping, and exploration paced by environment steps.
- Truncation ends a rollout while retaining the Bellman bootstrap.
- One training loop, one CLI, and a responsive desktop monitor with a background process.

Your original `best_race_model.pth` is preserved. Its seven-input weights are incompatible
with the changed sensors, actions, and physics. Version 2 rejects it with an explanation.
The recovered original code is committed at `c940c81` for historical comparisons.

Old script names are adapters: `main.py`, `simple_main.py`, and `debug_main.py` open the
menu; training scripts without arguments print help. `watch_model.py` and
`test_generalization.py` use version 2 checkpoints. `quick_test.py` reads diagnostics.

## Development

```powershell
& $racePython -m pytest -q
& $racePython -m ruff check .
& $racePython -m ruff format --check .
```

The default suite forces CPU operation, blocks backpropagation and Adam updates, and skips
GPU tests. It covers track finishes using fixed arithmetic steering, geometry, Gymnasium
contracts, seeded resets, Double DQN targets, replay/checkpoint restoration, device selection,
startup, rendering, and a mocked monitor process. CI runs these CPU checks on Windows and Linux.

To explicitly run the bounded CUDA checks on a machine with the CUDA runtime installed:

```powershell
& $racePython -m pytest -q -m gpu --run-gpu --junitxml=artifacts\gpu-validation.xml
```

These six checks execute GPU inference, backpropagation, Adam updates, target synchronization,
CPU/CUDA checkpoint restoration, CLI stop/resume, and a real desktop training subprocess.
Integration runs are limited to 24-40 steps per episode and write disposable models under
pytest's temporary directory. They verify the learning pipeline; they do not produce a
trained driver or measure whether CUDA is faster than CPU. CUDA checks fail if explicitly
requested without a working GPU.

See [architecture](docs/ARCHITECTURE.md) and [validation record](docs/VALIDATION.md).

References: [Gymnasium time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/),
[PyTorch installation](https://pytorch.org/get-started/locally/), and [pygame-ce](https://pyga.me/docs/).
