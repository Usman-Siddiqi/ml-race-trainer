# Developer reference

This describes the v2 contracts in the source. For setup and runnable user commands, use
[the README](../README.md). For design choices, use [ARCHITECTURE.md](ARCHITECTURE.md).

## Source map and interfaces

| File | Main interfaces and responsibility |
| --- | --- |
| [config.py](../race_trainer/config.py) | Frozen `EnvConfig`, `AgentConfig`, `TrainConfig`; track names and policy contract versions |
| [tracks.py](../race_trainer/tracks.py) | `make_track(name)`, `Track`, `Gate`; sampled centerline, polygons, cached segments, projection, containment, raycasts, directed crossings |
| [environment.py](../race_trainer/environment.py) | `RaceCarEnv(config, render_mode)`; `reset`, `step`, `render`, `close`; physics, reward, observations, episode state |
| [agent.py](../race_trainer/agent.py) | `ACTIONS`, `DQN`, `ReplayBuffer`, `double_dqn_targets`, `DQNAgent`; inference, exploration, experience, learning, agent state |
| [device.py](../race_trainer/device.py) | `ac_power_connected`, `choose_device`, `diagnostics`; explicit selection and metadata |
| [checkpoint.py](../race_trainer/checkpoint.py) | `save_checkpoint`, `load_checkpoint`, `load_policy`; version checks, atomic persistence, inference wrapper |
| [training.py](../race_trainer/training.py) | `run_training(train_config, env_config, agent_config, output=..., resume=..., render_every=...)`; sole coordinator |
| [evaluation.py](../race_trainer/evaluation.py) | `evaluate_policy(policy, env_config, tracks, episodes, seed)`, `summarize`, `ranking`; greedy seeded reports |
| [rendering.py](../race_trainer/rendering.py) | `RaceRenderer`, `track_layer`, drawing/font helpers; track, car, sensors, HUD, offscreen frames |
| [viewer.py](../race_trainer/viewer.py) | `play(env_config, policy=None, seed=..., episodes=...)`; manual/replay loop, returns whether the window was closed |
| [app.py](../race_trainer/app.py) | `RaceApp`, `run_app`; menu, setup, checkpoint picker, subprocess monitor, cooperative stop |
| [cli.py](../race_trainer/cli.py) | `parser`, `training_configuration`, `main(argv)`; lazy command imports and friendly errors |
| [legacy.py](../race_trainer/legacy.py) | `training_entry(visual=False)`; compatibility argument translation |
| [__main__.py](../race_trainer/__main__.py) | `python -m race_trainer` entry point |

`DQNAgent.act(observation, explore=False)` returns an integer index into `ACTIONS`.
`observe(state, action_index, reward, next_state, terminated)` stores a transition and
advances the environment-step counter. Call `learn_if_due()` once after each transition;
it returns a loss or `None` when warming up/not scheduled. Target parameters do not require
gradients. `state_dict`/`load_state_dict` carry agent state; the training coordinator restores
global PyTorch CPU/CUDA RNG from the outer checkpoint separately.

`load_policy(path, device)` returns `(policy, payload)`. The wrapper supports greedy
`act(..., explore=False)` and rejects exploration. It loads only the online network for
inference. Use training resume to restore the optimizer and replay.

## CLI surface

All commands below follow `python -m race_trainer`. No command defaults to training.
Run any subcommand with `--help` for argparse help. The installed `race-trainer` console
entry point invokes the same CLI when its environment is on PATH.

| Command | Options and behavior |
| --- | --- |
| No command / `app` | Open the desktop menu |
| `doctor` | Runtime, packages, power, and CUDA metadata; no GPU tensor workload |
| `tracks` | Layout names, circuit/sprint rules, gate counts, and lengths |
| `manual` | `--track` (oval), `--seed` (42); keyboard driving |
| `render` | `--track` (oval), `--seed` (42), `--output` (`artifacts/track-preview.png`); CPU PNG |
| `train` | Config flags below; `--output`, `--resume`, `--render-every` (0), `--dry-run` |
| `watch` | Required `--model`; optional `--track`, `--episodes` (1), `--seed` (42), `--device` (cpu) |
| `evaluate` | Required `--model`; optional `--tracks`, `--episodes` (5 per track), `--seed` (100000), `--device` (cpu), `--output` |

Train defaults come from the dataclasses below. Duplicate track names are removed while
preserving order. Watch defaults to the saved environment track/physics; evaluate defaults
to the saved training tracks. In the desktop app, watch uses the selected track with the
checkpoint's physics, and training uses one selected track. Multi-track training is a CLI
or Python API feature. Desktop watch uses CPU; CLI watch/evaluate allow CUDA.

CLI success returns 0. Handled configuration/runtime/dependency errors return 2 and print
an explanation to stderr. Argument parsing errors also use argparse's exit behavior.

## Configuration defaults

Dataclasses are frozen and validate the supplied settings. `dataclasses.replace` creates
modified copies. The CLI exposes the flags listed; other settings require the Python API
or a deliberate CLI extension. Editing a saved run's `config.json` does not modify its
checkpoint or the running process.

### EnvConfig

| Field | Default | Meaning / train flag |
| --- | --- | --- |
| `track` | `oval` | Starting layout; CLI derives it from the first `--tracks` item |
| `laps` | `1` | Circuit laps; S sprint only supports one finish |
| `max_steps` | `2000` | Time-limit steps / `--max-steps` |
| `stall_steps` | `240` | Steps without sufficient new forward progress |
| `random_start` | `True` | Seeded lateral offset and heading variation |
| `dt` | `1 / 30` | Simulation seconds per step |
| `max_speed` | `180.0` | Simulation units per second |
| `acceleration` | `120.0` | Positive-throttle acceleration |
| `braking` | `220.0` | Negative-throttle deceleration magnitude |
| `drag` | `15.0` | Constant deceleration; coasting slows down |
| `wheelbase` | `26.0` | Bicycle turning parameter |
| `max_steering` | `0.55` | Maximum steering angle in radians |
| `sensor_range` | `220.0` | Maximum ray distance and normalization scale |

Physics values must be finite; required positive values, nonnegative drag, steering bounds,
and `max_speed * dt <= 10` are enforced. The car is 24 by 12 units. Each step updates speed
and uses two movement/turning substeps, testing all four corners against the road after
each. Negative throttle brakes to zero rather than reversing. Headless execution advances
fixed simulated time without sleeping to match wall-clock time.

Built-in road half-widths are oval 44, S sprint 43, and rounded square 40. Circuits have
12 gates per lap; the sprint has six, with start/finish 25 units inside the road endpoints.
Track objects cache their segment geometry once during construction; `make_track` creates
a track object, not a global singleton.

### AgentConfig

| Field | Default | Meaning / train flag |
| --- | --- | --- |
| `hidden_size` | `128` | Width of each of three ReLU hidden layers |
| `learning_rate` | `0.0003` | Adam rate / `--learning-rate` |
| `gamma` | `0.99` | Discount factor |
| `batch_size` | `128` | Minibatch size / `--batch-size` |
| `replay_capacity` | `50000` | Preallocated CPU ring buffer capacity |
| `learning_starts` | `2000` | Warm-up environment steps / `--learning-starts` |
| `train_every` | `4` | Learn once every N environment steps |
| `target_update_every` | `1000` | Hard target copy every N optimizer updates |
| `epsilon_start` | `1.0` | Initial exploration probability |
| `epsilon_end` | `0.05` | Final exploration probability |
| `exploration_steps` | `100000` | Linear decay duration after warm-up / `--exploration-steps` |
| `gradient_clip` | `10.0` | Gradient norm clipping bound |

Learning requires sufficient replay entries and warm-up steps and a matching step interval.
The loss is Huber/Smooth L1, with a finite-loss check before backpropagation. Online values
select the next action; the target network evaluates it. Replay arrays stay on CPU, and
sampled tensors move to the selected network device.

### TrainConfig

| Field | Default | Meaning / train flag |
| --- | --- | --- |
| `episodes` | `1000` | Total completed-episode target / `--episodes` |
| `tracks` | `oval, s_track, square` | Balanced rotation / `--tracks` |
| `seed` | `42` | Initialization and reset seed base / `--seed` |
| `eval_every` | `25` | Evaluate every N completed episodes / `--eval-every` |
| `eval_episodes` | `5` | Evaluation episodes per track / `--eval-episodes` |
| `save_every` | `100` | Periodic latest checkpoint interval / `--save-every` |
| `device` | `cpu` | `cpu`, `cuda`, `auto` / `--device` |
| `cpu_threads` | `2` | PyTorch CPU thread count / `--cpu-threads` |

Evaluation also runs at the final target episode. Latest checkpoints are saved at the
save interval, evaluation interval, normal finish, and cooperative interruption. Render
intervals use the zero-based episode index, so an enabled interval renders the first episode.
Training environment seeds are `seed + zero_based_episode`; evaluation seeds start at
`100000 + seed` and are reused consistently across evaluations and tracks.

## Observation and action contracts

`OBSERVATION_VERSION = 2`, `ACTION_VERSION = 2`, `OBSERVATION_SIZE = 13`.
Observations are float32 arrays of shape `(13,)`:

| Index | Value | Range |
| --- | --- | --- |
| 0-8 | Ray distance / sensor range, at relative angles -90, -65, -40, -20, 0, 20, 40, 65, 90 degrees | 0 to 1 |
| 9 | Speed / maximum speed | 0 to 1 |
| 10 | Sine of heading to the centerline lookahead relative to car angle | -1 to 1 |
| 11 | Cosine of that relative heading | -1 to 1 |
| 12 | Signed lateral offset / road half-width, clipped | -1 to 1 |

Lookahead distance is `max(45, speed * 0.4)`. The default network is
`13 -> 128 -> 128 -> 128 -> 9`. The environment accepts continuous finite `(throttle, steer)`
pairs, clipped to -1..1. The agent's discrete table is:

| Index | Throttle | Steering | Meaning |
| --- | --- | --- | --- |
| 0 | 1 | -1 | Accelerate left |
| 1 | 1 | 0 | Accelerate straight |
| 2 | 1 | 1 | Accelerate right |
| 3 | 0 | -1 | Coast left |
| 4 | 0 | 0 | Coast straight |
| 5 | 0 | 1 | Coast right |
| 6 | -1 | -1 | Brake left |
| 7 | -1 | 0 | Brake straight |
| 8 | -1 | 1 | Brake right |

`reset(seed=...)` returns `(observation, info)`. `step(action)` returns
`(observation, reward, terminated, truncated, info)`; reset after either ending.
Crash and finish terminate. Stalls and the step limit truncate. Store only `terminated`
in replay's terminal mask so truncation retains the value bootstrap.

Reward is signed centerline advance / 12, minus 0.02 per step, minus 0.01 times the absolute
steering change, plus 2 per ordered gate, plus 20 at the finish, minus 10 on a crash.
Projection advance wraps on circuits and is clipped to plausible maximum movement.

The info dictionary includes `track`, `steps`, `checkpoints_passed`, `total_checkpoints`,
`current_checkpoint`, `laps_completed`, `progress`, `crashed`, `completed`, `speed`,
`elapsed_seconds`, and `termination_reason`. Reasons are `running`, `crash`, `finish`,
`stall`, `time_limit`. Info's total gates include all configured laps; the environment's
`total_checkpoints` property is the per-lap gate count. Current gate indices are zero-based.

## Run files and stdout events

Runs use a newly created directory under `runs/`, or a new explicit `--output` directory.
Existing directories are rejected. Paths are relative to the process working directory.

| File | Contents |
| --- | --- |
| `config.json` | `training`, `environment`, `agent`, `resolved_device`, `device_reason`, `resumed_from` |
| `metrics.jsonl` | `started`, `episode`, and `best` events |
| `evaluations.jsonl` | `evaluation` events with full reports |
| `best.pt` | Highest evaluation ranking so far |
| `latest.pt` | Latest periodic/final resumable state |
| `stop.request` | Cooperative stop marker |

Each logged event is also flushed to stdout as JSON. The desktop reader thread enqueues
lines; the UI polls them and updates its chart/status without running learning on the UI
thread. `finished`/`stopped` result events are printed to stdout after the final save;
they are not currently appended to `metrics.jsonl`.

Episode events include `episode`, `total`, `reward`, `epsilon`, mean `loss` (or null during
warm-up), `env_steps`, `updates`, `wall_seconds`, and the environment info fields.
Evaluation reports contain `summary`, per-track `tracks`, and individual `episodes`.
Summary fields are episode count, completion/crash rates, mean checkpoint fraction,
mean reward, reward standard deviation, and mean finish seconds (null if none finished).
Ranking compares completion rate, then checkpoint fraction, then mean reward.

## Checkpoint schema and resume

`FORMAT_VERSION = 2`. Files use `torch.save`, a temporary file in the destination directory,
and atomic `os.replace`. Loading uses `weights_only=True` and `map_location="cpu"` before
moving network/optimizer tensors to the selected device. Loading validates versions,
required fields, completed episode count, and dataclass configurations.

| Outer key | State |
| --- | --- |
| `format_version`, `observation_version`, `action_version` | Compatibility contract |
| `env_config`, `train_config` | Serialized dataclass settings |
| `agent` | Fields listed below |
| `completed_episodes` | Count of fully completed training episodes |
| `best_score` | Evaluation ranking tuple; negative infinities before any evaluation |
| `episode_boundary` | Whether the checkpoint was saved between complete episodes |
| `torch_rng`, `cuda_rng` | Global PyTorch CPU RNG and CUDA RNG state list |

Agent fields: `config`, `online`, `target`, `optimizer`, `replay`, `rng`, `env_steps`,
`updates`. Replay fields: `capacity`, `size`, `position`, `states`, `next_states`, `actions`,
`rewards`, `terminated`; arrays are serialized as CPU tensors for their populated portion.
Agent RNG is NumPy's generator state. Optimizer restore deep-copies its input state so
Adam CPU step counters and moments cannot alias another restored agent/checkpoint.

CLI resume permits only episodes, device, CPU threads, output, and rendering overrides.
Episodes are a total target above the completed count. Track order, seed, physics, learning
settings, evaluation schedule, and save schedule remain the checkpoint's saved settings.
The new run retains best ranking and all agent state. A partial episode restarts from its
seed with its prior experience still in replay; car state is not serialized mid-episode.

The stop marker is checked every 32 environment steps. Evaluation/saving can delay stop
handling. Normal completion, Ctrl+C, and the cooperative stop path save `latest.pt`.
Other runtime exceptions propagate; recovery then uses the most recent periodic checkpoint.

The original seven-input `best_race_model.pth` is deliberately incompatible. Keep it with
the original source baseline rather than converting it to a v2 checkpoint.

## Compatibility files and infrastructure

| Files | Role |
| --- | --- |
| `main.py`, `simple_main.py`, `debug_main.py` | Open the menu |
| `train_model.py`, `visual_train.py`, `visual_training.py`, `improved_visual_training.py` | Legacy training/test argument adapters; no arguments prints help |
| `watch_model.py` | Unified watch command; requires a v2 model |
| `test_generalization.py` | Unified evaluation; supplies all three tracks unless explicitly chosen |
| `quick_test.py`, `print_checkpoints.py` | Doctor metadata and track listing |
| `race_environment.py` | Compatibility exports for `RaceCarEnv` and `EnvConfig` |
| `pyproject.toml`, `.python-version`, `uv.lock` | Packaging, Python 3.12 setup, CPU/CUDA extras, dev tools, dependency resolution |
| `requirements.txt`, `requirements-cuda.txt` | Pip fallbacks; the full reproducible lock is uv's |
| `setup.ps1`, `Launch Race Lab.cmd` | Isolated Windows environment and menu launcher |
| `.github/workflows/ci.yml` | Windows/Linux CPU tests, Ruff lint, and format checks |
| `.gitignore`, `.gitattributes` | Generated/local data exclusions and source line endings |

The base package depends on NumPy, Gymnasium, and pygame-ce. CPU/CUDA extras are mutually
exclusive and select official PyTorch wheel indexes. The recorded CUDA build is cu128.
No external API, credential, downloaded training dataset, or service is required.
