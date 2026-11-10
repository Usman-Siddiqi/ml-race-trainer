# Developer reference

This describes the version 3 contracts in the source. For setup and runnable user commands,
use [the README](../README.md). For design choices, use [ARCHITECTURE.md](ARCHITECTURE.md).

## Source map and interfaces

| File | Main interfaces and responsibility |
| --- | --- |
| [config.py](../race_trainer/config.py) | Frozen `EnvConfig`, `AgentConfig`, `TrainConfig`; contract versions; `check_track`, `track_seed`, `validation_tracks`, `heldout_tracks`, `generated_tracks` |
| [tracks.py](../race_trainer/tracks.py) | `make_track(name)`, `generate_track(seed)`, `Track`, `Gate`, `TrackBatch`; `batch_raycast`, `batch_on_road`, `batch_project`, `gate_crossed` |
| [simulation.py](../race_trainer/simulation.py) | `RaceSim(config, count)`: `reset_env`, `step`, `observe`, `progress`, `info`, `corners`; physics, reward, endings for N cars |
| [environment.py](../race_trainer/environment.py) | `RaceCarEnv(config, render_mode)` for one car; `VectorRaceEnv(config, tracks, count, seed)` for training |
| [agent.py](../race_trainer/agent.py) | `ACTIONS`, `EXPLORATION_WEIGHTS`, `DQN`, `ReplayBuffer`, `double_dqn_targets`, `DQNAgent` |
| [device.py](../race_trainer/device.py) | `ac_power_connected`, `choose_device`, `default_cpu_threads`, `diagnostics` |
| [checkpoint.py](../race_trainer/checkpoint.py) | `save_checkpoint`, `load_checkpoint`, `load_policy`; version checks, atomic persistence |
| [training.py](../race_trainer/training.py) | `run_training(train_config, env_config, agent_config, output=..., resume=..., json_events=...)`; sole coordinator |
| [evaluation.py](../race_trainer/evaluation.py) | `evaluate_policy(policy, env_config, tracks, episodes, seed)`, `summarize`, `ranking` |
| [benchmark.py](../race_trainer/benchmark.py) | `benchmark_simulator`, `benchmark_learner`, `benchmark_training`, `run_benchmark` |
| [rendering.py](../race_trainer/rendering.py) | `RaceRenderer`, `track_layer`, drawing/font helpers |
| [viewer.py](../race_trainer/viewer.py) | `play(env_config, policy=None, seed=..., episodes=...)`; returns whether the window was closed |
| [app.py](../race_trainer/app.py) | `RaceApp`, `run_app`, `MENU_TRACKS`; menu, setup, checkpoint picker, subprocess monitor |
| [cli.py](../race_trainer/cli.py) | `parser`, `training_configuration`, `main(argv)`; lazy command imports and friendly errors |
| [legacy.py](../race_trainer/legacy.py) | `training_entry(visual=False)`; compatibility argument translation |

`DQNAgent(config, device, seed, num_envs)` owns the networks, optimizer, replay, and
exploration state for `num_envs` cars.

- `act_batch(observations, explore=False)` returns one action index per row.
- `act(observation)` is the greedy single-observation form; it rejects `explore=True`.
- `observe(states, actions, rewards, next_states, terminated, ended)` stores one step of
  every car and advances `env_steps` by `num_envs`. `ended` is terminated or truncated.
- `updates_due()` says how many optimizer updates the schedule owes; `learn()` performs one
  and returns its loss.
- `learn()` is `apply_update(sample_batch())` followed by a driving-copy refresh when
  stale. The coordinator calls the two halves itself: `sample_batch()` on the main thread,
  `apply_update(batch)` on a worker thread, which touches only learner-side state.
- `sync_actor(only_if_stale=False)` copies the learner's weights to `actor`, the CPU
  network that chooses actions. It must not be called while an update is running.
- `set_progress(fraction)` sets the annealed learning rate for `steps / total_steps`.
- `state_dict` / `load_state_dict` carry agent state; the coordinator restores global
  PyTorch RNG from the outer checkpoint.

`load_policy(path, device)` returns `(policy, payload)`. The policy offers greedy `act` and
`act_batch` and rejects exploration. It loads only the online network.

`VectorRaceEnv.step(actions)` returns `(observations, rewards, terminated, truncated,
next_observations, episodes)`. `next_observations` are true successors, including final
ones, and go to replay. `observations` are what to act on next: a restarted car sees its
new track. `episodes` is `None` or arrays (`progress`, `completed`, `crashed`, `reward`,
`steps`) for the cars that just ended. With `steering_noise`, the executed steering is
the given steering plus noise; replay still records the action the agent chose, and the
next observation reports the steering that was executed.

## CLI surface

All commands follow `python -m race_trainer`. No command defaults to training. The installed
`race-trainer` console entry point invokes the same CLI.

| Command | Options and behavior |
| --- | --- |
| No command / `app` | Open the desktop menu |
| `doctor` | Runtime, packages, power, and CUDA metadata; no workload |
| `tracks` | Built-in layouts and the generated seed ranges |
| `benchmark` | `--seconds` per measurement (1.0, at most 10); simulator alone, learner alone, and a short real training run per device |
| `manual` | `--track` (oval), `--seed` (42); keyboard driving |
| `render` | `--track` (oval), `--seed` (42), `--output` (`artifacts/track-preview.png`) |
| `train` | Flags below; `--output`, `--resume`, `--json`, `--dry-run` |
| `watch` | Required `--model`; `--track` (`random:2000000000`), `--episodes` (1), `--seed` (42), `--device` (cpu) |
| `evaluate` | Required `--model`; `--tracks`, `--generated` (20), `--episodes` (1), `--seed` (100000), `--device` (cpu), `--output`, `--json` |

`--track` and `--tracks` accept a built-in name or `random:<seed>` with a seed up to
2^31 - 1. `train --tracks` also accepts `random`, the generated pool. Duplicates are removed
in order. Without `--tracks`, `evaluate` uses the three built-ins plus `--generated` test
layouts. Its report lists `seen_in_training`: evaluated tracks that were in the model's
training set. The table gives totals to one decimal, the count of runs finished, and up to
ten failures with the `watch` arguments that replay each. `watch` always uses one lap with
the checkpoint's physics.

CLI success returns 0. Handled configuration/runtime/dependency errors return 2 and print an
explanation to stderr. Argument parsing errors use argparse's exit behavior.

## Configuration defaults

Dataclasses are frozen and validate their settings. The CLI exposes the flags listed; other
settings need the Python API. Editing a saved run's `config.json` changes nothing.

### EnvConfig

| Field | Default | Meaning / train flag |
| --- | --- | --- |
| `track` | `oval` | Layout for the one-car environment; training uses `TrainConfig.tracks` |
| `laps` | `1` | Circuit laps; sprints always finish once |
| `max_steps` | `2000` | Time-limit steps / `--max-steps` |
| `stall_steps` | `120` | Steps without one unit of new progress before truncation |
| `random_start` | `True` | Seeded lateral offset (6 units) and heading offset (0.06 rad) |
| `dt` | `1 / 30` | Simulation seconds per step |
| `max_speed` | `180.0` | Units per second |
| `acceleration` | `120.0` | Positive-throttle acceleration |
| `braking` | `220.0` | Negative-throttle deceleration |
| `drag` | `15.0` | Constant deceleration |
| `wheelbase` | `26.0` | Bicycle turning parameter |
| `max_steering` | `0.55` | Maximum steering angle in radians |
| `grip` | `260.0` | Maximum lateral acceleration; caps yaw rate at `grip / speed` |
| `sensor_range` | `220.0` | Maximum ray distance and normalization scale |
| `progress_reward` | `0.05` | Reward per unit advanced along the centerline |
| `steering_penalty` | `0.01` | Reward lost per unit of steering change |
| `crash_penalty` | `5.0` | Reward lost on a crash |
| `finish_bonus` | `5.0` | Reward gained at the finish |

Physics values must be finite and positive where that is required; `max_speed * dt <= 10` is
enforced. The car is 24 by 12 units. Headless execution advances simulated time without
sleeping.

### AgentConfig

| Field | Default | Meaning / train flag |
| --- | --- | --- |
| `hidden_size` | `256` | Width of the two hidden layers |
| `learning_rate` | `0.0003` | Initial Adam rate / `--learning-rate` |
| `final_learning_rate` | `0.00003` | Rate reached at `total_steps`, annealed linearly; at most `learning_rate` |
| `gamma` | `0.99` | Discount factor |
| `n_step` | `3` | Steps summed before bootstrapping |
| `batch_size` | `2048` | Transitions per update / `--batch-size` |
| `replay_capacity` | `400000` | Transitions kept, shared across cars |
| `learning_starts` | `20000` | Warm-up environment steps / `--learning-starts` |
| `train_every` | `64` | Environment steps per update |
| `target_update_every` | `100` | Updates between target copies |
| `epsilon_start` | `1.0` | Initial exploration probability |
| `epsilon_end` | `0.004` | Final exploration of the most careful car |
| `epsilon_spread` | `15.0` | Ratio between the most and least exploratory car |
| `exploration_steps` | `300000` | Linear decay duration after warm-up / `--exploration-steps` |
| `exploration_hold` | `4` | Mean steps a random action is held |
| `actor_sync_every` | `4` | Updates between refreshes of a CUDA learner's CPU driving copy |
| `gradient_clip` | `10.0` | Gradient norm bound |

Replay holds `replay_capacity // num_envs` steps of every car. Exploration probabilities per
car are `max(epsilon_start * (1 - fraction), floor)`, with floors spaced geometrically from
`epsilon_end` to `epsilon_end * epsilon_spread`.

### TrainConfig

| Field | Default | Meaning / train flag |
| --- | --- | --- |
| `total_steps` | `1000000` | Environment steps across all cars / `--steps` |
| `num_envs` | `64` | Cars simulated in parallel / `--envs` |
| `tracks` | `random` | Training tracks / `--tracks` |
| `track_pool` | `512` | Layouts `random:0..N-1` that `random` expands to / `--track-pool` |
| `eval_tracks` | 16 validation layouts | Scored during training to choose `best.pt` |
| `seed` | `42` | Initialization, exploration, and track draws / `--seed` |
| `eval_every` | `100000` | Steps between evaluations / `--eval-every` |
| `eval_episodes` | `1` | Start variations per evaluation track / `--eval-episodes` |
| `log_every` | `20000` | Steps between progress events / `--log-every` |
| `steering_noise` | `0.2` | Standard deviation of steering error added in training, as a fraction of full lock; 0 disables / `--steering-noise` |
| `device` | `auto` | `auto`, `cuda`, `cpu` / `--device` |
| `cpu_threads` | `0` | PyTorch CPU threads; 0 picks 2 on CUDA, up to 8 on CPU / `--cpu-threads` |

`training_specs()` expands `random` into explicit names. Each restarted car draws one of
those names uniformly. Evaluation also runs at the final step. `latest.pt` is saved after
every evaluation, at the end, and on a stop.

## Observation and action contracts

`OBSERVATION_VERSION = 3`, `ACTION_VERSION = 3`, `OBSERVATION_SIZE = 18`.

| Index | Value | Range |
| --- | --- | --- |
| 0-14 | Ray distance / sensor range, at -90, -70, -50, -35, -22, -13, -6, 0, 6, 13, 22, 35, 50, 70, 90 degrees from the heading | 0 to 1 |
| 15 | Speed / maximum speed | 0 to 1 |
| 16 | Steering applied on the previous step | -1 to 1 |
| 17 | Throttle applied on the previous step | -1 to 1 |

Rays start at the car's center. The action table is throttle-major:

| Index | Throttle | Steering |
| --- | --- | --- |
| 0-4 | 1 (accelerate) | -1, -0.5, 0, 0.5, 1 |
| 5-9 | 0 (coast) | -1, -0.5, 0, 0.5, 1 |
| 10-14 | -1 (brake) | -1, -0.5, 0, 0.5, 1 |

A random exploratory action picks throttle with weights 0.6 / 0.25 / 0.15 and steering with
weights 0.15 / 0.2 / 0.3 / 0.2 / 0.15.

`RaceCarEnv.reset(seed=...)` returns `(observation, info)`. `step(action)` returns
`(observation, reward, terminated, truncated, info)`; reset after either ending. The info
dictionary has `track`, `steps`, `checkpoints_passed`, `total_checkpoints`,
`current_checkpoint`, `laps_completed`, `progress`, `crashed`, `completed`, `speed`,
`elapsed_seconds`, and `termination_reason` (`running`, `crash`, `finish`, `stall`,
`time_limit`). `total_checkpoints` is the number of gates needed to finish.

## Tracks

| Property | Built-ins | Generated |
| --- | --- | --- |
| Half-width | oval 44, S sprint 43, square 40 | 28 to 44 |
| Gates | 12 per lap; sprint 6 | About one per 108 units; at least 6 (circuit) or 3 (sprint) |
| Centerline spacing | As defined | 12 units |

Circuits start at distance 0 and finish after one lap of gates. Sprints start 25 units in
and finish 100 units (`SPRINT_RUNOFF`) before the end wall. `make_track` caches up to 4,096
immutable tracks. `GENERATOR_VERSION` is part of the generator's seed; change it only when
changing what a seed produces, since that invalidates comparisons across versions.

## Run files and events

Runs use a new directory under `runs/`, or a new explicit `--output` directory. Existing
directories are rejected.

| File | Contents |
| --- | --- |
| `config.json` | `training`, `environment`, `agent`, `resolved_device`, `device_reason`, `resumed_from` |
| `metrics.jsonl` | `started`, `progress`, `best`, and the final `finished` / `stopped` event |
| `evaluations.jsonl` | `evaluation` events with full reports |
| `best.pt` | Highest validation ranking so far; copied forward on resume |
| `latest.pt` | Latest resumable state |
| `stop.request` | Cooperative stop marker |

Events go to the files as JSON and to stdout as readable text, or as JSON lines with
`--json`. The desktop app launches training with `--json` and reads those lines.

- `progress`: `steps`, `total`, `episodes`, `completion_rate`, `crash_rate`,
  `mean_progress`, `reward`, `mean_episode_steps`, `epsilon`, `loss` (null during warm-up),
  `updates`, `steps_per_second`, `eta_seconds`, `wall_seconds`. Rates describe the exploring
  training cars since the previous event, so they understate the greedy policy.
- `evaluation`: `steps`, `summary`, per-track `tracks`, and individual `episodes`.
- Summaries: `episodes`, `completion_rate`, `crash_rate`, `mean_progress`,
  `mean_checkpoint_fraction`, `mean_reward`, `reward_std`, `mean_speed`,
  `mean_finish_seconds` (null if none finished).
- Ranking compares completion rate, then mean progress. `best.pt` is replaced when a
  new evaluation ranks at least as high, so the newest leader is kept.

## Checkpoint schema and resume

`FORMAT_VERSION = 3`. Files use `torch.save`, a temporary file in the destination directory,
and atomic `os.replace`. Loading uses `weights_only=True` and `map_location="cpu"`, then
validates versions, required fields, and the dataclass configurations.

| Outer key | State |
| --- | --- |
| `format_version`, `observation_version`, `action_version` | Compatibility contract |
| `env_config`, `train_config` | Serialized dataclass settings |
| `agent` | Fields listed below |
| `completed_steps` | Environment steps completed |
| `best_score` | Validation ranking tuple (completion, progress); negative infinities before any evaluation |
| `torch_rng`, `cuda_rng` | Global PyTorch CPU RNG and CUDA RNG state list |

Agent fields: `config`, `num_envs`, `online`, `target`, `optimizer`, `replay`, `rng`,
`env_steps`, `updates`. Replay fields: `horizon`, `num_envs`, `size`, `position`, and the
populated rows of `states`, `next_states`, `actions`, `rewards`, `terminated`, `ended`.
A full replay makes a checkpoint of roughly 65 MB.

CLI resume permits only `--steps`, `--device`, `--cpu-threads`, and `--output`. Everything
else, including the number of cars, comes from the checkpoint. The new run keeps the best
ranking, copies `best.pt` from beside the resumed checkpoint when present, and marks a
break in replay so earlier steps do not chain into the restarted cars. The learning rate
follows `steps / total_steps`, so raising the target on resume raises the rate again.

The stop marker is checked every 16 simulator steps. Normal completion, Ctrl+C, and the
stop marker all save `latest.pt`. Other exceptions propagate; recovery then uses the most
recent evaluation-time checkpoint.

Earlier models are rejected with an explanation: the original seven-input
`best_race_model.pth`, and version 2 checkpoints, whose inputs included centerline guidance.

## Compatibility files and infrastructure

| Files | Role |
| --- | --- |
| `main.py`, `simple_main.py`, `debug_main.py` | Open the menu |
| `train_model.py`, `visual_train.py`, `visual_training.py`, `improved_visual_training.py` | Legacy argument adapters; no arguments prints help. Training is headless; `--episodes` is rejected in favor of `--steps` |
| `watch_model.py`, `test_generalization.py` | Forward to `watch` and `evaluate` |
| `quick_test.py`, `print_checkpoints.py` | Doctor metadata and track listing |
| `race_environment.py` | Exports `RaceCarEnv`, `VectorRaceEnv`, `EnvConfig` |
| `pyproject.toml`, `.python-version`, `uv.lock` | Packaging, Python 3.12 setup, CPU/CUDA extras, dev tools |
| `requirements.txt`, `requirements-cuda.txt` | Pip fallbacks; the full reproducible lock is uv's |
| `setup.ps1`, `Launch Race Lab.cmd` | Isolated Windows environment and menu launcher |
| `.github/workflows/ci.yml` | Windows/Linux CPU tests, Ruff lint, and format checks |

The base package depends on NumPy, Gymnasium, and pygame-ce. CPU/CUDA extras are mutually
exclusive and select official PyTorch wheel indexes. The recorded CUDA build is cu128.
No external API, credential, downloaded dataset, or service is required.
