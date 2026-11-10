"""Short throughput measurements for choosing --envs and --device on this machine.

Each measurement runs for a second or a few. Nothing is kept and no usable driving
policy is produced.
"""

import time

import numpy as np

from .agent import ACTIONS, EXPLORATION_WEIGHTS, DQNAgent
from .config import OBSERVATION_SIZE, AgentConfig, EnvConfig, TrainConfig, generated_tracks
from .environment import VectorRaceEnv


def benchmark_simulator(count, seconds=1.0, seed=0):
    """Environment steps per second for ``count`` cars taking random training actions."""
    env = VectorRaceEnv(EnvConfig(), generated_tracks(0, 64), count, seed)
    rng = np.random.default_rng(seed)
    env.reset()
    steps, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds:
        env.step(ACTIONS[rng.choice(len(ACTIONS), count, p=EXPLORATION_WEIGHTS)])
        steps += count
    return steps / (time.perf_counter() - started)


def benchmark_learner(device, batch_size, seconds=1.0, cpu_threads=None):
    """Seconds per optimizer update on random transitions, after a warm-up."""
    import torch

    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    config = AgentConfig(batch_size=batch_size, replay_capacity=max(20_000, batch_size))
    agent = DQNAgent(config, device, seed=0, num_envs=64)
    rng = np.random.default_rng(0)
    for _ in range(max(64, batch_size // 64 + 1)):
        states = rng.random((64, OBSERVATION_SIZE), dtype=np.float32)
        ended = rng.random(64) < 0.02
        agent.observe(
            states, rng.integers(0, len(ACTIONS), 64), rng.random(64), states, ended, ended
        )
    for _ in range(10):
        agent.learn()
    updates, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds:
        agent.learn()
        updates += 1
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / updates


def benchmark_training(device, steps=40_000):
    """Steps per second of the real training loop, simulator and learner together.

    A short run in a temporary directory; the warm-up before learning starts is excluded.
    """
    import contextlib
    import io
    import json
    import tempfile
    from pathlib import Path

    from .training import run_training

    train = TrainConfig(
        total_steps=steps,
        track_pool=64,
        eval_tracks=("oval",),
        eval_every=10**9,
        log_every=max(steps // 4, 64),
        device=device,
    )
    agent = AgentConfig(learning_starts=steps // 4)
    with tempfile.TemporaryDirectory() as folder:
        with contextlib.redirect_stdout(io.StringIO()):
            run_training(train, EnvConfig(), agent, output=Path(folder) / "run")
        lines = (Path(folder) / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    rates = [e["steps_per_second"] for e in events if e["event"] == "progress"]
    return float(np.median(rates[1:]))


def run_benchmark(seconds=1.0):
    """Print simulator, learner, and end-to-end training throughput on this machine."""
    import torch

    from .device import default_cpu_threads

    defaults, agent = TrainConfig(), AgentConfig()
    print("Simulator alone (cars in parallel -> environment steps per second)")
    simulator = {}
    for count in (1, 16, 32, 64, 128):
        simulator[count] = benchmark_simulator(count, seconds)
        print(f"  {count:4d} cars  {simulator[count]:9,.0f} steps/s")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("Learner alone (milliseconds per update -> transitions learned per second)")
    learner = {}
    for device in devices:
        for batch in sorted({512, agent.batch_size, 4096}):
            cost = benchmark_learner(device, batch, seconds, default_cpu_threads(device))
            learner[device, batch] = cost
            print(f"  {device:4} batch {batch:5d}  {cost * 1000:6.2f} ms  {batch / cost:10,.0f} /s")
    print("Training, both together with default settings")
    training = {}
    for device in devices:
        torch.set_num_threads(default_cpu_threads(device))
        training[device] = benchmark_training(device, max(8_192, int(40_000 * seconds)))
        print(f"  --device {device:4}  {training[device]:9,.0f} steps/s")
    best = max(training, key=training.get)
    minutes = defaults.total_steps / training[best] / 60
    # A short test flatters a long run: replay sampling slows as the buffer outgrows the
    # CPU cache, and a real run also stops to evaluate and save.
    print(
        f"Fastest here: --device {best}. A default {defaults.total_steps:,}-step run takes "
        f"at least {minutes:.1f} min at this rate; expect up to half as long again."
    )
    return {"simulator": simulator, "learner": learner, "training": training}
