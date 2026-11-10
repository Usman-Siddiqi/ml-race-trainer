"""Short throughput measurements for choosing --envs and --device on this machine.

Each measurement runs for about a second on synthetic work. Nothing is saved and no
driving policy is produced.
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


def run_benchmark(seconds=1.0):
    """Print simulator and learner throughput and the settings they suggest."""
    import torch

    from .device import default_cpu_threads

    defaults, agent = TrainConfig(), AgentConfig()
    print("Simulator (cars in parallel -> environment steps per second)")
    simulator = {}
    for count in (1, 16, 32, 64, 128):
        simulator[count] = benchmark_simulator(count, seconds)
        print(f"  {count:4d} cars  {simulator[count]:9,.0f} steps/s")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("Learner (milliseconds per update -> transitions learned per second)")
    learner = {}
    for device in devices:
        for batch in sorted({512, agent.batch_size, 4096}):
            cost = benchmark_learner(device, batch, seconds, default_cpu_threads(device))
            learner[device, batch] = cost
            print(f"  {device:4} batch {batch:5d}  {cost * 1000:6.2f} ms  {batch / cost:10,.0f} /s")
    print("Estimated training speed with the default update schedule")
    estimates = {}
    for device in devices:
        per_step = 1 / simulator[defaults.num_envs] + learner[device, agent.batch_size] / (
            agent.train_every
        )
        estimates[device] = 1 / per_step
        print(f"  --device {device:4}  about {estimates[device]:,.0f} steps/s")
    best = max(estimates, key=estimates.get)
    minutes = defaults.total_steps / estimates[best] / 60
    print(
        f"Suggested: --device {best} --envs {defaults.num_envs} "
        f"(a default {defaults.total_steps:,}-step run takes about {minutes:.0f} min)"
    )
    return {"simulator": simulator, "learner": learner, "estimates": estimates}
