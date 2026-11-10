"""Repeatable evaluation on explicit tracks and seeds, with no optimizer updates."""

import numpy as np

from .agent import ACTIONS
from .simulation import RaceSim
from .tracks import make_track


def evaluate_policy(policy, env_config, tracks, episodes=1, seed=100000):
    """Drive every (track, seed) pair greedily, all at once, from the start line."""
    if episodes < 1 or not tracks or seed < 0:
        raise ValueError("Evaluation needs tracks, positive episodes, and a nonnegative seed")
    runs = [(name, seed + episode) for name in tracks for episode in range(episodes)]
    sim = RaceSim(env_config, len(runs))
    for index, (name, run_seed) in enumerate(runs):
        # The same generator as RaceCarEnv.reset(seed=...), so a result can be re-watched.
        sim.reset_env(index, make_track(name), np.random.default_rng(run_seed))
    batched = getattr(policy, "act_batch", None)
    while not sim.finished.all():
        observations = sim.observe()
        if batched is not None:
            actions = batched(observations, explore=False)
        else:
            actions = [policy.act(observation, explore=False) for observation in observations]
        sim.step(ACTIONS[np.asarray(actions)])
    records = []
    for index, (_, run_seed) in enumerate(runs):
        info = sim.info(index)
        records.append(
            {
                **info,
                "seed": run_seed,
                "reward": float(sim.episode_reward[index]),
                "mean_speed": float(sim.route_distance[index] / max(info["elapsed_seconds"], 1e-9)),
                "checkpoint_fraction": info["checkpoints_passed"] / info["total_checkpoints"],
            }
        )
    return {
        "summary": summarize(records),
        "tracks": {name: summarize([r for r in records if r["track"] == name]) for name in tracks},
        "episodes": records,
    }


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize zero episodes")
    completed = [r for r in records if r["completed"]]
    return {
        "episodes": len(records),
        "completion_rate": float(np.mean([r["completed"] for r in records])),
        "crash_rate": float(np.mean([r["crashed"] for r in records])),
        "mean_progress": float(np.mean([r["progress"] for r in records])),
        "mean_checkpoint_fraction": float(np.mean([r["checkpoint_fraction"] for r in records])),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records])),
        "mean_speed": float(np.mean([r["mean_speed"] for r in records])),
        "mean_finish_seconds": float(np.mean([r["elapsed_seconds"] for r in completed]))
        if completed
        else None,
    }


def ranking(report):
    metrics = report["summary"]
    return (metrics["completion_rate"], metrics["mean_progress"], metrics["mean_reward"])
