"""Repeatable evaluation on explicit tracks and seeds, with no optimizer updates."""

from dataclasses import replace

import numpy as np

from .agent import ACTIONS
from .environment import RaceCarEnv


def evaluate_policy(policy, env_config, tracks, episodes=5, seed=100000):
    if episodes < 1 or not tracks or seed < 0:
        raise ValueError("Evaluation needs tracks, positive episodes, and a nonnegative seed")
    records = []
    for name in tracks:
        env = RaceCarEnv(
            replace(env_config, track=name, laps=1 if name == "s_track" else env_config.laps)
        )
        try:
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                total_reward = 0.0
                while True:
                    action = policy.act(observation, explore=False)
                    observation, reward, terminated, truncated, info = env.step(ACTIONS[action])
                    total_reward += reward
                    if terminated or truncated:
                        break
                records.append(
                    {
                        **info,
                        "seed": seed + episode,
                        "reward": total_reward,
                        "checkpoint_fraction": info["checkpoints_passed"]
                        / info["total_checkpoints"],
                    }
                )
        finally:
            env.close()
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
        "mean_checkpoint_fraction": float(np.mean([r["checkpoint_fraction"] for r in records])),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records])),
        "mean_finish_seconds": float(np.mean([r["elapsed_seconds"] for r in completed]))
        if completed
        else None,
    }


def ranking(report):
    metrics = report["summary"]
    return (metrics["completion_rate"], metrics["mean_checkpoint_fraction"], metrics["mean_reward"])
