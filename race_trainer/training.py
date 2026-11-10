"""The sole training loop. Calling run_training is always an explicit user action."""

import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .agent import ACTIONS, DQNAgent
from .checkpoint import load_checkpoint, save_checkpoint
from .config import AgentConfig, EnvConfig, TrainConfig
from .device import choose_device, default_cpu_threads
from .environment import VectorRaceEnv
from .evaluation import evaluate_policy, ranking

# Settings a resumed run may change; everything else must match the checkpoint.
RESUME_OVERRIDES = ("total_steps", "device", "cpu_threads")


class TrainingStopped(Exception):
    pass


def _describe(event):
    """One readable console line per event; the JSON form is for tools."""
    kind = event["event"]
    if kind == "started":
        return (
            f"Training on {event['device']} ({event['reason']}) | {event['num_envs']} cars, "
            f"{event['training_tracks']} training tracks | {event['total_steps']:,} steps\n"
            f"Run directory: {event['output']}"
        )
    if kind == "progress":
        loss = "warm-up" if event["loss"] is None else f"{event['loss']:.4f}"
        return (
            f"[{event['steps'] / event['total']:6.1%}] {event['steps']:>10,} steps | "
            f"finish {event['completion_rate']:4.0%}  crash {event['crash_rate']:4.0%}  "
            f"progress {event['mean_progress']:4.0%} | explore {event['epsilon']:5.1%}  "
            f"loss {loss} | {event['steps_per_second']:,.0f} steps/s  "
            f"ETA {event['eta_seconds'] / 60:.1f} min"
        )
    if kind == "evaluation":
        summary = event["summary"]
        return (
            f"  Unseen tracks at {event['steps']:,} steps: finish "
            f"{summary['completion_rate']:.0%}, crash {summary['crash_rate']:.0%}, "
            f"progress {summary['mean_progress']:.0%}, {summary['mean_speed']:.0f} units/s"
        )
    if kind == "best":
        return f"  New best model saved: {event['path']}"
    return (
        f"{kind.title()} at {event['steps']:,} steps in {event['wall_seconds'] / 60:.1f} min. "
        f"Checkpoint: {event['checkpoint']}"
    )


def run_training(
    train_config=None,
    env_config=None,
    agent_config=None,
    *,
    output=None,
    resume=None,
    json_events=False,
):
    cfg = train_config or TrainConfig()
    env_cfg = env_config or EnvConfig()
    agent_cfg = agent_config or AgentConfig()
    device = choose_device(cfg.device)
    torch.set_num_threads(cfg.cpu_threads or default_cpu_threads(device.name))
    best_score = (-float("inf"),) * 3
    payload = load_checkpoint(resume) if resume else None
    if payload:
        prior = asdict(TrainConfig(**payload["train_config"]))
        changed = [
            key
            for key, value in asdict(cfg).items()
            if key not in RESUME_OVERRIDES and value != prior[key]
        ]
        if changed or EnvConfig(**payload["env_config"]) != env_cfg:
            raise ValueError(
                "Resume must preserve the saved seed, tracks, and environment configuration"
            )
        if AgentConfig(**payload["agent"]["config"]) != agent_cfg:
            raise ValueError("Resume must preserve the saved agent configuration")
        if payload["completed_steps"] >= cfg.total_steps:
            raise ValueError("--steps is a total target; choose a value above the completed count")
        best_score = tuple(payload["best_score"])
    directory = (
        Path(output) if output else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    directory.mkdir(parents=True, exist_ok=False)
    configuration = {
        "training": asdict(cfg),
        "environment": asdict(env_cfg),
        "agent": asdict(agent_cfg),
        "resolved_device": device.name,
        "device_reason": device.reason,
        "resumed_from": str(resume) if resume else None,
    }
    (directory / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    previous_best = Path(resume).with_name("best.pt") if resume else None
    if previous_best and previous_best.exists():
        # The new directory stays self-contained even if this run never beats the old best.
        shutil.copy2(previous_best, directory / "best.pt")
    agent = DQNAgent(agent_cfg, device.name, cfg.seed, cfg.num_envs)
    if payload:
        agent.load_state_dict(payload["agent"])
        # Cars restart on resume, so earlier steps must not chain into the new episodes.
        agent.replay.mark_break()
        torch.set_rng_state(payload["torch_rng"])
        if device.name == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state(payload["cuda_rng"][0])
    specs = cfg.training_specs()
    env = VectorRaceEnv(env_cfg, specs, cfg.num_envs, seed=[cfg.seed, agent.env_steps])
    stop_file = directory / "stop.request"
    started = time.monotonic()
    first_step = agent.env_steps
    metrics = (directory / "metrics.jsonl").open("w", encoding="utf-8")
    evaluations = (directory / "evaluations.jsonl").open("w", encoding="utf-8")

    def emit(event, stream=metrics):
        encoded = json.dumps(event, allow_nan=False)
        stream.write(encoded + "\n")
        stream.flush()
        print(encoded if json_events else _describe(event), flush=True)

    def evaluate_and_save():
        nonlocal best_score
        # Score exactly the weights that would be saved, not a slightly older driving copy.
        agent.sync_actor()
        report = evaluate_policy(agent, env_cfg, cfg.eval_tracks, cfg.eval_episodes, cfg.seed)
        emit({"event": "evaluation", "steps": agent.env_steps, **report}, evaluations)
        score = ranking(report)
        if score > best_score:
            best_score = score
            save_checkpoint(directory / "best.pt", agent, env_cfg, cfg, best_score)
            emit(
                {
                    "event": "best",
                    "steps": agent.env_steps,
                    "path": str(directory / "best.pt"),
                    "score": score,
                }
            )
        save_checkpoint(directory / "latest.pt", agent, env_cfg, cfg, best_score)

    stopped = False
    try:
        emit(
            {
                "event": "started",
                "device": device.name,
                "reason": device.reason,
                "output": str(directory.resolve()),
                "total_steps": cfg.total_steps,
                "completed_steps": agent.env_steps,
                "num_envs": cfg.num_envs,
                "training_tracks": len(specs),
            }
        )
        observations = env.reset()
        finished, losses = [], []
        window_start, window_steps = time.monotonic(), agent.env_steps
        iteration = 0
        while agent.env_steps < cfg.total_steps:
            if iteration % 16 == 0 and stop_file.exists():
                raise TrainingStopped
            iteration += 1
            before = agent.env_steps
            actions = agent.act_batch(observations, explore=True)
            following, rewards, terminated, truncated, successors, episodes = env.step(
                ACTIONS[actions]
            )
            # Keep truncation separate: it ends the rollout, not the Bellman bootstrap.
            agent.observe(
                observations, actions, rewards, successors, terminated, terminated | truncated
            )
            observations = following
            if episodes is not None:
                finished.append(episodes)
            for _ in range(agent.updates_due()):
                losses.append(agent.learn())
            done = agent.env_steps >= cfg.total_steps
            if before // cfg.log_every != agent.env_steps // cfg.log_every or done:
                now = time.monotonic()
                rate = (agent.env_steps - window_steps) / max(now - window_start, 1e-9)
                merged = {
                    key: np.concatenate([episode[key] for episode in finished] or [np.zeros(0)])
                    for key in ("progress", "completed", "crashed", "reward", "steps")
                }
                count = len(merged["reward"])
                emit(
                    {
                        "event": "progress",
                        "steps": agent.env_steps,
                        "total": cfg.total_steps,
                        "episodes": count,
                        "completion_rate": float(merged["completed"].mean()) if count else 0.0,
                        "crash_rate": float(merged["crashed"].mean()) if count else 0.0,
                        "mean_progress": float(merged["progress"].mean()) if count else 0.0,
                        "reward": float(merged["reward"].mean()) if count else 0.0,
                        "mean_episode_steps": float(merged["steps"].mean()) if count else 0.0,
                        "epsilon": agent.epsilon,
                        "loss": float(np.mean(losses)) if losses else None,
                        "updates": agent.updates,
                        "steps_per_second": rate,
                        "eta_seconds": (cfg.total_steps - agent.env_steps) / max(rate, 1e-9),
                        "wall_seconds": now - started,
                    }
                )
                finished, losses = [], []
                window_start, window_steps = now, agent.env_steps
            if before // cfg.eval_every != agent.env_steps // cfg.eval_every or done:
                paused = time.monotonic()
                evaluate_and_save()
                window_start += time.monotonic() - paused
    except (KeyboardInterrupt, TrainingStopped):
        stopped = True
        save_checkpoint(directory / "latest.pt", agent, env_cfg, cfg, best_score)
    finally:
        result = {
            "event": "stopped" if stopped else "finished",
            "steps": agent.env_steps,
            "steps_this_run": agent.env_steps - first_step,
            "wall_seconds": time.monotonic() - started,
            "output": str(directory.resolve()),
            "checkpoint": str((directory / "latest.pt").resolve()),
            "best": str((directory / "best.pt").resolve())
            if (directory / "best.pt").exists()
            else None,
        }
        if stopped or agent.env_steps >= cfg.total_steps:
            emit(result)
        metrics.close()
        evaluations.close()
    return result
