"""The sole training loop. Calling run_training is always an explicit user action."""

import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch

from .agent import ACTIONS, DQNAgent
from .checkpoint import load_checkpoint, save_checkpoint
from .config import AgentConfig, EnvConfig, TrainConfig
from .device import choose_device
from .environment import RaceCarEnv
from .evaluation import evaluate_policy, ranking


class TrainingStopped(Exception):
    pass


def _write_event(stream, event):
    encoded = json.dumps(event, allow_nan=False)
    stream.write(encoded + "\n")
    stream.flush()
    print(encoded, flush=True)


def run_training(
    train_config=None,
    env_config=None,
    agent_config=None,
    *,
    output=None,
    resume=None,
    render_every=0,
):
    cfg = train_config or TrainConfig()
    env_cfg = env_config or EnvConfig(track=cfg.tracks[0])
    agent_cfg = agent_config or AgentConfig()
    if render_every < 0:
        raise ValueError("render_every must be zero or positive")
    if "s_track" in cfg.tracks and env_cfg.laps != 1:
        raise ValueError("Mixed training with the sprint requires laps=1")
    device = choose_device(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    completed = 0
    best_score = (-float("inf"),) * 3
    payload = load_checkpoint(resume) if resume else None
    if payload:
        prior = TrainConfig(**payload["train_config"])
        if (
            prior.seed != cfg.seed
            or tuple(prior.tracks) != tuple(cfg.tracks)
            or EnvConfig(**payload["env_config"]) != env_cfg
        ):
            raise ValueError(
                "Resume must preserve the saved seed, tracks, and environment configuration"
            )
        if AgentConfig(**payload["agent"]["config"]) != agent_cfg:
            raise ValueError("Resume must preserve the saved agent configuration")
        completed = payload["completed_episodes"]
        best_score = tuple(payload["best_score"])
        if completed >= cfg.episodes:
            raise ValueError(
                "--episodes is a total target; choose a value above the completed count"
            )
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
    agent = DQNAgent(agent_cfg, device.name, cfg.seed)
    if payload:
        agent.load_state_dict(payload["agent"])
        torch.set_rng_state(payload["torch_rng"])
        if device.name == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state(payload["cuda_rng"][0])
    environments = {name: RaceCarEnv(replace(env_cfg, track=name)) for name in cfg.tracks}
    stop_file = directory / "stop.request"
    boundary, stopped = True, False
    started = time.monotonic()
    try:
        with (
            (directory / "metrics.jsonl").open("w", encoding="utf-8") as metrics,
            (directory / "evaluations.jsonl").open("w", encoding="utf-8") as evaluations,
        ):
            _write_event(
                metrics,
                {
                    "event": "started",
                    "device": device.name,
                    "reason": device.reason,
                    "output": str(directory.resolve()),
                    "episodes": cfg.episodes,
                    "completed_episodes": completed,
                    "resume_restarts_partial_episode": bool(
                        payload and not payload["episode_boundary"]
                    ),
                },
            )
            for episode in range(completed, cfg.episodes):
                env = environments[cfg.tracks[episode % len(cfg.tracks)]]
                should_render = render_every > 0 and episode % render_every == 0
                env.render_mode = "human" if should_render else None
                observation, _ = env.reset(seed=cfg.seed + episode)
                boundary = False
                reward_sum, losses = 0.0, []
                while True:
                    if agent.env_steps % 32 == 0 and stop_file.exists():
                        raise TrainingStopped
                    action = agent.act(observation, explore=True)
                    next_observation, reward, terminated, truncated, info = env.step(
                        ACTIONS[action]
                    )
                    # Keep truncation separate: it ends the rollout, not the Bellman bootstrap.
                    agent.observe(observation, action, reward, next_observation, terminated)
                    loss = agent.learn_if_due()
                    if loss is not None:
                        losses.append(loss)
                    observation = next_observation
                    reward_sum += reward
                    if should_render:
                        import pygame

                        env.render()
                        if any(
                            e.type == pygame.QUIT
                            or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                            for e in pygame.event.get()
                        ):
                            raise TrainingStopped
                    if terminated or truncated:
                        break
                env.close()
                completed = episode + 1
                boundary = True
                _write_event(
                    metrics,
                    {
                        "event": "episode",
                        "episode": completed,
                        "total": cfg.episodes,
                        "reward": reward_sum,
                        "epsilon": agent.epsilon,
                        "loss": sum(losses) / len(losses) if losses else None,
                        "env_steps": agent.env_steps,
                        "updates": agent.updates,
                        "wall_seconds": time.monotonic() - started,
                        **info,
                    },
                )
                if completed % cfg.eval_every == 0 or completed == cfg.episodes:
                    report = evaluate_policy(
                        agent, env_cfg, cfg.tracks, cfg.eval_episodes, 100000 + cfg.seed
                    )
                    _write_event(
                        evaluations, {"event": "evaluation", "episode": completed, **report}
                    )
                    score = ranking(report)
                    if score > best_score:
                        best_score = score
                        save_checkpoint(
                            directory / "best.pt", agent, env_cfg, cfg, completed, best_score
                        )
                        _write_event(
                            metrics,
                            {
                                "event": "best",
                                "episode": completed,
                                "path": str(directory / "best.pt"),
                                "score": score,
                            },
                        )
                if completed % cfg.save_every == 0 or completed % cfg.eval_every == 0:
                    save_checkpoint(
                        directory / "latest.pt", agent, env_cfg, cfg, completed, best_score
                    )
    except (KeyboardInterrupt, TrainingStopped):
        stopped = True
    finally:
        for env in environments.values():
            env.close()
    save_checkpoint(
        directory / "latest.pt",
        agent,
        env_cfg,
        cfg,
        completed,
        best_score,
        episode_boundary=boundary,
    )
    result = {
        "event": "stopped" if stopped else "finished",
        "episodes": completed,
        "env_steps": agent.env_steps,
        "output": str(directory.resolve()),
        "checkpoint": str((directory / "latest.pt").resolve()),
    }
    print(json.dumps(result), flush=True)
    return result
