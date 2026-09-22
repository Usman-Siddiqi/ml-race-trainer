"""One command surface for the desktop app, simulator, learner, and evaluation."""

import argparse
import json
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .config import TRACK_NAMES, AgentConfig, EnvConfig, TrainConfig


def parser():
    result = argparse.ArgumentParser(
        prog="race-trainer", description="Race Lab: drive, train, and evaluate"
    )
    sub = result.add_subparsers(dest="command")
    sub.add_parser("app", help="Open the menu; never starts training automatically")
    sub.add_parser("doctor", help="Read runtime and GPU metadata without a benchmark")
    sub.add_parser("tracks", help="List tracks and finish rules")
    manual = sub.add_parser("manual", help="Drive with keyboard controls")
    manual.add_argument("--track", choices=TRACK_NAMES, default="oval")
    manual.add_argument("--seed", type=int, default=42)
    render = sub.add_parser(
        "render", help="Save a CPU-rendered track preview; no policy or training"
    )
    render.add_argument("--track", choices=TRACK_NAMES, default="oval")
    render.add_argument("--output", type=Path, default=Path("artifacts/track-preview.png"))
    render.add_argument("--seed", type=int, default=42)
    watch = sub.add_parser("watch", help="Replay a version 2 checkpoint")
    watch.add_argument("--model", type=Path, required=True)
    watch.add_argument("--track", choices=TRACK_NAMES)
    watch.add_argument("--episodes", type=int, default=1)
    watch.add_argument("--seed", type=int, default=42)
    watch.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    evaluate = sub.add_parser("evaluate", help="Evaluate a checkpoint on explicit tracks and seeds")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--tracks", nargs="+", choices=TRACK_NAMES)
    evaluate.add_argument("--episodes", type=int, default=5, help="Episodes per track")
    evaluate.add_argument("--seed", type=int, default=100000)
    evaluate.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    evaluate.add_argument("--output", type=Path)
    train = sub.add_parser("train", help="Explicitly start or resume training")
    train.add_argument(
        "--episodes", type=int, help="Total target, including episodes already completed"
    )
    train.add_argument("--tracks", nargs="+", choices=TRACK_NAMES)
    train.add_argument("--device", choices=("cpu", "cuda", "auto"))
    train.add_argument("--seed", type=int)
    train.add_argument("--cpu-threads", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--learning-starts", type=int)
    train.add_argument("--exploration-steps", type=int)
    train.add_argument("--eval-every", type=int)
    train.add_argument("--eval-episodes", type=int)
    train.add_argument("--save-every", type=int)
    train.add_argument(
        "--output", type=Path, help="New run directory; existing directories are never overwritten"
    )
    train.add_argument("--resume", type=Path, help="Version 2 checkpoint to continue from")
    train.add_argument(
        "--render-every", type=int, default=0, help="Render one episode every N; zero is headless"
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration without creating an agent or training",
    )
    return result


def training_configuration(args):
    payload = None
    if args.resume:
        fixed = (
            "tracks",
            "seed",
            "max_steps",
            "batch_size",
            "learning_rate",
            "learning_starts",
            "exploration_steps",
            "eval_every",
            "eval_episodes",
            "save_every",
        )
        if any(getattr(args, key) is not None for key in fixed):
            raise ValueError(
                "Resume preserves saved parameters; only episodes, device, CPU threads, "
                "output, and rendering may be overridden"
            )
        from .checkpoint import load_checkpoint

        payload = load_checkpoint(args.resume)
        train_cfg = TrainConfig(**payload["train_config"])
        env_cfg = EnvConfig(**payload["env_config"])
        agent_cfg = AgentConfig(**payload["agent"]["config"])
    else:
        train_cfg, env_cfg, agent_cfg = TrainConfig(), EnvConfig(), AgentConfig()
    train_updates = {
        k: getattr(args, k)
        for k in (
            "episodes",
            "device",
            "seed",
            "cpu_threads",
            "eval_every",
            "eval_episodes",
            "save_every",
        )
        if getattr(args, k) is not None
    }
    if args.tracks:
        train_updates["tracks"] = tuple(dict.fromkeys(args.tracks))
    train_cfg = replace(train_cfg, **train_updates)
    env_updates = {"track": train_cfg.tracks[0]} if not payload else {}
    if args.max_steps is not None:
        env_updates["max_steps"] = args.max_steps
    env_cfg = replace(env_cfg, **env_updates)
    agent_updates = {
        k: getattr(args, k)
        for k in ("batch_size", "learning_rate", "learning_starts", "exploration_steps")
        if getattr(args, k) is not None
    }
    agent_cfg = replace(agent_cfg, **agent_updates)
    if args.render_every < 0:
        raise ValueError("render-every must be zero or positive")
    if payload and payload["completed_episodes"] >= train_cfg.episodes:
        raise ValueError("--episodes must exceed the number already completed")
    return train_cfg, env_cfg, agent_cfg


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in (None, "app"):
            from .app import run_app

            run_app()
        elif args.command == "doctor":
            from .device import diagnostics

            print(json.dumps(diagnostics(), indent=2))
        elif args.command == "tracks":
            from .tracks import make_track

            for name in TRACK_NAMES:
                track = make_track(name)
                print(
                    f"{name:8}  {'circuit' if track.closed else 'sprint':7}  "
                    f"{len(track.gates):2} gates  {track.length:.0f} units"
                )
        elif args.command == "manual":
            from .viewer import play

            play(EnvConfig(track=args.track), seed=args.seed)
        elif args.command == "render":
            import numpy as np
            import pygame

            from .environment import RaceCarEnv

            env = RaceCarEnv(EnvConfig(track=args.track), render_mode="rgb_array")
            try:
                env.reset(seed=args.seed)
                pixels = env.render()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(
                    pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2))), args.output
                )
            finally:
                env.close()
            print(args.output.resolve())
        elif args.command in ("watch", "evaluate"):
            if args.episodes < 1 or args.seed < 0:
                raise ValueError("episodes must be positive and seed nonnegative")
            from .checkpoint import load_policy
            from .device import choose_device

            choice = choose_device(args.device)
            policy, payload = load_policy(args.model, choice.name)
            env_cfg = EnvConfig(**payload["env_config"])
            if args.command == "watch":
                from .viewer import play

                if args.track:
                    env_cfg = replace(
                        env_cfg,
                        track=args.track,
                        laps=1 if args.track == "s_track" else env_cfg.laps,
                    )
                print(f"Replay: {env_cfg.track} | {choice.reason}")
                play(env_cfg, policy, seed=args.seed, episodes=args.episodes)
            else:
                from .evaluation import evaluate_policy

                tracks = tuple(dict.fromkeys(args.tracks or payload["train_config"]["tracks"]))
                report = evaluate_policy(policy, env_cfg, tracks, args.episodes, args.seed)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
                    )
                print(
                    json.dumps(
                        {
                            "device": choice.name,
                            "tracks": report["tracks"],
                            "summary": report["summary"],
                        },
                        indent=2,
                        allow_nan=False,
                    )
                )
        elif args.command == "train":
            configs = training_configuration(args)
            if args.dry_run:
                from .device import choose_device

                choice = choose_device(configs[0].device)
                print(
                    json.dumps(
                        {
                            "training_started": False,
                            "device": asdict(choice),
                            **dict(
                                zip(
                                    ("training", "environment", "agent"),
                                    (asdict(c) for c in configs),
                                    strict=True,
                                )
                            ),
                        },
                        indent=2,
                    )
                )
            else:
                from .training import run_training

                run_training(
                    *configs, output=args.output, resume=args.resume, render_every=args.render_every
                )
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, pickle.UnpicklingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print(
                "Install dependencies with: uv sync --extra cuda (or --extra cpu)", file=sys.stderr
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
