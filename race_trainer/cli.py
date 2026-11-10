"""One command surface for the desktop app, simulator, learner, and evaluation."""

import argparse
import json
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .config import (
    TEST_SEED_BASE,
    TRACK_NAMES,
    VALIDATION_SEED_BASE,
    AgentConfig,
    EnvConfig,
    TrainConfig,
    check_track,
    heldout_tracks,
)

DEFAULT_WATCH_TRACK = f"random:{TEST_SEED_BASE}"
TRAIN_FLAGS = {
    "training": ("total_steps", "num_envs", "track_pool", "seed", "eval_every", "eval_episodes")
    + ("log_every", "steering_noise", "device", "cpu_threads"),
    "environment": ("max_steps",),
    "agent": ("batch_size", "learning_rate", "learning_starts", "exploration_steps"),
}


def track_argument(value):
    try:
        return check_track(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def training_track_argument(value):
    try:
        return check_track(value, allow_pool=True)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{exc}, or 'random' for the generated pool") from None


def parser():
    result = argparse.ArgumentParser(
        prog="race-trainer", description="Race Lab: drive, train, and evaluate"
    )
    sub = result.add_subparsers(dest="command")
    sub.add_parser("app", help="Open the menu; never starts training automatically")
    sub.add_parser("doctor", help="Read runtime and GPU metadata without a benchmark")
    sub.add_parser("tracks", help="List the built-in layouts and the generated-track seed ranges")
    benchmark = sub.add_parser(
        "benchmark", help="Measure simulator, learner, and training speed here (under a minute)"
    )
    benchmark.add_argument("--seconds", type=float, default=1.0, help="Time per measurement")
    manual = sub.add_parser("manual", help="Drive with keyboard controls")
    manual.add_argument("--track", type=track_argument, default="oval")
    manual.add_argument("--seed", type=int, default=42)
    render = sub.add_parser(
        "render", help="Save a CPU-rendered track preview; no policy or training"
    )
    render.add_argument("--track", type=track_argument, default="oval")
    render.add_argument("--output", type=Path, default=Path("artifacts/track-preview.png"))
    render.add_argument("--seed", type=int, default=42)
    watch = sub.add_parser("watch", help="Watch a trained model drive any track")
    watch.add_argument("--model", type=Path, required=True)
    watch.add_argument(
        "--track",
        type=track_argument,
        default=DEFAULT_WATCH_TRACK,
        help="Built-in name or random:<seed>; defaults to a layout never used in training",
    )
    watch.add_argument("--episodes", type=int, default=1)
    watch.add_argument("--seed", type=int, default=42)
    watch.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    evaluate = sub.add_parser(
        "evaluate", help="Score a model on tracks it was not trained on (headless)"
    )
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument(
        "--tracks",
        nargs="+",
        type=track_argument,
        help="Explicit tracks; default is the built-ins plus unseen generated layouts",
    )
    evaluate.add_argument(
        "--generated", type=int, default=20, help="Unseen generated layouts in the default set"
    )
    evaluate.add_argument("--episodes", type=int, default=1, help="Start variations per track")
    evaluate.add_argument("--seed", type=int, default=100000)
    evaluate.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    evaluate.add_argument("--output", type=Path, help="Write the full JSON report here")
    evaluate.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    train = sub.add_parser("train", help="Explicitly start or resume headless training")
    train.add_argument(
        "--steps",
        dest="total_steps",
        type=int,
        help="Total environment steps, including steps already completed",
    )
    train.add_argument("--envs", dest="num_envs", type=int, help="Cars simulated in parallel")
    train.add_argument(
        "--tracks",
        nargs="+",
        type=training_track_argument,
        help="'random' (generated pool, the default), built-in names, or random:<seed>",
    )
    train.add_argument("--track-pool", type=int, help="Generated layouts in the 'random' pool")
    train.add_argument("--device", choices=("cpu", "cuda", "auto"))
    train.add_argument("--seed", type=int)
    train.add_argument("--cpu-threads", type=int, help="PyTorch CPU threads; 0 picks a default")
    train.add_argument("--max-steps", type=int, help="Time limit of one episode, in steps")
    train.add_argument(
        "--steering-noise",
        type=float,
        help="Random steering error while training, as a fraction of full lock; 0 disables",
    )
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--learning-starts", type=int)
    train.add_argument("--exploration-steps", type=int)
    train.add_argument("--eval-every", type=int, help="Steps between unseen-track evaluations")
    train.add_argument("--eval-episodes", type=int)
    train.add_argument("--log-every", type=int, help="Steps between progress lines")
    train.add_argument(
        "--output", type=Path, help="New run directory; existing directories are never overwritten"
    )
    train.add_argument("--resume", type=Path, help="Version 3 checkpoint to continue from")
    train.add_argument("--json", action="store_true", help="Print events as JSON lines")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration without creating an agent or training",
    )
    return result


def training_configuration(args):
    payload = None
    if args.resume:
        allowed = ("total_steps", "device", "cpu_threads")
        flags = [name for group in TRAIN_FLAGS.values() for name in group if name not in allowed]
        if args.tracks or any(getattr(args, key) is not None for key in flags):
            raise ValueError(
                "Resume preserves saved parameters; only steps, device, CPU threads, "
                "and output may be overridden"
            )
        from .checkpoint import load_checkpoint

        payload = load_checkpoint(args.resume)
        train_cfg = TrainConfig(**payload["train_config"])
        env_cfg = EnvConfig(**payload["env_config"])
        agent_cfg = AgentConfig(**payload["agent"]["config"])
    else:
        train_cfg, env_cfg, agent_cfg = TrainConfig(), EnvConfig(), AgentConfig()

    def chosen(group):
        return {k: getattr(args, k) for k in TRAIN_FLAGS[group] if getattr(args, k) is not None}

    train_updates = chosen("training")
    if args.tracks:
        train_updates["tracks"] = tuple(dict.fromkeys(args.tracks))
    train_cfg = replace(train_cfg, **train_updates)
    env_cfg = replace(env_cfg, **chosen("environment"))
    agent_cfg = replace(agent_cfg, **chosen("agent"))
    if agent_cfg.replay_capacity < train_cfg.num_envs:
        raise ValueError("Replay capacity must hold at least one step of every environment")
    if payload and payload["completed_steps"] >= train_cfg.total_steps:
        raise ValueError("--steps must exceed the number already completed")
    return train_cfg, env_cfg, agent_cfg


def _list_tracks():
    from .tracks import make_track

    for name in TRACK_NAMES:
        track = make_track(name)
        print(
            f"{name:8}  {'circuit' if track.closed else 'sprint':7}  "
            f"{len(track.gates):2} gates  {track.length:.0f} units"
        )
    pool = TrainConfig().track_pool
    print(
        "\nrandom:<seed> is a generated layout; every seed is a different track.\n"
        f"  random:0 .. random:{pool - 1}   default training pool (train --tracks random)\n"
        f"  random:{VALIDATION_SEED_BASE}+  validation; picks best.pt during training\n"
        f"  random:{TEST_SEED_BASE}+  unseen test layouts used by evaluate and watch\n"
        "The three built-in layouts are also held out of default training."
    )


def _evaluate(args, policy, payload, choice):
    from .evaluation import evaluate_policy

    env_cfg = EnvConfig(**payload["env_config"])
    tracks = tuple(dict.fromkeys(args.tracks or heldout_tracks(args.generated)))
    report = evaluate_policy(policy, env_cfg, tracks, args.episodes, args.seed)
    trained_on = set(TrainConfig(**payload["train_config"]).training_specs())
    report["seen_in_training"] = sorted(set(tracks) & trained_on)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.json:
        shown = {key: report[key] for key in ("summary", "tracks", "seen_in_training")}
        print(json.dumps({"device": choice.name, **shown}, indent=2, allow_nan=False))
        return

    def row(label, seen, metrics, digits=0):
        seconds = metrics["mean_finish_seconds"]
        time = "     -" if seconds is None else f"{seconds:5.1f}s"
        return (
            f"{label:24} {seen:8} {metrics['completion_rate']:6.{digits}%} "
            f"{metrics['crash_rate']:6.{digits}%} {metrics['mean_progress']:9.{digits}%} "
            f"{time:>7} {metrics['mean_speed']:6.0f}"
        )

    print(f"Model: {args.model} ({payload['completed_steps']:,} training steps)")
    print(f"{'Track':24} {'Trained':8} {'Finish':>6} {'Crash':>6} {'Progress':>9} {'Lap':>7} Speed")
    for name in tracks:
        print(row(name, "yes" if name in trained_on else "no", report["tracks"][name]))
    unseen = [r for r in report["episodes"] if r["track"] not in trained_on]
    if unseen:
        from .evaluation import summarize

        label = f"UNSEEN ({len({r['track'] for r in unseen})} tracks)"
        print(row(label, "no", summarize(unseen), 1))
    if report["seen_in_training"]:
        print(row("ALL", "mixed", report["summary"], 1))
    # Totals are stated as counts, and every failure is named so it can be watched.
    failed = [r for r in report["episodes"] if not r["completed"]]
    total = len(report["episodes"])
    print(f"Finished {total - len(failed)} of {total} runs.")
    for record in failed[:10]:
        print(
            f"  {record['termination_reason']} at {record['progress']:.0%} of {record['track']}: "
            f"watch --track {record['track']} --seed {record['seed']}"
        )
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more in the full report")
    if args.output:
        print(f"Full report: {args.output.resolve()}")


def _require_model(path):
    """Fail early on a missing model and point at the checkpoints that do exist."""
    if path.is_file():
        return
    found = sorted(Path("runs").glob("*/best.pt"))
    hint = (
        "Models found: " + ", ".join(str(p) for p in found)
        if found
        else "No runs\\*\\best.pt found; train one first with: train --output runs\\my-run"
    )
    raise FileNotFoundError(f"Model not found: {path}. {hint}")


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
            _list_tracks()
        elif args.command == "benchmark":
            if not 0 < args.seconds <= 10:
                raise ValueError("seconds must be between 0 and 10 per measurement")
            from .benchmark import run_benchmark

            run_benchmark(args.seconds)
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
            if args.command == "evaluate" and args.generated < 0:
                raise ValueError("generated must be zero or positive")
            from .checkpoint import load_policy
            from .device import choose_device

            _require_model(args.model)
            choice = choose_device(args.device)
            policy, payload = load_policy(args.model, choice.name)
            if args.command == "watch":
                from .viewer import play

                # Laps are a property of the saved physics; sprints always finish once.
                env_cfg = replace(EnvConfig(**payload["env_config"]), track=args.track, laps=1)
                print(f"Replay: {args.track} | {choice.reason}")
                play(env_cfg, policy, seed=args.seed, episodes=args.episodes)
            else:
                _evaluate(args, policy, payload, choice)
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
                    *configs, output=args.output, resume=args.resume, json_events=args.json
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
