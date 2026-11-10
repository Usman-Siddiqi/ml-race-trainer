"""Old script names remain small adapters; every training request uses one loop."""

import argparse
import sys

from .cli import main, track_argument, training_track_argument


def training_entry(*, visual=False):
    parser = argparse.ArgumentParser(
        description="Compatibility launcher. Prefer python -m race_trainer train."
    )
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--steps", type=int, help="Total training steps")
    parser.add_argument("--episodes", type=int, help="Test mode: start variations per track")
    parser.add_argument("--model", default="runs/latest/best.pt")
    parser.add_argument("--track", help="Built-in name, random:<seed>, or 'random' to train")
    parser.add_argument("--visual", action="store_true", help="Ignored: training is headless")
    parser.add_argument("--render-interval", type=int, help="Ignored: training is headless")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    if not sys.argv[1:]:
        parser.print_help()
        return 0
    args = parser.parse_args()
    if args.mode == "test":
        if args.dry_run or args.steps is not None:
            parser.error("--dry-run and --steps are for training")
        command = ["evaluate", "--model", args.model, "--device", args.device]
        command += ["--episodes", str(args.episodes or 1)]
        if args.track:
            command += ["--tracks", track_argument(args.track)]
        return main(command)
    if args.episodes is not None:
        parser.error("Training is measured in steps now; use --steps instead of --episodes")
    if visual or args.visual or args.render_interval is not None:
        print("Training runs headless. Watch progress with: python -m race_trainer watch --model")
    command = ["train", "--device", args.device]
    if args.steps is not None:
        command += ["--steps", str(args.steps)]
    if args.track:
        command += ["--tracks", training_track_argument(args.track)]
    if args.dry_run:
        command += ["--dry-run"]
    return main(command)
