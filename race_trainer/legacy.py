"""Old script names remain small adapters; every training request uses one loop."""

import argparse
import sys

from .cli import main
from .config import TRACK_NAMES


def training_entry(*, visual=False):
    parser = argparse.ArgumentParser(
        description="Compatibility launcher. Prefer python -m race_trainer train."
    )
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--model", default="best_race_model.pth")
    parser.add_argument("--track", choices=TRACK_NAMES)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--render-interval", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    if not sys.argv[1:]:
        parser.print_help()
        return 0
    args = parser.parse_args()
    command = ["evaluate", "--model", args.model] if args.mode == "test" else ["train"]
    command += [
        "--episodes",
        str(args.episodes if args.episodes is not None else (5 if args.mode == "test" else 1000)),
    ]
    command += ["--device", args.device]
    if args.track:
        command += ["--tracks", args.track]
    if args.mode == "train":
        if args.visual or visual:
            command += ["--render-every", str(args.render_interval)]
        if args.dry_run:
            command += ["--dry-run"]
    elif args.dry_run:
        parser.error("--dry-run is supported for training setup only")
    return main(command)
