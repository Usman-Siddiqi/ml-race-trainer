"""Compatibility launcher for the unified evaluate command."""

import sys

from race_trainer.cli import main
from race_trainer.config import TRACK_NAMES

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and "--tracks" not in arguments:
        arguments += ["--tracks", *TRACK_NAMES]
    raise SystemExit(main(["evaluate", *(arguments or ["--help"])]))
