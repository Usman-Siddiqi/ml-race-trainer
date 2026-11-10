"""Compatibility launcher for the unified evaluate command.

With only --model it scores the checkpoint on the built-in layouts plus generated
tracks that were never used in training.
"""

import sys

from race_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *(sys.argv[1:] or ["--help"])]))
