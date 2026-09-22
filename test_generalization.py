"""Compatibility launcher for the unified evaluate command."""

import sys

from race_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *(sys.argv[1:] or ["--help"])]))
