"""Compatibility launcher for the unified watch command."""

import sys

from race_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["watch", *(sys.argv[1:] or ["--help"])]))
