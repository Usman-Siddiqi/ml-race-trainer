"""Compatibility entry point for the Race Lab desktop menu."""

from race_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["app"]))
