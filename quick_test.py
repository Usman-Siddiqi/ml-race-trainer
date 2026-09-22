"""Read-only runtime diagnostics. This script never runs training."""

from race_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["doctor"]))
