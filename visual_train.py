"""Compatibility launcher; no arguments prints help without training."""

from race_trainer.legacy import training_entry

if __name__ == "__main__":
    raise SystemExit(training_entry(visual=True))
