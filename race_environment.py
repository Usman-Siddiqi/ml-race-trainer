"""Version 3 environment exports. See README for the sensors-only observation contract."""

from race_trainer.config import EnvConfig
from race_trainer.environment import RaceCarEnv, VectorRaceEnv

__all__ = ["EnvConfig", "RaceCarEnv", "VectorRaceEnv"]
