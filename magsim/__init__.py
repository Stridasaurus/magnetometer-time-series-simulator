"""magsim — Magnetometer time-series simulator for magnetic source localisation."""

from .config import SimulatorConfig
from .normalization import NormalizationStats, denormalize_target, fit_normalization
from .sensors import SensorArrayLoader
from .simulator import TimeSeriesSimulator
from .types import NoiseModel, SourceType

__all__ = [
    "SimulatorConfig",
    "TimeSeriesSimulator",
    "NoiseModel",
    "SourceType",
    "SensorArrayLoader",
    "NormalizationStats",
    "fit_normalization",
    "denormalize_target",
]
