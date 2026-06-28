from pathlib import Path

import numpy as np
import pytest

from magsim import SimulatorConfig, TimeSeriesSimulator


@pytest.fixture
def tiny_sensors() -> np.ndarray:
    """5-sensor geocentric Cartesian array in km, no duplicates."""
    return np.array([
        [1000.0,     0.0, 6000.0],
        [-1000.0,    0.0, 6000.0],
        [0.0,     1000.0, 6000.0],
        [0.0,    -1000.0, 6000.0],
        [500.0,    300.0, 6200.0],  # off-pole to avoid IGRF singularity
    ])


@pytest.fixture
def reference_config(tiny_sensors) -> SimulatorConfig:
    return SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1.0,
        default_source_bounds=(-100.0, 100.0),
        noise_std=0.01,
        random_seed=0,
    )


@pytest.fixture
def sim(tiny_sensors, reference_config) -> TimeSeriesSimulator:
    return TimeSeriesSimulator(tiny_sensors, reference_config)


@pytest.fixture
def L058_path() -> Path:
    return Path(__file__).parent.parent / "L058.txt"
