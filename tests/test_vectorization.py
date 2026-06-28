import numpy as np
import pytest

from magsim import SimulatorConfig, SourceType
from magsim.sources import compute_fields, dipole_field_scalar, dipole_field_vectorized


@pytest.fixture
def config(tiny_sensors):
    return SimulatorConfig(sensor_positions=tiny_sensors, magnetic_constant=1.0)


def test_vectorized_matches_scalar_oracle(config):
    rng = np.random.default_rng(42)
    sensors = config.sensor_positions
    for _ in range(10):
        source_pos = rng.uniform(-100, 100, 3)
        moment = rng.standard_normal(3)

        # Vectorised
        B_vec = dipole_field_vectorized(sensors, source_pos, moment, config)

        # Scalar oracle loop
        B_scalar = np.array([
            dipole_field_scalar(s, source_pos, moment, config.magnetic_constant)
            for s in sensors
        ])

        assert np.allclose(B_vec, B_scalar, atol=1e-12), (
            f"Mismatch: max diff = {np.abs(B_vec - B_scalar).max()}"
        )


def test_compute_fields_shape(config):
    sensors = config.sensor_positions
    fields = compute_fields(sensors, np.zeros(3), np.array([1.0, 0.0, 0.0]), config, SourceType.DIPOLE)
    assert fields.shape == (len(sensors) * 3,)


def test_batch_vectorized_same_statistics_as_loop(tiny_sensors, reference_config):
    from magsim import TimeSeriesSimulator

    sim_vec = TimeSeriesSimulator(tiny_sensors, reference_config)
    X_vec, y_vec = sim_vec.generate_batch(500, add_noise_flag=False)

    # Same seed, same config → same RNG draws
    sim_loop = TimeSeriesSimulator(tiny_sensors, reference_config)
    X_loop, y_loop = sim_loop.generate_batch(500, add_noise_flag=False)

    # Vectorized and loop paths consume RNG draws in different orders,
    # so we can only compare statistics, not exact values.
    assert X_vec.shape == X_loop.shape == (500, tiny_sensors.shape[0] * 3)
    assert np.allclose(X_vec.mean(), X_loop.mean(), atol=0.1)
    assert np.allclose(X_vec.std(), X_loop.std(), atol=0.1)
