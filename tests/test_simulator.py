import numpy as np
import pytest

from magsim import NoiseModel, SimulatorConfig, SourceType, TimeSeriesSimulator


def test_rng_reproducibility(tiny_sensors, reference_config):
    sim1 = TimeSeriesSimulator(tiny_sensors, reference_config)
    X1, y1 = sim1.generate_batch(100)

    sim2 = TimeSeriesSimulator(tiny_sensors, reference_config)
    X2, y2 = sim2.generate_batch(100)

    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


def test_rng_different_seeds(tiny_sensors):
    cfg1 = SimulatorConfig(sensor_positions=tiny_sensors, magnetic_constant=1.0, random_seed=1)
    cfg2 = SimulatorConfig(sensor_positions=tiny_sensors, magnetic_constant=1.0, random_seed=2)
    sim1 = TimeSeriesSimulator(tiny_sensors, cfg1)
    sim2 = TimeSeriesSimulator(tiny_sensors, cfg2)
    X1, _ = sim1.generate_batch(50)
    X2, _ = sim2.generate_batch(50)
    assert not np.allclose(X1, X2)


def test_no_global_rng_pollution(tiny_sensors, reference_config):
    np.random.seed(99)
    a = np.random.rand()
    np.random.seed(99)
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    sim.generate_batch(200)
    b = np.random.rand()
    assert a == b, "TimeSeriesSimulator polluted the global numpy RNG"


def test_generate_batch_shape(sim, tiny_sensors):
    X, y = sim.generate_batch(50)
    assert X.shape == (50, tiny_sensors.shape[0] * 3)
    assert y.shape == (50, 3)


def test_generate_time_series_shape(sim, tiny_sensors):
    X, y = sim.generate_time_series(100)
    assert X.shape == (100, tiny_sensors.shape[0] * 3)
    assert y.shape == (100, 3)


def test_generate_batch_no_nan(sim):
    X, y = sim.generate_batch(200)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))
    assert not np.any(np.isnan(y))


def test_dropout_all_zeros(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    X, _ = sim.generate_batch(10, add_noise_flag=False, dropout_rate=1.0)
    assert np.all(X == 0.0)


def test_dropout_partial(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    X, _ = sim.generate_batch(500, add_noise_flag=False, dropout_rate=0.5)
    # Check that each sensor slot can be all-zero (dropped) or non-zero
    # Rough check: some sensor readings are zero
    sensor_max = np.abs(X).reshape(500, len(tiny_sensors), 3).max(axis=2)
    zero_sensors = (sensor_max == 0.0).mean()
    assert 0.3 < zero_sensors < 0.7, f"Dropout fraction {zero_sensors:.2f} out of expected range"


def test_calibration_gains_stored(tiny_sensors):
    config = SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1.0,
        sensor_gain_error_std=0.1,
        random_seed=5,
    )
    sim = TimeSeriesSimulator(tiny_sensors, config)
    assert sim.sensor_gains.shape == (len(tiny_sensors),)
    assert not np.allclose(sim.sensor_gains, 1.0)


def test_no_calibration_error_unity_gains(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    # sensor_gain_error_std defaults to 0.0 → gains should be exactly 1.0
    assert np.allclose(sim.sensor_gains, 1.0)
    assert np.allclose(sim.sensor_offsets, 0.0)


def test_multisource_centroid_shape(sim):
    fields, target = sim.generate_sample(n_sources=3)
    assert target.shape == (3,)
    assert fields.shape == (sim.n_features,)


def test_multisource_all_shape(tiny_sensors):
    config = SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1.0,
        random_seed=0,
        multi_source_target="all",
    )
    sim = TimeSeriesSimulator(tiny_sensors, config)
    _, target = sim.generate_sample(n_sources=3)
    assert target.shape == (3, 3)


def test_multisource_batch_all_shape(tiny_sensors):
    config = SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1.0,
        random_seed=0,
        multi_source_target="all",
    )
    sim = TimeSeriesSimulator(tiny_sensors, config)
    X, y = sim.generate_batch(20, n_sources=2)
    assert X.shape == (20, len(tiny_sensors) * 3)
    assert y.shape == (20, 2, 3)


def test_trajectory_func(sim):
    traj = lambda t: np.array([np.sin(t), np.cos(t), 0.0])
    X, y = sim.generate_time_series(50, trajectory_func=traj, dt=0.1)
    assert X.shape == (50, sim.n_features)
    # Confirm positions match trajectory
    times = np.arange(50) * 0.1
    expected_positions = np.array([traj(t) for t in times])
    assert np.allclose(y, expected_positions)


def test_save_load_dataset(sim, tmp_path):
    X, y = sim.generate_batch(30)
    filepath = str(tmp_path / "test_data.npz")
    sim.save_dataset(X, y, filepath)
    loaded = TimeSeriesSimulator.load_dataset(filepath)
    assert np.allclose(loaded["X"], X)
    assert np.allclose(loaded["y"], y)


def test_monopole_source_type(sim):
    X, y = sim.generate_batch(20, source_type=SourceType.MONOPOLE)
    assert X.shape == (20, sim.n_features)
    assert not np.any(np.isnan(X))


def test_quadrupole_source_type(sim):
    X, y = sim.generate_batch(20, source_type=SourceType.QUADRUPOLE)
    assert X.shape == (20, sim.n_features)
    assert not np.any(np.isnan(X))
