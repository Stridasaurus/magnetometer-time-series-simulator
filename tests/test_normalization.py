import numpy as np
import pytest

from magsim import SimulatorConfig, TimeSeriesSimulator
from magsim.normalization import (
    NormalizationStats,
    denormalize_target,
    fit_normalization,
    normalize_inputs,
    normalize_target,
)


def test_fit_normalization_stats_shape(sim):
    X, y = sim.generate_batch(200)
    stats = fit_normalization(X, y)
    assert stats.input_mean.shape == (sim.n_features,)
    assert stats.input_std.shape == (sim.n_features,)
    assert stats.output_mean.shape == (3,)
    assert stats.output_std.shape == (3,)


def test_normalize_denormalize_round_trip(sim):
    X, y = sim.generate_batch(200)
    stats = fit_normalization(X, y)
    y_norm = normalize_target(y, stats)
    y_back = denormalize_target(y_norm, stats)
    assert np.allclose(y_back, y, atol=1e-10)


def test_normalize_inputs_round_trip(sim):
    X, y = sim.generate_batch(200)
    stats = fit_normalization(X, y)
    X_norm = normalize_inputs(X, stats)
    X_back = X_norm * stats.input_std + stats.input_mean
    assert np.allclose(X_back, X, atol=1e-10)


def test_constant_feature_no_nan(tiny_sensors, reference_config):
    """Constant feature columns (std=0) must not produce NaN after clipping."""
    X = np.ones((100, len(tiny_sensors) * 3))
    y = np.random.randn(100, 3)
    stats = fit_normalization(X, y)
    assert not np.any(np.isnan(stats.input_std))
    X_norm = normalize_inputs(X, stats)
    assert not np.any(np.isnan(X_norm))


def test_fit_normalization_on_simulator(sim):
    X, y = sim.generate_batch(300)
    stats = sim.fit_normalization(X, y)
    assert sim.norm_stats is stats
    y_norm = sim.normalize_target(y)
    y_back = sim.denormalize_target(y_norm)
    assert np.allclose(y_back, y, atol=1e-10)


def test_normalization_raises_without_fit(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    with pytest.raises(RuntimeError, match="fit_normalization"):
        sim.normalize_target(np.zeros((10, 3)))
