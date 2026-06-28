"""Noise models for sensor readings."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import SimulatorConfig
from .types import NoiseModel


def add_noise(
    fields: np.ndarray,
    noise_model: NoiseModel,
    config: SimulatorConfig,
    rng: np.random.Generator,
    cholesky: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add noise to a flat (n_features,) field vector.

    Parameters
    ----------
    fields:
        (n_features,) clean sensor readings.
    noise_model:
        Which noise type to apply. PINK requires the time-series path; calling
        this function with PINK raises ValueError.
    config:
        Simulator config (noise_std, noise_outlier_fraction used).
    rng:
        Instance RNG — no global state is touched.
    cholesky:
        Pre-computed Cholesky factor (n_sensors, n_sensors) of the spatial
        covariance. Required for CORRELATED; ignored otherwise.

    Returns
    -------
    (n_features,) noisy field vector.
    """
    if noise_model == NoiseModel.PINK:
        raise ValueError(
            "NoiseModel.PINK requires a time axis. Use generate_time_series() "
            "instead of generate_batch() or generate_sample()."
        )

    noisy = fields.copy()
    n_sensors = len(fields) // 3

    if noise_model == NoiseModel.GAUSSIAN:
        noisy += rng.standard_normal(len(fields)) * config.noise_std

    elif noise_model == NoiseModel.UNIFORM:
        noise_range = config.noise_std * np.sqrt(3)
        noisy += rng.uniform(-noise_range, noise_range, size=len(fields))

    elif noise_model == NoiseModel.MIXED:
        noisy += rng.standard_normal(len(fields)) * config.noise_std
        n_outliers = max(1, int(len(fields) * config.noise_outlier_fraction))
        idx = rng.choice(len(fields), size=n_outliers, replace=False)
        noisy[idx] += rng.standard_normal(n_outliers) * config.noise_std * 10

    elif noise_model == NoiseModel.CORRELATED:
        if cholesky is None:
            raise ValueError(
                "cholesky matrix required for CORRELATED noise. "
                "It is built automatically in TimeSeriesSimulator.__init__."
            )
        # Each sensor's noise amplitude is spatially correlated; components are independent.
        for component in range(3):
            z = rng.standard_normal(n_sensors)
            noisy[component::3] += config.noise_std * (cholesky @ z)

    return noisy


def pink_noise_timeseries(n: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a 1-D pink (1/f) noise time series.

    Normalised so that the standard deviation matches noise_std.

    Parameters
    ----------
    n:
        Number of time steps.
    noise_std:
        Target standard deviation.
    rng:
        Instance RNG.

    Returns
    -------
    (n,) pink noise array.
    """
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0  # avoid DC divide-by-zero; we zero DC anyway
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    spectrum[0] = 0.0  # zero DC component
    pink = np.fft.irfft(spectrum, n=n)
    std = pink.std()
    if std < 1e-12:
        return np.zeros(n)
    return pink * (noise_std / std)


def build_cholesky(
    sensor_positions: np.ndarray,
    correlation_length: float,
) -> np.ndarray:
    """Pre-compute the Cholesky factor of the spatial noise covariance matrix.

    Covariance: C_ij = exp(-d_ij / L) + 1e-8 * I  (jitter for numerical PD)

    Parameters
    ----------
    sensor_positions:
        (n_sensors, 3) positions.
    correlation_length:
        L in the exponential kernel, same unit as sensor_positions.

    Returns
    -------
    (n_sensors, n_sensors) lower-triangular Cholesky factor.
    """
    D = np.linalg.norm(
        sensor_positions[:, None, :] - sensor_positions[None, :, :],
        axis=-1,
    )  # (n_sensors, n_sensors)
    C = np.exp(-D / correlation_length)
    C += 1e-8 * np.eye(len(sensor_positions))
    return np.linalg.cholesky(C)
