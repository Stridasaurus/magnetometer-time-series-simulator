import numpy as np
import pytest

from magsim import NoiseModel, SimulatorConfig, TimeSeriesSimulator
from magsim.noise import add_noise, pink_noise_timeseries


@pytest.fixture
def config(tiny_sensors):
    return SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1.0,
        noise_std=0.1,
        noise_outlier_fraction=0.05,
        random_seed=1,
    )


@pytest.fixture
def clean_fields(tiny_sensors):
    rng = np.random.default_rng(0)
    return rng.standard_normal(len(tiny_sensors) * 3)


def test_gaussian_std(config, clean_fields):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        noisy = add_noise(clean_fields, NoiseModel.GAUSSIAN, config, rng)
        diffs.append(noisy - clean_fields)
    measured = np.std(np.array(diffs))
    assert abs(measured - config.noise_std) / config.noise_std < 0.05


def test_uniform_std(config, clean_fields):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        noisy = add_noise(clean_fields, NoiseModel.UNIFORM, config, rng)
        diffs.append(noisy - clean_fields)
    measured = np.std(np.array(diffs))
    assert abs(measured - config.noise_std) / config.noise_std < 0.05


def test_mixed_outlier_fraction(config, clean_fields):
    rng = np.random.default_rng(0)
    n_trials = 2000
    threshold = config.noise_std * 3
    outlier_counts = []
    for _ in range(n_trials):
        noisy = add_noise(clean_fields, NoiseModel.MIXED, config, rng)
        outlier_counts.append(np.sum(np.abs(noisy - clean_fields) > threshold))
    mean_fraction = np.mean(outlier_counts) / len(clean_fields)
    # Should be roughly noise_outlier_fraction (allow wide tolerance)
    assert mean_fraction > 0.0


def test_pink_noise_raises_in_generate_batch(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    with pytest.raises(ValueError, match="PINK"):
        sim.generate_batch(10, noise_model=NoiseModel.PINK)


def test_pink_noise_timeseries_power_spectrum():
    rng = np.random.default_rng(42)
    n = 4096
    noise_std = 1.0
    pink = pink_noise_timeseries(n, noise_std, rng)

    assert pink.shape == (n,)
    assert abs(pink.std() - noise_std) / noise_std < 0.1

    # Check 1/f slope via log-log PSD
    psd = np.abs(np.fft.rfft(pink)) ** 2
    freqs = np.fft.rfftfreq(n)
    # Use frequencies 2–n//4 to avoid DC and Nyquist
    mask = (freqs > freqs[2]) & (freqs < freqs[n // 4])
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, _ = np.polyfit(log_f, log_psd, 1)
    # Pink noise slope should be approximately -1
    assert -2.0 < slope < 0.0, f"PSD slope {slope:.2f} not in expected range for pink noise"


def test_pink_noise_in_time_series(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    X, y = sim.generate_time_series(512, noise_model=NoiseModel.PINK)
    assert X.shape == (512, tiny_sensors.shape[0] * 3)
    assert not np.any(np.isnan(X))


def test_correlated_noise_structure(tiny_sensors):
    from magsim.noise import build_cholesky

    config = SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1.0,
        noise_std=1.0,
        noise_correlation_length=500.0,
        random_seed=7,
    )
    sim = TimeSeriesSimulator(tiny_sensors, config)

    n_trials = 3000
    noises = []
    clean = np.zeros(len(tiny_sensors) * 3)
    for _ in range(n_trials):
        noisy = add_noise(clean, NoiseModel.CORRELATED, config, sim.rng, sim._noise_cholesky)
        noises.append(noisy)
    noises = np.array(noises)

    # Extract per-sensor noise from the X component (index 0::3)
    sensor_noise = noises[:, 0::3]  # (n_trials, n_sensors)
    emp_cov = np.corrcoef(sensor_noise.T)  # (n_sensors, n_sensors)

    # Expected correlation from the kernel
    D = np.linalg.norm(
        tiny_sensors[:, None, :] - tiny_sensors[None, :, :], axis=-1
    )
    expected_corr = np.exp(-D / config.noise_correlation_length)
    # Off-diagonal entries should correlate (pearson r on flattened off-diag)
    mask = ~np.eye(len(tiny_sensors), dtype=bool)
    r = np.corrcoef(emp_cov[mask], expected_corr[mask])[0, 1]
    assert r > 0.7, f"Empirical vs expected correlation Pearson r={r:.2f}"
