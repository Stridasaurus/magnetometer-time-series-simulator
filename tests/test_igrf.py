import numpy as np
import pandas as pd
import pytest

ppigrf = pytest.importorskip("ppigrf", reason="ppigrf not installed")

from magsim import SimulatorConfig, TimeSeriesSimulator
from magsim.igrf import igrf_at_sensors, igrf_xyz


TEST_DATE = pd.Timestamp("2020-01-01")


def test_igrf_xyz_shape():
    result = igrf_xyz(54.7, 246.7, 0.0, TEST_DATE)
    assert result.shape == (3,)


def test_igrf_xyz_magnitude_range():
    result = igrf_xyz(54.7, 246.7, 0.0, TEST_DATE)
    magnitude = np.linalg.norm(result)
    assert 20_000 < magnitude < 70_000, f"IGRF magnitude {magnitude:.0f} nT outside expected range"


def test_igrf_at_sensors_shape(tiny_sensors):
    result = igrf_at_sensors(tiny_sensors, TEST_DATE)
    assert result.shape == (len(tiny_sensors), 3)


def test_igrf_at_sensors_magnitude(tiny_sensors):
    result = igrf_at_sensors(tiny_sensors, TEST_DATE)
    magnitudes = np.linalg.norm(result, axis=1)
    assert np.all(magnitudes > 20_000), "Some IGRF magnitudes too small"
    assert np.all(magnitudes < 70_000), "Some IGRF magnitudes too large"


def test_apply_igrf_requires_date(tiny_sensors, reference_config):
    sim = TimeSeriesSimulator(tiny_sensors, reference_config)
    with pytest.raises(ValueError, match="igrf_date"):
        sim.generate_batch(5, apply_igrf=True)


def test_apply_igrf_changes_output(tiny_sensors):
    config_base = SimulatorConfig(
        sensor_positions=tiny_sensors,
        magnetic_constant=1e-7,
        random_seed=0,
        igrf_date=TEST_DATE,
    )
    sim_base = TimeSeriesSimulator(tiny_sensors, config_base)
    X_no_igrf, _ = sim_base.generate_batch(5, add_noise_flag=False, apply_igrf=False)

    sim_igrf = TimeSeriesSimulator(tiny_sensors, config_base)
    X_igrf, _ = sim_igrf.generate_batch(5, add_noise_flag=False, apply_igrf=True)

    # IGRF adds ~50,000 nT; outputs must differ
    assert not np.allclose(X_no_igrf, X_igrf)
