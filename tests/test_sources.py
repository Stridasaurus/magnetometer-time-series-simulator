import numpy as np
import pytest

from magsim import SimulatorConfig, SourceType
from magsim.sources import (
    SOURCE_REGISTRY,
    compute_fields,
    dipole_field_scalar,
    dipole_field_vectorized,
    monopole_field_vectorized,
)


@pytest.fixture
def config(tiny_sensors):
    return SimulatorConfig(sensor_positions=tiny_sensors, magnetic_constant=1.0)


def test_dipole_zeros_at_singularity(config):
    sensors = config.sensor_positions
    source_pos = sensors[0].copy()
    B = dipole_field_vectorized(sensors, source_pos, np.array([1.0, 0.0, 0.0]), config)
    assert np.allclose(B[0], 0.0)


def test_dipole_distance_law(config):
    # Place a sensor on the z-axis so geometry is self-similar at both distances,
    # allowing the clean 1/r^3 check (don't rely on fixture sensor positions).
    sensor_axis = np.array([[0.0, 0.0, 1.0]])
    moment = np.array([1.0, 0.0, 0.0])  # transverse moment → clean geometry
    source_near = np.array([0.0, 0.0, 0.0])
    r1, r2 = 2.0, 4.0

    sensor1 = np.array([[r1, 0.0, 0.0]])
    sensor2 = np.array([[r2, 0.0, 0.0]])

    B1 = np.linalg.norm(dipole_field_vectorized(sensor1, source_near, moment, config)[0])
    B2 = np.linalg.norm(dipole_field_vectorized(sensor2, source_near, moment, config)[0])

    # field ratio should be exactly (r2/r1)^3
    expected_ratio = (r2 / r1) ** 3
    actual_ratio = B1 / B2
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10


def test_monopole_distance_law(config):
    sensors = config.sensor_positions
    source_near = np.array([0.0, 0.0, 5500.0])
    source_far = np.array([0.0, 0.0, 5000.0])

    r_near = np.linalg.norm(sensors[4] - source_near)
    r_far = np.linalg.norm(sensors[4] - source_far)

    B_near = np.linalg.norm(monopole_field_vectorized(sensors, source_near, 1.0, config)[4])
    B_far = np.linalg.norm(monopole_field_vectorized(sensors, source_far, 1.0, config)[4])

    expected_ratio = (r_far / r_near) ** 2
    actual_ratio = B_near / B_far
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.05


def test_dipole_superposition(config):
    sensors = config.sensor_positions
    moment = np.array([1.0, 0.0, 0.0])
    src1 = np.array([50.0, 0.0, 0.0])
    src2 = np.array([-50.0, 0.0, 0.0])

    B1 = compute_fields(sensors, src1, moment, config, SourceType.DIPOLE)
    B2 = compute_fields(sensors, src2, moment, config, SourceType.DIPOLE)
    B_both = B1 + B2

    # Build the "both" manually
    f1 = dipole_field_vectorized(sensors, src1, moment, config)
    f2 = dipole_field_vectorized(sensors, src2, moment, config)
    expected = (f1 + f2).reshape(-1)

    assert np.allclose(B_both, expected, atol=1e-14)


def test_source_registry_has_implemented_types():
    for st in [SourceType.DIPOLE, SourceType.MONOPOLE, SourceType.QUADRUPOLE,
               SourceType.DF_SECS, SourceType.CF_SECS]:
        assert st in SOURCE_REGISTRY, f"{st} not in SOURCE_REGISTRY"


def test_secs_stubs_raise(config):
    sensors = config.sensor_positions
    with pytest.raises(NotImplementedError):
        compute_fields(sensors, np.zeros(3), None, config, SourceType.DF_SECS)
    with pytest.raises(NotImplementedError):
        compute_fields(sensors, np.zeros(3), None, config, SourceType.CF_SECS)
