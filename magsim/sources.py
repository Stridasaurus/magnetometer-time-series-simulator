"""Magnetic field physics functions and source type registry."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from .config import SimulatorConfig
from .types import SourceType

SOURCE_REGISTRY: Dict[SourceType, Callable] = {}


def _register(source_type: SourceType):
    def decorator(fn: Callable) -> Callable:
        SOURCE_REGISTRY[source_type] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Reference scalar oracle — kept for testing vectorised paths against
# ---------------------------------------------------------------------------

def dipole_field_scalar(
    sensor_pos: np.ndarray,
    source_pos: np.ndarray,
    magnetic_moment: np.ndarray,
    magnetic_constant: float,
) -> np.ndarray:
    """Return (3,) dipole field at one sensor. Used as correctness oracle."""
    r_vec = sensor_pos - source_pos
    r = float(np.linalg.norm(r_vec))
    if r < 1e-9:
        return np.zeros(3)
    r_hat = r_vec / r
    m_dot_r = float(np.dot(magnetic_moment, r_hat))
    return magnetic_constant * (3 * r_hat * m_dot_r - magnetic_moment) / r**3


# ---------------------------------------------------------------------------
# Vectorised source functions — all return (n_sensors, 3)
# ---------------------------------------------------------------------------

@_register(SourceType.DIPOLE)
def dipole_field_vectorized(
    sensor_positions: np.ndarray,
    source_pos: np.ndarray,
    source_params: np.ndarray,
    config: SimulatorConfig,
) -> np.ndarray:
    """Vectorised dipole field for all sensors simultaneously.

    Parameters
    ----------
    sensor_positions : (n_sensors, 3)
    source_pos : (3,)
    source_params : (3,) magnetic moment vector
    config : SimulatorConfig

    Returns
    -------
    (n_sensors, 3)
    """
    magnetic_moment = source_params
    r_vec = sensor_positions - source_pos                          # (n_sensors, 3)
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)              # (n_sensors, 1)
    safe = r > 1e-9
    r_safe = np.where(safe, r, 1.0)
    r_hat = r_vec / r_safe                                         # (n_sensors, 3)
    m_dot_r = (magnetic_moment * r_hat).sum(axis=1, keepdims=True) # (n_sensors, 1)
    B = config.magnetic_constant * (3 * r_hat * m_dot_r - magnetic_moment) / r_safe**3
    return np.where(safe, B, 0.0)                                  # (n_sensors, 3)


@_register(SourceType.MONOPOLE)
def monopole_field_vectorized(
    sensor_positions: np.ndarray,
    source_pos: np.ndarray,
    source_params: float,
    config: SimulatorConfig,
) -> np.ndarray:
    """Vectorised monopole field. Not physically real; useful for algorithm testing.

    Field: B = k * strength * r_vec / r^3  (falls off as 1/r^2)
    """
    strength = float(source_params)
    r_vec = sensor_positions - source_pos
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    safe = r > 1e-9
    r_safe = np.where(safe, r, 1.0)
    B = config.magnetic_constant * strength * r_vec / r_safe**3
    return np.where(safe, B, 0.0)


@_register(SourceType.QUADRUPOLE)
def quadrupole_field_vectorized(
    sensor_positions: np.ndarray,
    source_pos: np.ndarray,
    source_params: np.ndarray,
    config: SimulatorConfig,
) -> np.ndarray:
    """Two antiparallel dipoles offset along config.quadrupole_axis.

    source_params : (3,) magnetic moment of the positive dipole
    """
    magnetic_moment = source_params
    axis = config.quadrupole_axis / np.linalg.norm(config.quadrupole_axis)
    offset = 0.5 * config.quadrupole_separation * axis
    pos_plus = source_pos + offset
    pos_minus = source_pos - offset
    B_plus = dipole_field_vectorized(sensor_positions, pos_plus, magnetic_moment, config)
    B_minus = dipole_field_vectorized(sensor_positions, pos_minus, -magnetic_moment, config)
    return B_plus + B_minus


@_register(SourceType.DF_SECS)
def df_secs_field_vectorized(
    sensor_positions: np.ndarray,
    source_pos: np.ndarray,
    source_params: Any,
    config: SimulatorConfig,
) -> np.ndarray:
    raise NotImplementedError(
        "DF-SECS not yet implemented. See Amm (1997) eq. 6, "
        "J. Geomagn. Geoelectr. 49:947-955."
    )


@_register(SourceType.CF_SECS)
def cf_secs_field_vectorized(
    sensor_positions: np.ndarray,
    source_pos: np.ndarray,
    source_params: Any,
    config: SimulatorConfig,
) -> np.ndarray:
    raise NotImplementedError(
        "CF-SECS not yet implemented. See Amm (1997) eq. 7, "
        "J. Geomagn. Geoelectr. 49:947-955."
    )


def compute_fields(
    sensor_positions: np.ndarray,
    source_pos: np.ndarray,
    source_params: Any,
    config: SimulatorConfig,
    source_type: SourceType = SourceType.DIPOLE,
) -> np.ndarray:
    """Dispatch to the correct source function and return flat (n_features,) vector."""
    fn = SOURCE_REGISTRY[source_type]
    B = fn(sensor_positions, source_pos, source_params, config)  # (n_sensors, 3)
    return B.reshape(-1)                                          # (n_features,)
