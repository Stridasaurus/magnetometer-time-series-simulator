from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulatorConfig:
    """All physics and generation parameters for TimeSeriesSimulator.

    Parameters
    ----------
    sensor_positions:
        (n_sensors, 3) geocentric Cartesian coordinates in km.
    magnetic_constant:
        mu_0/(4pi). Use 1e-7 for SI output (Tesla), 1.0 for normalized units.
    default_source_bounds:
        (min, max) uniform box for random source positions, same unit as sensor_positions.
    default_moment_range:
        (min, max) uniform range for magnetic moment magnitude.
    noise_std:
        Base standard deviation for noise models.
    noise_outlier_fraction:
        Fraction of sensor readings to corrupt with large outliers (MIXED model).
    noise_correlation_length:
        Spatial correlation length (km) for CORRELATED noise. Covariance ~ exp(-d/L).
    random_seed:
        Seed for the instance RNG. Does NOT set the global numpy seed.
    normalize_outputs:
        If True and norm_stats is set on the simulator, normalize targets in generate_*.
    output_bounds:
        (min, max) for simple linear output normalization. Defaults to default_source_bounds.
    sensor_metadata:
        Optional DataFrame from SensorArrayLoader (columns include G_lat, G_lon).
    igrf_date:
        pd.Timestamp for IGRF background field. Required when apply_igrf=True.
    sensor_gain_error_std:
        Per-sensor multiplicative gain drawn from N(1, std). 0 = no gain error.
    sensor_offset_error_std:
        Per-sensor additive offset drawn from N(0, std). 0 = no offset error.
    quadrupole_separation:
        Distance (km) between the two antiparallel dipoles comprising a quadrupole.
    quadrupole_axis:
        Unit vector along which the quadrupole dipoles are offset from the source centre.
    ionospheric_height:
        Height (km) of the ionospheric shell for SECS sources.
    earth_radius:
        Nominal Earth radius (km) used for SECS and IGRF.
    multi_source_target:
        'centroid' — y is the centroid of all source positions (shape stays (3,));
        'all'      — y is (n_sources, 3), generate_batch returns (n_samples, n_sources, 3).
    """

    sensor_positions: np.ndarray
    magnetic_constant: float = 1e-7
    default_source_bounds: Tuple[float, float] = (-5.0, 5.0)
    default_moment_range: Tuple[float, float] = (0.5, 2.0)
    noise_std: float = 0.01
    noise_outlier_fraction: float = 0.01
    noise_correlation_length: float = 1000.0
    random_seed: Optional[int] = 42
    normalize_outputs: bool = False
    output_bounds: Optional[Tuple[float, float]] = None
    sensor_metadata: Optional[pd.DataFrame] = field(default=None, compare=False, repr=False)
    igrf_date: Optional[pd.Timestamp] = None
    sensor_gain_error_std: float = 0.0
    sensor_offset_error_std: float = 0.0
    quadrupole_separation: float = 1.0
    quadrupole_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0]),
        compare=False,
        repr=False,
    )
    ionospheric_height: float = 110.0
    earth_radius: float = 6371.0
    multi_source_target: str = "centroid"

    # Computed, not user-supplied
    n_sensors: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sensor_positions.ndim != 2 or self.sensor_positions.shape[1] != 3:
            raise ValueError(
                f"sensor_positions must be (n_sensors, 3), got {self.sensor_positions.shape}"
            )
        self.n_sensors = len(self.sensor_positions)
        if self.output_bounds is None:
            self.output_bounds = self.default_source_bounds
        if self.multi_source_target not in ("centroid", "all"):
            raise ValueError("multi_source_target must be 'centroid' or 'all'")
