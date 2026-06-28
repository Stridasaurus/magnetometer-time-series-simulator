"""IGRF background field helpers.

Uses ppigrf.igrf_gc (geocentric coordinates) so that sensor positions in
geocentric Cartesian XYZ km can be used directly without needing geodetic
lat/lon metadata.

Important unit caveat
---------------------
IGRF returns fields in nT. The simulator's dipole fields depend on
``magnetic_constant``:
  - ``magnetic_constant = 1e-7`` (SI): dipole fields are in Tesla.
  - ``magnetic_constant = 1.0``  (normalised): dipole fields are dimensionless.

``apply_igrf=True`` is only physically meaningful when using SI units
(``magnetic_constant=1e-7``) with sources positioned in real geocentric km
from the Earth's centre (~6371 km from the origin). Mixing normalised units
with IGRF nT values produces unphysical results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def igrf_at_sensors(
    sensor_positions: np.ndarray,
    date: pd.Timestamp,
) -> np.ndarray:
    """Compute IGRF field at each sensor, in geocentric Cartesian nT.

    Parameters
    ----------
    sensor_positions:
        (n_sensors, 3) geocentric Cartesian positions in km.
    date:
        Evaluation date as a pd.Timestamp.

    Returns
    -------
    (n_sensors, 3) IGRF field vectors in geocentric Cartesian nT.
    """
    import ppigrf  # deferred so the rest of magsim works without ppigrf installed

    positions = np.asarray(sensor_positions, dtype=float)
    r = np.linalg.norm(positions, axis=1)                # (n_sensors,)
    # Clip cos to [-1, 1] to avoid arccos domain error from floating-point rounding
    cos_theta = np.clip(positions[:, 2] / r, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))             # colatitude deg
    # Nudge exact poles (theta=0 or 180) by a tiny epsilon to avoid ppigrf singularity
    theta = np.clip(theta, 1e-4, 180.0 - 1e-4)
    phi = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))  # longitude deg

    # ppigrf.igrf_gc returns arrays of shape (n_dates, n_sensors)
    Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, date)
    Br = np.asarray(Br[0], dtype=float)      # (n_sensors,)
    Btheta = np.asarray(Btheta[0], dtype=float)
    Bphi = np.asarray(Bphi[0], dtype=float)

    # Convert geocentric spherical (Br, Btheta, Bphi) → Cartesian XYZ
    # where theta is colatitude and phi is longitude.
    # r_hat     = (sin θ cos φ, sin θ sin φ, cos θ)
    # theta_hat = (cos θ cos φ, cos θ sin φ, -sin θ)   [south direction]
    # phi_hat   = (-sin φ,       cos φ,        0)       [east direction]
    th = np.radians(theta)
    ph = np.radians(phi)
    Bx = Br * np.sin(th) * np.cos(ph) + Btheta * np.cos(th) * np.cos(ph) - Bphi * np.sin(ph)
    By = Br * np.sin(th) * np.sin(ph) + Btheta * np.cos(th) * np.sin(ph) + Bphi * np.cos(ph)
    Bz = Br * np.cos(th)               - Btheta * np.sin(th)

    return np.column_stack([Bx, By, Bz])  # (n_sensors, 3)


def igrf_xyz(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    date: pd.Timestamp,
) -> np.ndarray:
    """Convenience wrapper for a single geographic location.

    Uses a spherical-Earth approximation (not WGS84 ellipsoid) consistent
    with the geocentric Cartesian used throughout magsim.

    Parameters
    ----------
    lat_deg:
        Geographic latitude in degrees.
    lon_deg:
        Geographic longitude in degrees.
    alt_km:
        Altitude above surface in km.
    date:
        Evaluation date as a pd.Timestamp.

    Returns
    -------
    (3,) IGRF field in geocentric Cartesian nT.
    """
    R = 6371.0 + alt_km
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    xyz = np.array([[
        R * np.cos(lat_r) * np.cos(lon_r),
        R * np.cos(lat_r) * np.sin(lon_r),
        R * np.sin(lat_r),
    ]])
    return igrf_at_sensors(xyz, date)[0]
