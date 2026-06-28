"""SensorArrayLoader — load sensor positions from files or raw arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class SensorArrayLoader:
    """Load sensor position arrays from fixed-width files or raw lat/lon arrays.

    All methods return ``(sensor_xyz, metadata)`` where:
    - ``sensor_xyz`` is an ``(n_sensors, 3)`` array of geocentric Cartesian
      coordinates in km (R_E = 6371 km by default).
    - ``metadata`` is a DataFrame preserving all original columns plus computed
      X/Y/Z columns.
    """

    _DEFAULT_COLUMNS = ["Station", "Lshell", "M_Lat", "M_Lon", "G_lat", "G_lon"]

    @classmethod
    def from_fwf(
        cls,
        filepath: str | Path,
        lat_col: str = "G_lat",
        lon_col: str = "G_lon",
        name_col: str = "Station",
        earth_radius_km: float = 6371.0,
        column_names: Optional[Sequence[str]] = None,
        skip_header_rows: int = 1,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load sensor positions from a fixed-width text file.

        Parameters
        ----------
        filepath:
            Path to the fixed-width file (e.g., L058.txt).
        lat_col:
            Column name containing geographic latitude in degrees.
        lon_col:
            Column name containing geographic longitude in degrees.
        name_col:
            Column to sort by after loading. Pass None to skip sorting.
        earth_radius_km:
            Nominal Earth radius used for geocentric Cartesian conversion.
        column_names:
            Override column names. Defaults to
            ['Station', 'Lshell', 'M_Lat', 'M_Lon', 'G_lat', 'G_lon'].
        skip_header_rows:
            Number of header rows to skip (L058.txt has a count line at row 0).
        """
        names = column_names if column_names is not None else cls._DEFAULT_COLUMNS
        df = pd.read_fwf(filepath, names=names, skiprows=skip_header_rows)
        if name_col and name_col in df.columns:
            df = df.sort_values(by=name_col).reset_index(drop=True)
        return cls._add_xyz(df, lat_col, lon_col, earth_radius_km)

    @classmethod
    def from_latlon(
        cls,
        lats: np.ndarray,
        lons: np.ndarray,
        names: Optional[Sequence[str]] = None,
        earth_radius_km: float = 6371.0,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Build sensor positions directly from latitude/longitude arrays.

        Parameters
        ----------
        lats:
            Geographic latitudes in degrees.
        lons:
            Geographic longitudes in degrees.
        names:
            Optional list of station names.
        earth_radius_km:
            Nominal Earth radius used for geocentric Cartesian conversion.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        df = pd.DataFrame({"G_lat": lats, "G_lon": lons})
        if names is not None:
            df.insert(0, "Station", names)
        return cls._add_xyz(df, "G_lat", "G_lon", earth_radius_km)

    @staticmethod
    def _add_xyz(
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        earth_radius_km: float,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        lat_rad = np.radians(df[lat_col].to_numpy(dtype=float))
        lon_rad = np.radians(df[lon_col].to_numpy(dtype=float))
        R = earth_radius_km
        df = df.copy()
        df["X"] = R * np.cos(lat_rad) * np.cos(lon_rad)
        df["Y"] = R * np.cos(lat_rad) * np.sin(lon_rad)
        df["Z"] = R * np.sin(lat_rad)
        sensor_xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float)
        return sensor_xyz, df
