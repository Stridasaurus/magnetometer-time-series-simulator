import numpy as np
import pytest

from magsim import SensorArrayLoader


def test_from_fwf_shape(L058_path):
    sensors, meta = SensorArrayLoader.from_fwf(L058_path)
    assert sensors.shape == (29, 3)


def test_from_fwf_no_nan(L058_path):
    sensors, meta = SensorArrayLoader.from_fwf(L058_path)
    assert not np.any(np.isnan(sensors))
    assert not np.any(np.isinf(sensors))


def test_from_fwf_metadata_columns(L058_path):
    _, meta = SensorArrayLoader.from_fwf(L058_path)
    for col in ("G_lat", "G_lon", "Station", "X", "Y", "Z"):
        assert col in meta.columns, f"Missing column: {col}"


def test_from_fwf_sorted_by_station(L058_path):
    _, meta = SensorArrayLoader.from_fwf(L058_path)
    stations = meta["Station"].tolist()
    assert stations == sorted(stations)


def test_from_latlon_round_trip():
    lats = np.array([0.0, 45.0, -45.0, 90.0])
    lons = np.array([0.0, 90.0, 180.0, 270.0])
    R = 6371.0
    sensors, meta = SensorArrayLoader.from_latlon(lats, lons, earth_radius_km=R)
    radii = np.linalg.norm(sensors, axis=1)
    assert np.allclose(radii, R, atol=1e-6)


def test_from_latlon_with_names():
    lats = np.array([50.0, 60.0])
    lons = np.array([10.0, 20.0])
    sensors, meta = SensorArrayLoader.from_latlon(lats, lons, names=["A", "B"])
    assert "Station" in meta.columns
    assert list(meta["Station"]) == ["A", "B"]


def test_from_latlon_no_nan():
    lats = np.linspace(-80, 80, 10)
    lons = np.linspace(-180, 180, 10)
    sensors, _ = SensorArrayLoader.from_latlon(lats, lons)
    assert not np.any(np.isnan(sensors))
