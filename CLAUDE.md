# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Uses the `mhd_env` conda environment.

Launch Jupyter from `base` and select the `mhd_env` kernel:
```bash
conda activate base
jupyter notebook "mag-sim.ipynb"
```

### First-time setup on a new machine
```bash
conda env create -f environment.yml
conda run -n mhd_env python -m ipykernel install --user --name=mhd_env --display-name="mhd_env"
```
`environment.yml` includes the package in editable mode (`-e .`), so no separate pip install step is needed.

## Commands

Run all tests:
```bash
conda run -n mhd_env pytest tests/
```

Run a single test file or specific test:
```bash
conda run -n mhd_env pytest tests/test_sources.py -v
conda run -n mhd_env pytest tests/test_simulator.py::test_generate_batch_shape -v
```

Run tests with coverage:
```bash
conda run -n mhd_env pytest --cov=magsim tests/
```

## Data dependency

`L058.txt` is a fixed-width file of 29 ground-based magnetometer station locations (station name, L-shell, magnetic/geographic lat-lon). It is committed to the repo. `SensorArrayLoader.from_fwf('L058.txt')` converts geographic lat/lon → geocentric Cartesian (km, R_E = 6371 km) and returns a `(29, 3)` array plus a metadata DataFrame.

## Package structure

```
magsim/
  __init__.py        # public API
  types.py           # NoiseModel, SourceType enums
  config.py          # SimulatorConfig dataclass
  sources.py         # dipole/monopole/quadrupole field functions + SOURCE_REGISTRY
  sensors.py         # SensorArrayLoader
  igrf.py            # IGRF background field via ppigrf
  noise.py           # add_noise(), pink_noise_timeseries(), build_cholesky()
  normalization.py   # NormalizationStats, fit/normalize/denormalize
  simulator.py       # TimeSeriesSimulator (orchestrates full pipeline)
tests/
  conftest.py        # shared fixtures
  test_sources.py    test_vectorization.py  test_noise.py
  test_sensors.py    test_simulator.py      test_igrf.py
  test_normalization.py
mag-sim.ipynb        # demo notebook (imports from magsim)
```

## Architecture

**`SimulatorConfig`** (dataclass in `config.py`) — all physics and generation parameters. Key fields:
- `magnetic_constant` — defaults to `1e-7` (SI, Tesla); use `1.0` for normalized/dimensionless units
- `default_source_bounds` — uniform box for random source positions (km)
- `noise_std`, `noise_outlier_fraction`, `noise_correlation_length`
- `igrf_date` (`pd.Timestamp`) — required when `apply_igrf=True`
- `sensor_gain_error_std`, `sensor_offset_error_std` — per-sensor calibration errors
- `multi_source_target` — `'centroid'` or `'all'`

**`TimeSeriesSimulator`** — core class in `simulator.py`. Generation pipeline per sample:
1. Compute clean fields from source(s) via `SOURCE_REGISTRY[source_type]`
2. Add IGRF background (opt-in, `apply_igrf=True`)
3. Apply per-sensor gain/offset calibration errors
4. Add noise (GAUSSIAN / UNIFORM / MIXED / CORRELATED; PINK only in time series)
5. Zero dropped-out sensors (exact 0.0)

Three main generation methods:
- `generate_batch(n_samples, ...)` — vectorised fast path for single dipole; loop fallback otherwise. Returns `X: (n_samples, n_sensors*3)`, `y: (n_samples, 3)`.
- `generate_time_series(n_timesteps, trajectory_func, moment_func, dt, ...)` — sequential moving source. Returns `X: (n_timesteps, n_features)`, `y: (n_timesteps, 3)`.
- `generate_sample(...)` — single sample, used internally.

**Feature layout** — each row of `X` is `[Bx₁, By₁, Bz₁, ..., Bx_N, By_N, Bz_N]` (n_sensors × 3 features).

**`NoiseModel`** enum — `GAUSSIAN`, `UNIFORM`, `MIXED`, `PINK` (time series only), `CORRELATED`.

**`SourceType`** enum — `DIPOLE`, `MONOPOLE`, `QUADRUPOLE`, `DF_SECS` (stub), `CF_SECS` (stub).

**`SensorArrayLoader`** — `from_fwf(filepath, ...)` and `from_latlon(lats, lons, ...)`. Both return `(sensor_xyz: np.ndarray (n,3), metadata: pd.DataFrame)`.

**`SOURCE_REGISTRY`** — dict mapping `SourceType → callable`. Add new source types by decorating with `@_register(SourceType.NEW_TYPE)` in `sources.py` — no other files need editing.

Datasets saved/loaded as `.npz` via `save_dataset` / `load_dataset`.

## Planned features (not yet implemented)

The following are the next natural additions for future sessions:

- **DF-SECS / CF-SECS physics** — implement Amm (1997) eqs. 6–7 in `sources.py`. Stubs exist; sources live on ionospheric shell at `R_I = earth_radius + ionospheric_height`. Reference: Amm (1997), J. Geomagn. Geoelectr. 49:947–955.
- **Ring current / Sq variation** — global background current systems (large-scale, toroidal or diurnal). Adds as an additional background term like IGRF.
- **Ground conductivity induction** — mirror-current correction via Z-ratio transfer function.
- **Event catalog replay** — load real SuperMAG data to tune simulator statistics.
