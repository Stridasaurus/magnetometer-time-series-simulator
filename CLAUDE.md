# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Uses the `mhd-env` conda kernel. Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`.

Launch Jupyter from `base` and select the `mhd-env` kernel:
```bash
conda activate base
jupyter notebook "mag-sim.ipynb"
```

### First-time setup on a new machine
```bash
conda run -n mhd-env pip install numpy pandas scipy matplotlib ipykernel
conda run -n mhd-env python -m ipykernel install --user --name=mhd-env --display-name="mhd-env"
```

## Data dependency

`L058.txt` is a fixed-width file of 29 ground-based magnetometer station locations (station name, L-shell, magnetic/geographic lat-lon). It is committed to the repo. The notebook converts geographic lat/lon → 3D Cartesian (km, R_E = 6371 km) to produce the `(29, 3)` sensor position array.

## Architecture

All simulator code lives in the first notebook cell. The remaining cells load sensor data, run the simulator, and visualize results.

**`SimulatorConfig`** (dataclass) — holds all physics and generation parameters. Key fields: `magnetic_constant` (`1.0` for normalized units, `1e-7` for SI), `default_source_bounds`, `noise_std`.

**`TimeSeriesSimulator`** — the core class. Three data generation modes:
- `generate_batch(n_samples)` — independent static-source samples; returns `X: (n_samples, n_sensors*3)`, `y: (n_samples, 3)`
- `generate_time_series(n_timesteps, trajectory_func, moment_func, dt)` — sequential moving-source data; callables are `t → (3,)` array
- `generate_sample(...)` — single sample, used internally by the above

**Feature layout** — each row of `X` is `[Bx₁, By₁, Bz₁, ..., Bx₂₉, By₂₉, Bz₂₉]` (87 features for 29 sensors). Target `y` is always the 3D source position.

**`NoiseModel`** enum — `GAUSSIAN`, `UNIFORM`, `MIXED`. `MIXED` adds Gaussian background plus sparse large outliers (controlled by `noise_outlier_fraction`).

Datasets are saved/loaded as `.npz` via `save_dataset` / `load_dataset` (classmethod).

## Planned features (not yet implemented)

The following additions are planned for the next development session. All changes go into cell 1 of `mag-sim.ipynb`, plus cell 2 (replace manual L058.txt loading with `SensorArrayLoader`).

### 1. New source types

Extend `SourceType` enum with `DF_SECS` and `CF_SECS` (split from the existing stub). Implement physics methods and route through `generate_sample(source_type=...)`:

**Monopole** — scalar strength, field falls off as `1/r²`:
```python
B = magnetic_constant * strength * r_vec / r**3
```

**Quadrupole** — implemented as two antiparallel dipoles separated by `quadrupole_separation` along `quadrupole_axis`. Requires two new `SimulatorConfig` fields.

**DF-SECS** (Divergence-Free Spherical Elementary Current System) — sources on ionospheric shell at R_I = R_E + h_I (default h_I = 110 km). Purely horizontal ground field. Physics: Amm (1997) eq. 6. New config fields: `ionospheric_height`, `earth_radius`.

**CF-SECS** (Curl-Free) — field-aligned currents; has both horizontal and radial ground components. Physics: Amm (1997) eq. 7.

Reference: Amm (1997) "Ionospheric elementary current systems in spherical coordinates and their application", J. Geomagn. Geoelectr. 49:947–955.

### 2. `SensorArrayLoader` utility class

Generalizes the manual L058.txt loading into a reusable class:
```python
SensorArrayLoader.from_fwf(filepath, lat_col, lon_col, ...)  # fixed-width files
SensorArrayLoader.from_latlon(lats, lons, names=None)        # raw arrays
```
Returns `(sensor_xyz: np.ndarray (n,3), metadata: pd.DataFrame)`. Replace cell 2 loading code with this.

### 3. Vectorize hot loops

- `compute_fields` — replace Python loop over sensors with numpy broadcasting across all sensors at once
- `generate_batch` — batch all random draws and field computations

### 4. Cleanup
- Remove unused `from scipy.spatial.transform import Rotation`
- Replace global `np.random.seed()` with instance-level `np.random.default_rng(seed)` passed through all random calls
- Remove unfilled `input_mean/std/output_mean/std` placeholders or implement `fit_normalization(X, y)`
- Add `denormalize_target(y_norm)` inverse of existing normalization
