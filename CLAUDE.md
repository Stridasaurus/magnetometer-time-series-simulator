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

The notebook requires `L058.txt` — a fixed-width file of 29 ground-based magnetometer station locations (station name, L-shell, magnetic/geographic lat-lon). This file is **not in the repo** and must be present in the working directory before running. The notebook converts geographic lat/lon to 3D Cartesian coordinates (km, Earth radius = 6371 km) to produce the `(29, 3)` sensor position array.

## Architecture

All simulator code lives in the first notebook cell. The remaining cells load sensor data, run the simulator, and visualize results.

**`SimulatorConfig`** (dataclass) — holds all physics and generation parameters. Pass it to `TimeSeriesSimulator` to override defaults. Key fields: `magnetic_constant` (set to `1.0` for normalized units, `1e-7` for SI), `default_source_bounds`, `noise_std`.

**`TimeSeriesSimulator`** — the core class. Three data generation modes:
- `generate_batch(n_samples)` — independent static-source samples; returns `X: (n_samples, n_sensors*3)`, `y: (n_samples, 3)`
- `generate_time_series(n_timesteps, trajectory_func, moment_func, dt)` — sequential moving-source data; `trajectory_func` and `moment_func` are callables `t → (3,)` array
- `generate_sample(...)` — single sample, used internally by the above

**Feature layout** — each row of `X` is `[Bx₁, By₁, Bz₁, Bx₂, By₂, Bz₂, ..., Bx₂₉, By₂₉, Bz₂₉]` (87 features for 29 sensors). Target `y` is always the 3D source position.

**`NoiseModel`** enum — `GAUSSIAN`, `UNIFORM`, `MIXED`. `MIXED` adds Gaussian background plus sparse large outliers (controlled by `noise_outlier_fraction`).

Datasets are saved/loaded as `.npz` via `save_dataset` / `load_dataset` (classmethod).

## Known issues in the demo cell

- Typo: `1 0000` should be `10000` in `generate_batch(n_samples=1 0000, ...)`
- `y_test` is referenced in `save_dataset` but never defined (only `y_val` is generated)
