# Magnetometer Time Series Simulator

**Version:** 1.0.0  
**Author:** Strider Settgast

This module provides a high-fidelity simulator for generating synthetic magnetometer data from dipole sources. It is specifically designed for generating training data for neural networks focused on magnetic source localization.

---

## 1. Enumerations

### `SourceType`
Types of magnetic sources that can be simulated.

- `DIPOLE` (`"dipole"`): Standard magnetic dipole.
- `MONOPOLE` (`"monopole"`): Not physically real but useful for testing/baselines.
- `QUADRUPOLE` (`"quadrupole"`): Planned for future extension.

### `NoiseModel`
Available noise models for sensor simulation.

- `GAUSSIAN` (`"gaussian"`): Standard normally distributed noise.
- `UNIFORM` (`"uniform"`): Uniformly distributed noise matching the variance of the Gaussian model.
- `MIXED` (`"mixed"`): A combination of Gaussian noise and extreme outliers (designed for robust testing).

---

## 2. Configuration: `SimulatorConfig`

A dataclass used to configure the physics and generation settings of the simulator.

### Parameters / Attributes

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `sensor_positions` | `np.ndarray` | **Required** | Array of shape `(n_sensors, 3)` representing sensor coordinates (X, Y, Z) in meters. |
| `magnetic_constant` | `float` | `1e-7` | $\frac{\mu_0}{4\pi}$ in SI units. Set to `1.0` for normalized output. |
| `default_source_bounds` | `Tuple[float, float]` | `(-5.0, 5.0)` | `[min, max]` boundaries for the random source position cube. |
| `default_moment_range` | `Tuple[float, float]` | `(0.5, 2.0)` | `[min, max]` boundaries for the magnetic moment magnitude. |
| `noise_std` | `float` | `0.01` | Standard deviation for the Gaussian noise model. |
| `noise_outlier_fraction` | `float` | `0.01` | Fraction of data points that will be generated as outliers when using the `MIXED` noise model. |
| `random_seed` | `Optional[int]` | `42` | Seed for numpy's random number generator to ensure reproducibility. |
| `normalize_outputs` | `bool` | `False` | If `True`, normalizes the target output positions to a `[-1, 1]` range based on `output_bounds`. |
| `output_bounds` | `Optional[Tuple]` | `None` | Bounds used for output normalization. **Note:** If left as `None`, it automatically defaults to the values set in `default_source_bounds` upon initialization. |

### Computed Attributes (Generated after initialization)

- `n_sensors` (`int`): Automatically calculated from the length of `sensor_positions`.

---

## 3. Main Class: `TimeSeriesSimulator`

The core class that generates realistic magnetic field readings from moving or static dipole sources.

### Initialization

```python
sim = TimeSeriesSimulator(sensor_positions: np.ndarray, config: Optional[SimulatorConfig] = None)
```
## Instance Attributes (Available after initialization)

- `sensor_positions` (np.ndarray): Pre-computed array of sensor locations.
- `n_sensors` (int): Number of sensors.
- `n_features` (int): Total number of features generated per time step (`n_sensors * 3`, representing Bx, By, Bz per sensor).
- `config` (SimulatorConfig): The active configuration object.
- `input_mean` / `input_std` / `output_mean` / `output_std`: Placeholders (initialized to `None`) intended for online normalization statistics.

---

## Core Physics & Generation Methods

### `dipole_field(sensor_pos, source_pos, magnetic_moment)`

Calculates the theoretical magnetic field from a dipole source at a single specific sensor location.

**Parameters:**
- `sensor_pos` (np.ndarray): `(3,)` array of sensor position `[x, y, z]`
- `source_pos` (np.ndarray): `(3,)` array of source position `[x, y, z]`
- `magnetic_moment` (np.ndarray): `(3,)` array of magnetic moment vector `[mx, my, mz]`

**Returns:** np.ndarray of shape `(3,)` representing the magnetic field `[Bx, By, Bz]`

---

### `compute_fields(source_pos, magnetic_moment)`

Computes the clean, noiseless magnetic fields across all sensors simultaneously.

**Parameters:**
- `source_pos` (np.ndarray): `(3,)` array
- `magnetic_moment` (np.ndarray): `(3,)` array

**Returns:** np.ndarray of shape `(n_features,)` — flattened array of Bx, By, Bz for all sensors

---

### `generate_source()`

Generates a random magnetic source based on the bounding boxes defined in the configuration.

**Returns:** `Tuple[np.ndarray, np.ndarray]` containing `(source_position, magnetic_moment)`. Position is uniformly distributed within bounds; moment direction follows a uniform sphere distribution.

---

## Data Generation Pipelines

### `generate_sample(source_pos=None, magnetic_moment=None, add_noise=True, noise_model=NoiseModel.GAUSSIAN)`

Generates a single paired training sample (features and target).

**Parameters:**
- `source_pos` / `magnetic_moment` (Optional[np.ndarray]): If `None`, randomly generated
- `add_noise` (bool): Toggle noise application. Default: `True`
- `noise_model` (NoiseModel): Which noise distribution to use

**Returns:** `Tuple[np.ndarray, np.ndarray]` containing `(features, target)`. 
- `features` shape: `(n_features,)`
- `target` shape: `(3,)`

---

### `generate_batch(n_samples, add_noise=True, noise_model=NoiseModel.GAUSSIAN)`

Generates a dataset of independent, static samples.

**Parameters:**
- `n_samples` (int): Number of independent samples to create

**Returns:** `Tuple[np.ndarray, np.ndarray]` containing:
- Features matrix `X` of shape `(n_samples, n_features)`
- Targets matrix `y` of shape `(n_samples, 3)`

---

### `generate_time_series(n_timesteps, trajectory_func=None, moment_func=None, dt=0.01, add_noise=True, noise_model=NoiseModel.GAUSSIAN)`

Generates sequential time-series data mimicking a moving source.

**Parameters:**
- `n_timesteps` (int): Number of sequential data points to generate
- `trajectory_func` (callable, optional): Function taking time `t` and returning a `(3,)` position array. If `None`, defaults to a random walk constrained by source bounds
- `moment_func` (callable, optional): Function taking time `t` and returning a `(3,)` moment array. If `None`, defaults to a constant random moment
- `dt` (float): Time step delta in seconds. Default: `0.01`

**Returns:** `Tuple[np.ndarray, np.ndarray]` containing:
- Time series `X` of shape `(n_timesteps, n_features)`
- Time series `y` of shape `(n_timesteps, 3)`

---

## Utility, Analysis, and I/O Methods

### `add_noise(fields, noise_model=NoiseModel.GAUSSIAN)`

Applies noise to a clean field array based on the specified noise model enum.

**Returns:** np.ndarray of noisy field readings

---

### `compute_snr(clean_fields, noisy_fields)`

Calculates the Signal-to-Noise Ratio (SNR) for the generated data.

**Returns:** float representing the SNR in decibels (dB). Returns `inf` if noise is identically zero.

---

### `get_data_statistics(X, y)`

Computes comprehensive statistical metadata about a generated dataset.

**Returns:** Dict containing `mean`, `std`, `min`, `max`, `has_nan`, and `has_inf` for both the input (`X`) and output (`y`) arrays.

---

### `visualize_sensor_array(ax=None)`

Plots a 3D visualization of the sensor setup. Requires matplotlib.

**Parameters:**
- `ax`: An optional matplotlib 3D axis. If `None`, creates a new figure and axis

**Returns:** The matplotlib axis object (`ax`)

---

### `save_dataset(X, y, filepath)`

Saves generated features, targets, sensor positions, and configuration to a compressed numpy array file.

**Parameters:**
- `filepath` (str): File destination (should end in `.npz`)

---

### `load_dataset(filepath)` (classmethod)

A class method to easily load datasets generated by `save_dataset`.

**Returns:** Dict containing keys: `'X'`, `'y'`, `'sensor_positions'`, and `'config'`
