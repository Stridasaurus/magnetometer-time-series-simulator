"""TimeSeriesSimulator — the main data-generation orchestrator."""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import SimulatorConfig
from .igrf import igrf_at_sensors
from .noise import add_noise, build_cholesky, pink_noise_timeseries
from .normalization import (
    NormalizationStats,
    denormalize_target,
    fit_normalization,
    normalize_inputs,
    normalize_target,
)
from .sources import SOURCE_REGISTRY, compute_fields, dipole_field_scalar
from .types import NoiseModel, SourceType


class TimeSeriesSimulator:
    """Simulator for magnetometer time-series data from magnetic sources.

    Generates synthetic sensor readings with configurable physics, noise, and
    hardware error models for training and evaluating source-localisation algorithms.

    Generation pipeline (applied in this order for every sample):
      1. Compute clean fields from one or more sources (superposition).
      2. Add IGRF background field per sensor (opt-in).
      3. Apply per-sensor calibration gain and offset errors.
      4. Add noise (Gaussian / Uniform / Mixed / Correlated / Pink-in-time-series).
      5. Zero out dropped-out sensors (last, so dead sensors read exactly 0).

    Examples
    --------
    >>> import numpy as np
    >>> from magsim import TimeSeriesSimulator, SimulatorConfig, SensorArrayLoader
    >>> sensors, meta = SensorArrayLoader.from_fwf("L058.txt")
    >>> config = SimulatorConfig(sensor_positions=sensors, magnetic_constant=1.0,
    ...                          default_source_bounds=(-3, 3), random_seed=42)
    >>> sim = TimeSeriesSimulator(sensors, config)
    >>> X, y = sim.generate_batch(n_samples=1000)
    """

    def __init__(
        self,
        sensor_positions: np.ndarray,
        config: Optional[SimulatorConfig] = None,
    ) -> None:
        if config is None:
            config = SimulatorConfig(sensor_positions=sensor_positions)
        else:
            # Validate only; do NOT call __post_init__ again (avoid side effects)
            if np.any(np.isnan(sensor_positions)) or np.any(np.isinf(sensor_positions)):
                raise ValueError("sensor_positions contains NaN or Inf values")

        self.config = config
        self.sensor_positions = np.asarray(sensor_positions, dtype=float)
        self.n_sensors = config.n_sensors
        self.n_features = self.n_sensors * 3

        self._validate_sensor_positions()

        # Instance RNG — never touches global numpy state
        self.rng = np.random.default_rng(config.random_seed)

        # Per-sensor calibration errors (drawn once at init)
        self.sensor_gains = self.rng.normal(1.0, config.sensor_gain_error_std, self.n_sensors)
        self.sensor_offsets = self.rng.normal(0.0, config.sensor_offset_error_std, self.n_sensors)

        # Correlated noise Cholesky (built once at init)
        self._noise_cholesky: Optional[np.ndarray] = None
        if config.noise_correlation_length > 0:
            self._noise_cholesky = build_cholesky(
                self.sensor_positions, config.noise_correlation_length
            )

        # Normalisation stats (populated by fit_normalization)
        self.norm_stats: Optional[NormalizationStats] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_sensor_positions(self) -> None:
        if np.any(np.isnan(self.sensor_positions)):
            raise ValueError("sensor_positions contains NaN values")
        if np.any(np.isinf(self.sensor_positions)):
            raise ValueError("sensor_positions contains Inf values")
        unique = np.unique(np.round(self.sensor_positions, decimals=6), axis=0)
        if len(unique) < self.n_sensors:
            warnings.warn(
                f"{self.n_sensors - len(unique)} duplicate sensor positions detected."
            )

    def _generate_source_params(
        self, source_type: SourceType
    ) -> Tuple[np.ndarray, Union[np.ndarray, float]]:
        """Draw a random source position and parameters for the given source type."""
        low, high = self.config.default_source_bounds
        source_pos = self.rng.uniform(low, high, size=3)

        if source_type == SourceType.MONOPOLE:
            mag_low, mag_high = self.config.default_moment_range
            strength = float(self.rng.uniform(mag_low, mag_high))
            return source_pos, strength
        else:
            # DIPOLE, QUADRUPOLE, etc. — use a moment vector
            mag_low, mag_high = self.config.default_moment_range
            magnitude = float(self.rng.uniform(mag_low, mag_high))
            direction = self.rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            return source_pos, direction * magnitude

    def _apply_calibration(self, fields: np.ndarray) -> np.ndarray:
        """Apply per-sensor gain and offset (step 3 in pipeline)."""
        gains = np.repeat(self.sensor_gains, 3)
        offsets = np.repeat(self.sensor_offsets, 3)
        return fields * gains + offsets

    def _apply_dropout(self, fields: np.ndarray, dropout_rate: float) -> np.ndarray:
        """Zero out a random fraction of sensors (step 5 in pipeline)."""
        if dropout_rate <= 0.0:
            return fields
        mask = self.rng.random(self.n_sensors) < dropout_rate
        fields = fields.copy()
        fields[np.repeat(mask, 3)] = 0.0
        return fields

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sample(
        self,
        source_pos: Optional[np.ndarray] = None,
        source_params: Optional[Union[np.ndarray, float]] = None,
        source_type: SourceType = SourceType.DIPOLE,
        add_noise_flag: bool = True,
        noise_model: NoiseModel = NoiseModel.GAUSSIAN,
        apply_igrf: bool = False,
        n_sources: int = 1,
        dropout_rate: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single sample.

        Parameters
        ----------
        source_pos:
            (3,) source position. Randomised if None.
        source_params:
            Moment vector (3,) for DIPOLE/QUADRUPOLE, scalar strength for MONOPOLE.
            Randomised if None.
        source_type:
            Which source physics to use.
        add_noise_flag:
            Whether to add noise.
        noise_model:
            Noise model. PINK raises ValueError (requires time axis).
        apply_igrf:
            Add IGRF background field. Requires config.igrf_date to be set.
        n_sources:
            Number of simultaneous sources (superposition).
        dropout_rate:
            Fraction of sensors to zero out (simulates outages).

        Returns
        -------
        fields : (n_features,)
        target : (3,) if n_sources==1 or multi_source_target=='centroid';
                 (n_sources, 3) if multi_source_target=='all'.
        """
        config = self.config

        # --- Step 1: compute clean fields from one or more sources ---
        if n_sources == 1:
            if source_pos is None or source_params is None:
                source_pos, source_params = self._generate_source_params(source_type)
            fields = compute_fields(
                self.sensor_positions, source_pos, source_params, config, source_type
            )
            target = source_pos.copy()
        else:
            all_positions = []
            fields = np.zeros(self.n_features)
            for _ in range(n_sources):
                sp, pm = self._generate_source_params(source_type)
                fields += compute_fields(
                    self.sensor_positions, sp, pm, config, source_type
                )
                all_positions.append(sp)
            all_positions_arr = np.array(all_positions)  # (n_sources, 3)
            if config.multi_source_target == "centroid":
                target = all_positions_arr.mean(axis=0)
            else:
                target = all_positions_arr

        # --- Step 2: add IGRF background ---
        if apply_igrf:
            if config.igrf_date is None:
                raise ValueError(
                    "config.igrf_date must be set (as pd.Timestamp) to use apply_igrf=True."
                )
            igrf_bg = igrf_at_sensors(self.sensor_positions, config.igrf_date)  # (n_sensors, 3)
            fields = fields + igrf_bg.reshape(-1)

        # --- Step 3: calibration errors ---
        fields = self._apply_calibration(fields)

        # --- Step 4: noise ---
        if add_noise_flag:
            fields = add_noise(
                fields, noise_model, config, self.rng, self._noise_cholesky
            )

        # --- Step 5: dropout ---
        fields = self._apply_dropout(fields, dropout_rate)

        # Optional output normalisation
        if config.normalize_outputs and self.norm_stats is not None:
            target = normalize_target(target, self.norm_stats)

        return fields, target

    def generate_batch(
        self,
        n_samples: int,
        add_noise_flag: bool = True,
        noise_model: NoiseModel = NoiseModel.GAUSSIAN,
        source_type: SourceType = SourceType.DIPOLE,
        apply_igrf: bool = False,
        n_sources: int = 1,
        dropout_rate: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of independent samples.

        Uses a fully vectorised fast path for the common case
        (single dipole source, no IGRF, no dropout, Gaussian or Uniform noise).
        All other combinations fall through to a loop over generate_sample().

        Returns
        -------
        X : (n_samples, n_features)
        y : (n_samples, 3) normally;
            (n_samples, n_sources, 3) when n_sources > 1 and
            config.multi_source_target == 'all'.
        """
        config = self.config

        if noise_model == NoiseModel.PINK:
            raise ValueError(
                "NoiseModel.PINK requires a time axis. Use generate_time_series()."
            )

        use_fast_path = (
            source_type == SourceType.DIPOLE
            and n_sources == 1
            and not apply_igrf
            and dropout_rate == 0.0
            and noise_model in (NoiseModel.GAUSSIAN, NoiseModel.UNIFORM, NoiseModel.MIXED)
        )

        if use_fast_path:
            return self._generate_batch_vectorized(n_samples, add_noise_flag, noise_model)

        # General loop path
        samples_X = []
        samples_y = []
        for _ in range(n_samples):
            x, y = self.generate_sample(
                source_type=source_type,
                add_noise_flag=add_noise_flag,
                noise_model=noise_model,
                apply_igrf=apply_igrf,
                n_sources=n_sources,
                dropout_rate=dropout_rate,
            )
            samples_X.append(x)
            samples_y.append(y)
        return np.array(samples_X), np.array(samples_y)

    def _generate_batch_vectorized(
        self,
        n_samples: int,
        add_noise_flag: bool,
        noise_model: NoiseModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fully vectorised batch generation for the single-dipole common case."""
        config = self.config
        low, high = config.default_source_bounds
        mag_low, mag_high = config.default_moment_range

        # Draw all sources at once
        source_positions = self.rng.uniform(low, high, (n_samples, 3))
        magnitudes = self.rng.uniform(mag_low, mag_high, (n_samples, 1))
        directions = self.rng.standard_normal((n_samples, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        magnetic_moments = directions * magnitudes  # (n_samples, 3)

        # Vectorised dipole field: (n_samples, n_sensors, 3)
        r_vec = self.sensor_positions[None, :, :] - source_positions[:, None, :]
        r = np.linalg.norm(r_vec, axis=2, keepdims=True)  # (n_samples, n_sensors, 1)
        safe = r > 1e-9
        r_safe = np.where(safe, r, 1.0)
        r_hat = r_vec / r_safe
        m = magnetic_moments[:, None, :]  # (n_samples, 1, 3)
        m_dot_r = (m * r_hat).sum(axis=2, keepdims=True)
        B = config.magnetic_constant * (3 * r_hat * m_dot_r - m) / r_safe**3
        B = np.where(safe, B, 0.0)
        X = B.reshape(n_samples, -1)  # (n_samples, n_features)

        # Apply calibration
        gains = np.repeat(self.sensor_gains, 3)
        offsets = np.repeat(self.sensor_offsets, 3)
        X = X * gains[None, :] + offsets[None, :]

        # Add noise
        if add_noise_flag:
            if noise_model == NoiseModel.GAUSSIAN:
                X += self.rng.standard_normal(X.shape) * config.noise_std
            elif noise_model == NoiseModel.UNIFORM:
                noise_range = config.noise_std * np.sqrt(3)
                X += self.rng.uniform(-noise_range, noise_range, size=X.shape)
            elif noise_model == NoiseModel.MIXED:
                X += self.rng.standard_normal(X.shape) * config.noise_std
                n_outliers = max(1, int(X.shape[1] * config.noise_outlier_fraction))
                for i in range(n_samples):
                    idx = self.rng.choice(X.shape[1], size=n_outliers, replace=False)
                    X[i, idx] += self.rng.standard_normal(n_outliers) * config.noise_std * 10

        y = source_positions  # (n_samples, 3)
        return X, y

    def generate_time_series(
        self,
        n_timesteps: int,
        trajectory_func: Optional[Callable[[float], np.ndarray]] = None,
        moment_func: Optional[Callable[[float], np.ndarray]] = None,
        dt: float = 0.01,
        add_noise_flag: bool = True,
        noise_model: NoiseModel = NoiseModel.GAUSSIAN,
        source_type: SourceType = SourceType.DIPOLE,
        apply_igrf: bool = False,
        dropout_rate: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a time series with a moving source.

        Parameters
        ----------
        n_timesteps:
            Number of time steps.
        trajectory_func:
            ``t -> (3,)`` position callable. Defaults to bounded random walk.
        moment_func:
            ``t -> (3,)`` magnetic moment callable. Defaults to constant random moment.
        dt:
            Time step size in seconds.
        add_noise_flag:
            Whether to add noise.
        noise_model:
            Noise model. PINK applies 1/f noise across the time axis after computing
            all clean fields.
        source_type:
            Source physics.
        apply_igrf:
            Add IGRF background. Requires config.igrf_date.
        dropout_rate:
            Fraction of sensors to zero per timestep.

        Returns
        -------
        X : (n_timesteps, n_features)
        y : (n_timesteps, 3) source positions
        """
        config = self.config
        times = np.arange(n_timesteps) * dt

        # Build position trajectory
        if trajectory_func is None:
            low, high = config.default_source_bounds
            positions = np.zeros((n_timesteps, 3))
            positions[0] = self.rng.uniform(low, high, 3)
            step_size = 0.1
            for t in range(1, n_timesteps):
                positions[t] = positions[t - 1] + self.rng.standard_normal(3) * step_size
                for dim in range(3):
                    if positions[t, dim] < low:
                        positions[t, dim] = low + (low - positions[t, dim])
                    elif positions[t, dim] > high:
                        positions[t, dim] = high - (positions[t, dim] - high)
        else:
            positions = np.array([trajectory_func(t) for t in times])

        # Build moment trajectory
        if moment_func is None:
            _, moment = self._generate_source_params(source_type)
            moments = np.tile(moment, (n_timesteps, 1))
        else:
            moments = np.array([moment_func(t) for t in times])

        # Pre-compute IGRF background (same for all timesteps)
        igrf_bg_flat = None
        if apply_igrf:
            if config.igrf_date is None:
                raise ValueError(
                    "config.igrf_date must be set (as pd.Timestamp) to use apply_igrf=True."
                )
            igrf_bg_flat = igrf_at_sensors(self.sensor_positions, config.igrf_date).reshape(-1)

        # Generate clean fields for each timestep
        X = np.zeros((n_timesteps, self.n_features))
        for i in range(n_timesteps):
            fields = compute_fields(
                self.sensor_positions, positions[i], moments[i], config, source_type
            )
            if igrf_bg_flat is not None:
                fields = fields + igrf_bg_flat
            fields = self._apply_calibration(fields)
            X[i] = fields

        # Add noise
        if add_noise_flag:
            if noise_model == NoiseModel.PINK:
                for feat in range(self.n_features):
                    X[:, feat] += pink_noise_timeseries(n_timesteps, config.noise_std, self.rng)
            else:
                for i in range(n_timesteps):
                    X[i] = add_noise(
                        X[i], noise_model, config, self.rng, self._noise_cholesky
                    )

        # Dropout per timestep
        if dropout_rate > 0.0:
            for i in range(n_timesteps):
                X[i] = self._apply_dropout(X[i], dropout_rate)

        return X, positions

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def fit_normalization(self, X: np.ndarray, y: np.ndarray) -> NormalizationStats:
        """Compute and store normalisation statistics from training data."""
        self.norm_stats = fit_normalization(X, y)
        return self.norm_stats

    def normalize_inputs(self, X: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            raise RuntimeError("Call fit_normalization first.")
        return normalize_inputs(X, self.norm_stats)

    def normalize_target(self, y: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            raise RuntimeError("Call fit_normalization first.")
        return normalize_target(y, self.norm_stats)

    def denormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            raise RuntimeError("Call fit_normalization first.")
        return denormalize_target(y_norm, self.norm_stats)

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def compute_snr(self, clean_fields: np.ndarray, noisy_fields: np.ndarray) -> float:
        """Signal-to-Noise Ratio in dB."""
        signal_power = float(np.mean(clean_fields ** 2))
        noise_power = float(np.mean((clean_fields - noisy_fields) ** 2))
        if noise_power < 1e-12:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)

    def get_data_statistics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Summary statistics for a generated dataset."""
        return {
            "input": {
                "mean": np.mean(X, axis=0).tolist(),
                "std": np.std(X, axis=0).tolist(),
                "min": np.min(X, axis=0).tolist(),
                "max": np.max(X, axis=0).tolist(),
                "has_nan": bool(np.any(np.isnan(X))),
                "has_inf": bool(np.any(np.isinf(X))),
            },
            "output": {
                "mean": np.mean(y, axis=0).tolist(),
                "std": np.std(y, axis=0).tolist(),
                "min": np.min(y, axis=0).tolist(),
                "max": np.max(y, axis=0).tolist(),
                "has_nan": bool(np.any(np.isnan(y))),
                "has_inf": bool(np.any(np.isinf(y))),
            },
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_sensor_array(self, ax=None):
        """Plot the 3-D sensor positions. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Install with: pip install magsim[viz]")
            return None

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            self.sensor_positions[:, 0],
            self.sensor_positions[:, 1],
            self.sensor_positions[:, 2],
            c="red", marker="o", s=50, label="Sensors",
        )
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title(f"Sensor Array ({self.n_sensors} sensors)")
        ax.legend()

        spans = [np.ptp(self.sensor_positions[:, d]) for d in range(3)]
        half = max(spans) / 2.0
        centres = [np.mean(self.sensor_positions[:, d]) for d in range(3)]
        ax.set_xlim(centres[0] - half, centres[0] + half)
        ax.set_ylim(centres[1] - half, centres[1] + half)
        ax.set_zlim(centres[2] - half, centres[2] + half)
        return ax

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_dataset(self, X: np.ndarray, y: np.ndarray, filepath: str) -> None:
        """Save a dataset to compressed .npz format."""
        config = self.config
        serialisable = {
            k: v for k, v in vars(config).items()
            if not isinstance(v, (np.ndarray, type(None), pd.DataFrame))
            and k != "n_sensors"
        }
        np.savez_compressed(
            filepath,
            X=X,
            y=y,
            sensor_positions=self.sensor_positions,
            config_scalar=str(serialisable),
        )
        print(f"Dataset saved to {filepath}")

    @classmethod
    def load_dataset(cls, filepath: str) -> dict:
        """Load a dataset saved by save_dataset."""
        data = np.load(filepath, allow_pickle=True)
        return {
            "X": data["X"],
            "y": data["y"],
            "sensor_positions": data["sensor_positions"],
        }
