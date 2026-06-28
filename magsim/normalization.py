from dataclasses import dataclass
import numpy as np


@dataclass
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray


def fit_normalization(X: np.ndarray, y: np.ndarray) -> NormalizationStats:
    input_std = np.std(X, axis=0)
    input_std = np.where(input_std < 1e-8, 1e-8, input_std)
    output_std = np.std(y, axis=0)
    output_std = np.where(output_std < 1e-8, 1e-8, output_std)
    return NormalizationStats(
        input_mean=np.mean(X, axis=0),
        input_std=input_std,
        output_mean=np.mean(y, axis=0),
        output_std=output_std,
    )


def normalize_inputs(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (X - stats.input_mean) / stats.input_std


def normalize_target(y: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (y - stats.output_mean) / stats.output_std


def denormalize_target(y_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return y_norm * stats.output_std + stats.output_mean
