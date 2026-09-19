# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metrics and confidence intervals used by the SAC experiments."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_interval(
    values: Sequence[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 0,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    draws = array[indices]
    if statistic == "mean":
        stats = np.mean(draws, axis=1)
    elif statistic == "median":
        stats = np.median(draws, axis=1)
    else:
        raise ValueError("Unsupported bootstrap statistic: {}".format(statistic))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def summarize_errors(errors: Sequence[float], hit_radius: float, seed: int = 0) -> Dict[str, float]:
    array = np.asarray(errors, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return {
            "n": 0,
            "successes": 0,
            "success_rate": float("nan"),
            "success_ci_low": float("nan"),
            "success_ci_high": float("nan"),
            "mean_error": float("nan"),
            "mean_error_ci_low": float("nan"),
            "mean_error_ci_high": float("nan"),
            "median_error": float("nan"),
            "median_error_ci_low": float("nan"),
            "median_error_ci_high": float("nan"),
            "std_error": float("nan"),
        }
    successes = int(np.sum(valid <= hit_radius))
    success_ci = wilson_interval(successes, int(valid.size))
    mean_ci = bootstrap_interval(valid, "mean", seed=seed)
    median_ci = bootstrap_interval(valid, "median", seed=seed + 1)
    return {
        "n": int(valid.size),
        "successes": successes,
        "success_rate": float(successes / valid.size),
        "success_ci_low": success_ci[0],
        "success_ci_high": success_ci[1],
        "mean_error": float(np.mean(valid)),
        "mean_error_ci_low": mean_ci[0],
        "mean_error_ci_high": mean_ci[1],
        "median_error": float(np.median(valid)),
        "median_error_ci_low": median_ci[0],
        "median_error_ci_high": median_ci[1],
        "std_error": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
    }


def normalized_auc(x: Iterable[float], y: Iterable[float], log_x: bool = False) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if x_arr.size < 2:
        return float("nan")
    if log_x:
        x_arr = np.log(np.maximum(x_arr, 1.0))
    width = x_arr[-1] - x_arr[0]
    if width <= 0:
        return float("nan")
    return float(np.trapz(y_arr, x_arr) / width)
