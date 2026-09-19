# SPDX-License-Identifier: AGPL-3.0-or-later
"""Target-domain sampling and coordinate normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TargetDomain:
    l_min: float
    l_max: float
    gamma_max: float
    z: float

    @classmethod
    def from_config(cls, values: Dict[str, float]) -> "TargetDomain":
        domain = cls(
            l_min=float(values["l_min"]),
            l_max=float(values["l_max"]),
            gamma_max=float(values["gamma_max"]),
            z=float(values["z"]),
        )
        if domain.l_min <= 0 or domain.l_max <= domain.l_min:
            raise ValueError("Expected 0 < l_min < l_max")
        if domain.gamma_max <= 0:
            raise ValueError("gamma_max must be positive")
        return domain

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        radius = rng.uniform(self.l_min, self.l_max)
        gamma = rng.uniform(-self.gamma_max, self.gamma_max)
        return np.asarray(
            [radius * np.cos(gamma), radius * np.sin(gamma), self.z],
            dtype=np.float64,
        )

    def sample_many(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return np.stack([self.sample(rng) for _ in range(int(count))], axis=0)

    @property
    def cartesian_low(self) -> np.ndarray:
        return np.asarray(
            [self.l_min * np.cos(self.gamma_max), -self.l_max * np.sin(self.gamma_max), self.z],
            dtype=np.float64,
        )

    @property
    def cartesian_high(self) -> np.ndarray:
        return np.asarray(
            [self.l_max, self.l_max * np.sin(self.gamma_max), self.z],
            dtype=np.float64,
        )

    def contains(self, target: Sequence[float], atol: float = 1e-8) -> bool:
        target_arr = np.asarray(target, dtype=np.float64)
        radius = np.linalg.norm(target_arr[:2])
        gamma = np.arctan2(target_arr[1], target_arr[0])
        return bool(
            self.l_min - atol <= radius <= self.l_max + atol
            and -self.gamma_max - atol <= gamma <= self.gamma_max + atol
            and abs(target_arr[2] - self.z) <= atol
        )

    def normalize_cartesian(self, target: Sequence[float]) -> np.ndarray:
        target_arr = np.asarray(target, dtype=np.float64)
        low = self.cartesian_low
        high = self.cartesian_high
        scale = high - low
        normalized = np.zeros(3, dtype=np.float64)
        nonzero = np.abs(scale) > 1e-12
        normalized[nonzero] = 2.0 * (target_arr[nonzero] - low[nonzero]) / scale[nonzero] - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def normalize_radius(self, target: Sequence[float]) -> np.ndarray:
        """Map planar target radius to the normalized one-dimensional input."""
        target_arr = np.asarray(target, dtype=np.float64)
        radius = float(np.linalg.norm(target_arr[:2]))
        normalized = 2.0 * (radius - self.l_min) / (self.l_max - self.l_min) - 1.0
        return np.asarray([np.clip(normalized, -1.0, 1.0)], dtype=np.float32)
