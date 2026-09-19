# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configurable release-delay sampling for the Gazebo throwing node.

The release delay is expressed in simulator/odometry callbacks and is
configurable through ROS parameters.  The current paper experiment defaults to
a discrete uniform delay of 100--200 callbacks (approximately 100--200 ms at
1 kHz).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class DelayConfig:
    enabled: bool = True
    distribution: str = "uniform"
    min_steps: int = 120
    max_steps: int = 130
    fixed_steps: int = 125
    mean_steps: float = 125.0
    std_steps: float = 2.5
    seed: int = 0

    def validate(self) -> "DelayConfig":
        if self.min_steps < 0:
            raise ValueError("min_steps must be non-negative")
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")
        if self.fixed_steps < 0:
            raise ValueError("fixed_steps must be non-negative")
        if self.std_steps < 0:
            raise ValueError("std_steps must be non-negative")
        valid = {"none", "fixed", "uniform", "truncated_normal"}
        if self.distribution not in valid:
            raise ValueError(
                "Unknown delay distribution {!r}; expected one of {}".format(
                    self.distribution, sorted(valid)
                )
            )
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DelayConfig":
        return cls(
            enabled=bool(values.get("enabled", True)),
            distribution=str(values.get("distribution", "uniform")),
            min_steps=int(values.get("min_steps", 120)),
            max_steps=int(values.get("max_steps", 130)),
            fixed_steps=int(values.get("fixed_steps", 125)),
            mean_steps=float(values.get("mean_steps", 125.0)),
            std_steps=float(values.get("std_steps", 2.5)),
            seed=int(values.get("seed", 0)),
        ).validate()


def sample_delay_steps(config: DelayConfig, rng: np.random.Generator) -> int:
    """Draw one non-negative integer delay in simulator callbacks."""
    config.validate()
    if not config.enabled or config.distribution == "none":
        return 0
    if config.distribution == "fixed":
        return int(config.fixed_steps)
    if config.distribution == "uniform":
        # numpy's upper endpoint is exclusive; +1 preserves the historical
        # inclusive [min_steps, max_steps] behavior.
        return int(rng.integers(config.min_steps, config.max_steps + 1))
    if config.distribution == "truncated_normal":
        if config.std_steps == 0:
            value = config.mean_steps
        else:
            value = None
            for _ in range(10000):
                candidate = float(rng.normal(config.mean_steps, config.std_steps))
                if config.min_steps <= candidate <= config.max_steps:
                    value = candidate
                    break
            if value is None:
                # This can only occur for a severely inconsistent parameterization.
                value = float(np.clip(config.mean_steps, config.min_steps, config.max_steps))
        value = float(np.clip(value, config.min_steps, config.max_steps))
        return int(np.rint(value))
    raise AssertionError("DelayConfig.validate should have rejected this value")


class ReconfigurableDelaySampler:
    """A sampler whose RNG is reset only when the configured seed changes."""

    def __init__(self) -> None:
        self._seed: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

    def reset(self) -> None:
        """Force the next sample to restart from the configured seed."""
        self._seed = None
        self._rng = None

    def sample(self, config: DelayConfig) -> int:
        if self._rng is None or self._seed != config.seed:
            self._seed = config.seed
            self._rng = np.random.default_rng(config.seed)
        return sample_delay_steps(config, self._rng)
