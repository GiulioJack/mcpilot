# SPDX-License-Identifier: AGPL-3.0-or-later
"""ROS and mock backends for one-shot tossing episodes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


TRAJECTORY_COLUMNS = 10
LANDING_SOURCE = "target_plane_trajectory_endpoint"


@dataclass
class TossResult:
    target: np.ndarray
    command_velocity: float
    landing_position: np.ndarray
    trajectory: np.ndarray
    delay_steps: Optional[int]
    elapsed_seconds: float
    command_anticipation_seconds: Optional[float] = None
    landing_source: str = LANDING_SOURCE


def parse_service_trajectory(flat_trajectory: Sequence[float]) -> np.ndarray:
    flat = np.asarray(flat_trajectory, dtype=np.float64)
    if flat.size == 0:
        raise ValueError("Toss service returned an empty trajectory")
    if flat.size % TRAJECTORY_COLUMNS != 0:
        raise ValueError(
            "Trajectory length {} is not divisible by {}".format(
                flat.size, TRAJECTORY_COLUMNS
            )
        )
    sample_count = flat.size // TRAJECTORY_COLUMNS
    # tossing_experiment_script.py serializes one complete column at a time.
    return flat.reshape(TRAJECTORY_COLUMNS, sample_count).T


def target_plane_endpoint(flat_trajectory: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the parsed path and its final target-plane xyz sample."""
    trajectory = parse_service_trajectory(flat_trajectory)
    return trajectory, trajectory[-1, 1:4].copy()


class RosTossingBackend:
    def __init__(
        self,
        service_name: str,
        bullet_model_name: str,
        orientation: Sequence[float],
        service_timeout: float,
        retries: int,
        delay_config: Dict[str, Any],
    ) -> None:
        import rospy
        from mcpilot.srv import TossExperiment

        self._rospy = rospy
        self.service_name = service_name
        self.bullet_model_name = str(bullet_model_name)
        self.orientation = [float(value) for value in orientation]
        self.service_timeout = float(service_timeout)
        self.retries = int(retries)
        self.delay_config = dict(delay_config)
        self._throw_index = 0
        rospy.wait_for_service(service_name, timeout=self.service_timeout)
        self._service = rospy.ServiceProxy(service_name, TossExperiment, persistent=False)

    def set_delay_config(self, config: Dict[str, Any]) -> None:
        self.delay_config = dict(config)

    def reset_delay_stream(self) -> None:
        self._throw_index = 0
        self.delay_config["reset_id"] = int(self.delay_config.get("reset_id", 0)) + 1

    def _publish_delay_parameters(self, anticipation_seconds: Optional[float] = None) -> None:
        prefix = "/tossing/release_delay"
        values = dict(self.delay_config)
        synthetic_values = values.pop("synthetic_ground_truth", None)
        # Keep one seeded RNG stream across throws, matching the other paper
        # baselines. reset_id restarts that stream for paired evaluations.
        values["seed"] = int(values.get("seed", 0)) % 2147483647
        for key, value in values.items():
            self._rospy.set_param(prefix + "/" + str(key), value)
        if anticipation_seconds is not None:
            self._rospy.set_param(
                prefix + "/anticipation_seconds", float(anticipation_seconds)
            )
        synthetic_prefix = prefix + "/synthetic_ground_truth"
        if synthetic_values is not None:
            for key, value in dict(synthetic_values).items():
                self._rospy.set_param(synthetic_prefix + "/" + str(key), value)
        if self._rospy.get_param(synthetic_prefix + "/enabled", True):
            # In clean synthetic-ground-truth runs the physical sampler is
            # bypassed, so pair SAC through the active synthetic sampler too.
            self._rospy.set_param(synthetic_prefix + "/seed", int(values["seed"]))
            self._rospy.set_param(
                synthetic_prefix + "/reset_id", int(values.get("reset_id", 0))
            )

    def throw(
        self,
        target: Sequence[float],
        velocity: float,
        anticipation_seconds: Optional[float] = None,
    ) -> TossResult:
        import time

        self._publish_delay_parameters(anticipation_seconds)
        target_array = np.asarray(target, dtype=np.float64)
        last_error: Optional[Exception] = None
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                self._rospy.loginfo(
                    "Requesting toss %d (attempt %d/%d): target=%s, velocity=%.6f m/s, anticipation=%s s",
                    self._throw_index,
                    attempt + 1,
                    self.retries + 1,
                    np.array2string(target_array, precision=4),
                    float(velocity),
                    "default" if anticipation_seconds is None else "{:.6f}".format(float(anticipation_seconds)),
                )
                response = self._service(
                    target_array.tolist(),
                    False,
                    float(velocity),
                    self.bullet_model_name,
                    self.orientation,
                )
                trajectory, landing = target_plane_endpoint(response.trajectory)

                target_altitude = float(target_array[2])
                altitude_error = abs(float(landing[2]) - target_altitude)

                if not np.all(np.isfinite(landing)):
                    raise ValueError(
                        "Toss service returned a non-finite landing position: {}".format(
                            landing.tolist()
                        )
                    )

                if altitude_error > 0.05:
                    raise ValueError(
                        "Trajectory ended before a valid target-plane crossing: "
                        "target_z={:.4f}, last_z={:.4f}, velocity={:.4f}".format(
                            target_altitude,
                            float(landing[2]),
                            float(velocity),
                        )
                    )
                delay_steps = self._rospy.get_param(
                    "/tossing/release_delay/last_sample_steps", None
                )
                result = TossResult(
                    target=target_array,
                    command_velocity=float(velocity),
                    landing_position=landing,
                    trajectory=trajectory,
                    delay_steps=None if delay_steps is None else int(delay_steps),
                    elapsed_seconds=time.perf_counter() - started,
                    command_anticipation_seconds=anticipation_seconds,
                )
                self._throw_index += 1
                return result
            except Exception as exc:  # ROS exceptions vary by failure mode.
                last_error = exc
                try:
                    self._service.close()
                except Exception:
                    pass
                if attempt < self.retries:
                    self._rospy.logwarn(
                        "Toss failed (%s); retrying %d/%d",
                        exc,
                        attempt + 1,
                        self.retries,
                    )
                    self._rospy.sleep(1.0)
                    from mcpilot.srv import TossExperiment

                    self._rospy.wait_for_service(
                        self.service_name, timeout=self.service_timeout
                    )
                    self._service = self._rospy.ServiceProxy(
                        self.service_name, TossExperiment, persistent=False
                    )
        raise RuntimeError(
            "Toss service failed after retries for target={} and velocity={:.6f} m/s: {}".format(
                target_array.tolist(), float(velocity), last_error
            )
        )


class MockTossingBackend:
    """Fast deterministic-ish backend for installation tests only.

    It is deliberately not a replacement for Gazebo and must not be used for
    paper results.  Its sole purpose is validating the SAC pipeline before a
    long ROS experiment.
    """

    def __init__(self, seed: int, delay_config: Dict[str, Any], release_z: float = 2.55) -> None:
        from robot_tossing_utils.release_delay import DelayConfig, ReconfigurableDelaySampler

        self.rng = np.random.default_rng(seed)
        self.delay_config = dict(delay_config)
        self._throw_index = 0
        self.release_z = float(release_z)
        self._delay_cls = DelayConfig
        self._sampler = ReconfigurableDelaySampler()

    def set_delay_config(self, config: Dict[str, Any]) -> None:
        self.delay_config = dict(config)

    def reset_delay_stream(self) -> None:
        self._throw_index = 0
        self.delay_config["reset_id"] = int(self.delay_config.get("reset_id", 0)) + 1
        self._sampler.reset()

    def throw(
        self,
        target: Sequence[float],
        velocity: float,
        anticipation_seconds: Optional[float] = None,
    ) -> TossResult:
        import time

        started = time.perf_counter()
        delay_values = dict(self.delay_config)
        delay_values["seed"] = int(delay_values.get("seed", 0)) % 2147483647
        config = self._delay_cls.from_mapping(delay_values)
        delay_steps = self._sampler.sample(config)
        self._throw_index += 1
        target_array = np.asarray(target, dtype=np.float64)
        yaw = float(np.arctan2(target_array[1], target_array[0]))
        flight_time = np.sqrt(max(0.0, 2.0 * (self.release_z - target_array[2]) / 9.81))
        delay_seconds = delay_steps * 0.001
        effective_delay = delay_seconds - float(anticipation_seconds or 0.0)
        effective_velocity = max(0.0, float(velocity) * (1.0 - 0.35 * effective_delay))
        horizontal_distance = effective_velocity * flight_time
        # A small smooth drag term and measurement noise make this useful as a
        # smoke test without pretending to reproduce Gazebo.
        horizontal_distance -= 0.018 * effective_velocity * effective_velocity
        horizontal_distance += self.rng.normal(0.0, 0.01)
        landing = np.asarray(
            [horizontal_distance * np.cos(yaw), horizontal_distance * np.sin(yaw), target_array[2]],
            dtype=np.float64,
        )
        trajectory = np.zeros((2, TRAJECTORY_COLUMNS), dtype=np.float64)
        trajectory[-1, 1:4] = landing
        return TossResult(
            target=target_array,
            command_velocity=float(velocity),
            landing_position=landing,
            trajectory=trajectory,
            delay_steps=delay_steps,
            elapsed_seconds=time.perf_counter() - started,
            command_anticipation_seconds=anticipation_seconds,
        )
