# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gymnasium environment for target-conditioned one-shot throwing."""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sac_tossing.backend import MockTossingBackend, RosTossingBackend, TossResult
from sac_tossing.targets import TargetDomain


class TossingSACEnv(gym.Env):
    """A one-step contextual throwing task.

    Observation: target Cartesian position P.
    Action: normalized release speed, optionally followed by normalized release
    anticipation.
    Episode: exactly one physical/simulated throw.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int,
        service_name: Optional[str] = None,
        backend_name: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.domain = TargetDomain.from_config(config["target_domain"])
        self.action_config = dict(config["action"])
        self.reward_config = dict(config["reward"])
        self.hit_radius = float(config["hit_radius"])
        self.normalize_observation = bool(config.get("normalize_observation", True))
        self.observation_mode = str(config.get("observation_mode", "cartesian"))
        self._target: Optional[np.ndarray] = None
        self._episode_index = 0
        self._log_path = log_path
        self._log_header_written = bool(log_path and os.path.exists(log_path) and os.path.getsize(log_path) > 0)

        self.learn_anticipation = bool(self.action_config.get("learn_anticipation", False))
        fixed_anticipation = self.action_config.get("fixed_anticipation_seconds")
        self.fixed_anticipation_seconds = (
            None if fixed_anticipation is None else float(fixed_anticipation)
        )
        if self.fixed_anticipation_seconds is not None and self.fixed_anticipation_seconds < 0.0:
            raise ValueError("fixed_anticipation_seconds must be >= 0")
        action_dim = 2 if self.learn_anticipation else 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        if self.observation_mode == "radial_distance":
            if self.normalize_observation:
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=np.asarray([self.domain.l_min], dtype=np.float32),
                    high=np.asarray([self.domain.l_max], dtype=np.float32),
                    dtype=np.float32,
                )
        elif self.observation_mode == "cartesian":
            if self.normalize_observation:
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=self.domain.cartesian_low.astype(np.float32),
                    high=self.domain.cartesian_high.astype(np.float32),
                    dtype=np.float32,
                )
        else:
            raise ValueError(
                "Unknown observation_mode: {}".format(self.observation_mode)
            )

        selected_backend = backend_name or config.get("backend", "ros")
        delay_config = dict(config.get("delay", {}))
        # Each environment owns an independent deterministic stream. The backend
        # derives a per-throw seed from this base so alternating training and
        # evaluation calls cannot perturb one another's delay sequence.
        # ROS XML-RPC parameters store signed 32-bit integers. Keep derived
        # seeds deterministic while avoiding overflow for the evaluation stream.
        delay_config["seed"] = int((self.seed_value * 1000003) % 2147483647)
        if selected_backend == "ros":
            selected_service = service_name or config["service_train"]
            self.backend = RosTossingBackend(
                service_name=selected_service,
                bullet_model_name=config.get("bullet_model_name", "red_ball"),
                orientation=config.get("orientation", [0.0, 0.0, 0.0]),
                service_timeout=float(config.get("service_timeout", 600.0)),
                retries=int(config.get("service_retries", 1)),
                delay_config=delay_config,
            )
        elif selected_backend == "mock":
            self.backend = MockTossingBackend(
                seed=self.seed_value,
                delay_config=delay_config,
                release_z=float(config.get("release_z_world", 2.55)),
            )
        else:
            raise ValueError("Unknown backend: {}".format(selected_backend))

    def _observation(self, target: Sequence[float]) -> np.ndarray:
        if self.observation_mode == "radial_distance":
            if self.normalize_observation:
                return self.domain.normalize_radius(target)
            return np.asarray(
                [np.linalg.norm(np.asarray(target, dtype=np.float64)[:2])],
                dtype=np.float32,
            )
        if self.normalize_observation:
            return self.domain.normalize_cartesian(target)
        return np.asarray(target, dtype=np.float32)

    def _ballistic_velocity(self, target: Sequence[float]) -> float:
        target_arr = np.asarray(target, dtype=np.float64)
        release_radius = float(self.config.get("release_radius", 0.07))
        release_z = float(self.config.get("release_z_world", 2.55))
        target_radius = float(np.linalg.norm(target_arr[:2]))
        horizontal_distance = max(target_radius - release_radius, 1e-6)
        vertical_drop = max(release_z - target_arr[2], 1e-6)
        flight_time = np.sqrt(2.0 * vertical_drop / 9.81)
        return horizontal_distance / flight_time

    def action_to_velocity(self, action: Sequence[float]) -> float:
        action_value = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        v_min = float(self.action_config["v_min"])
        v_max = float(self.action_config["v_max"])
        mode = str(self.action_config.get("mode", "absolute"))
        if mode == "absolute":
            velocity = v_min + 0.5 * (action_value + 1.0) * (v_max - v_min)
        elif mode == "residual_ballistic":
            if self._target is None:
                raise RuntimeError("reset() must be called before action_to_velocity()")
            residual_scale = float(self.action_config.get("residual_scale", 0.5))
            velocity = self._ballistic_velocity(self._target) + residual_scale * action_value
            velocity = float(np.clip(velocity, v_min, v_max))
        else:
            raise ValueError("Unknown action mode: {}".format(mode))
        return float(velocity)

    def action_to_anticipation(self, action: Sequence[float]) -> Optional[float]:
        if not self.learn_anticipation:
            return self.fixed_anticipation_seconds
        values = np.asarray(action).reshape(-1)
        if values.size < 2:
            raise ValueError("Delay-compensation action requires two action values")
        normalized = float(np.clip(values[1], -1.0, 1.0))
        minimum = float(self.action_config.get("anticipation_min_seconds", 0.100))
        maximum = float(self.action_config.get("anticipation_max_seconds", 0.150))
        if maximum < minimum:
            raise ValueError("anticipation_max_seconds must be >= anticipation_min_seconds")
        return minimum + 0.5 * (normalized + 1.0) * (maximum - minimum)

    def _reward(self, planar_error: float) -> float:
        reward_type = str(self.reward_config.get("type", "saturated"))
        if reward_type == "saturated":
            length_scale = float(self.reward_config.get("length_scale", self.hit_radius))
            cost = 1.0 - np.exp(-((planar_error / length_scale) ** 2))
            reward = -float(cost)
        elif reward_type == "distance":
            reward = -float(planar_error)
        else:
            raise ValueError("Unknown reward type: {}".format(reward_type))
        if planar_error <= self.hit_radius:
            reward += float(self.reward_config.get("success_bonus", 0.0))
        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        requested_target = None if options is None else options.get("target")
        if requested_target is None:
            self._target = self.domain.sample(self.rng)
        else:
            target = np.asarray(requested_target, dtype=np.float64)
            if not self.domain.contains(target, atol=1e-6):
                raise ValueError("Requested target is outside the configured domain: {}".format(target))
            self._target = target
        return self._observation(self._target), {"target": self._target.copy()}

    def step(self, action: Sequence[float]):
        if self._target is None:
            raise RuntimeError("reset() must be called before step()")
        velocity = self.action_to_velocity(action)
        anticipation_seconds = self.action_to_anticipation(action)
        result = self.backend.throw(
            self._target, velocity, anticipation_seconds=anticipation_seconds
        )
        planar_error = float(np.linalg.norm(result.landing_position[:2] - self._target[:2]))
        reward = self._reward(planar_error)
        success = planar_error <= self.hit_radius
        info = {
            "target": self._target.copy(),
            "landing_position": result.landing_position.copy(),
            "landing_source": result.landing_source,
            "command_velocity": velocity,
            "command_anticipation_seconds": anticipation_seconds,
            "planar_error": planar_error,
            "is_success": bool(success),
            "delay_steps": result.delay_steps,
            "throw_elapsed_seconds": result.elapsed_seconds,
            "episode_index": self._episode_index,
        }
        self._write_log(info, reward)
        self._episode_index += 1
        observation = self._observation(self._target)
        return observation, reward, True, False, info

    def _write_log(self, info: Dict[str, Any], reward: float) -> None:
        if not self._log_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
        target = np.asarray(info["target"])
        landing = np.asarray(info["landing_position"])
        row = {
            "episode": info["episode_index"],
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "landing_x": landing[0],
            "landing_y": landing[1],
            "landing_z": landing[2],
            "landing_source": info["landing_source"],
            "command_velocity": info["command_velocity"],
            "command_anticipation_seconds": (
                "" if info["command_anticipation_seconds"] is None
                else info["command_anticipation_seconds"]
            ),
            "planar_error": info["planar_error"],
            "success": int(info["is_success"]),
            "reward": reward,
            "delay_steps": "" if info["delay_steps"] is None else info["delay_steps"],
            "throw_elapsed_seconds": info["throw_elapsed_seconds"],
        }
        with open(self._log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not self._log_header_written:
                writer.writeheader()
                self._log_header_written = True
            writer.writerow(row)

    def set_delay_config(self, delay_config: Dict[str, Any]) -> None:
        merged = dict(self.config.get("delay", {}))
        merged.update(delay_config)
        merged.setdefault("seed", int((self.seed_value * 1000003) % 2147483647))
        self.backend.set_delay_config(merged)

    def reset_delay_stream(self) -> None:
        self.backend.reset_delay_stream()
