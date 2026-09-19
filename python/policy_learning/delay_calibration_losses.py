# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loss functions used by release-delay calibration.

This module deliberately has no ROS or MC-PILCO dependencies so that the
scoring rules can be tested independently from the simulator.
"""

import numpy as np


DELAY_LOSSES = ('mse', 'energy_score', 'trajectory_energy_score')


def interpolate_descending_plane_crossing(positions, plane_altitude):
    """Linearly interpolate the first descending crossing of a horizontal plane.

    Returns ``None`` when the sampled trajectory never brackets the plane.
    The returned altitude is set exactly to ``plane_altitude`` so that landing
    scores are independent of the rollout sampling phase.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError('positions must be [time, dimension] with at least xyz')

    for k in range(1, positions.shape[0]):
        previous = positions[k - 1]
        current = positions[k]
        previous_z = previous[2]
        current_z = current[2]
        if not np.isfinite(previous_z) or not np.isfinite(current_z):
            continue
        if previous_z >= plane_altitude and current_z <= plane_altitude:
            delta_z = current_z - previous_z
            if delta_z == 0.0:
                crossing = current.copy()
            else:
                fraction = (plane_altitude - previous_z) / delta_z
                crossing = previous + fraction * (current - previous)
            crossing[2] = plane_altitude
            return crossing
    return None


def energy_score(particles, observation):
    """Return the multivariate energy score for one observation."""
    particles = np.asarray(particles, dtype=float)
    observation = np.asarray(observation, dtype=float)
    if particles.ndim != 2 or particles.shape[0] == 0:
        raise ValueError('particles must be a non-empty [particle, dimension] array')
    if observation.shape != particles.shape[1:]:
        raise ValueError('observation dimension does not match particles')
    first_term = np.linalg.norm(particles - observation[None, :], axis=1).mean()
    pairwise = particles[:, None, :] - particles[None, :, :]
    second_term = 0.5 * np.linalg.norm(pairwise, axis=2).mean()
    return float(first_term - second_term)


def align_position_trajectories(simulated_positions, sampling_period,
                                measured_times, measured_positions):
    """Align simulated particles and measured positions on the simulation grid.

    Only finite samples in the common valid time interval are returned.  The
    simulated array is expected to have shape [time, particle, xyz].
    """
    simulated_positions = np.asarray(simulated_positions, dtype=float)
    measured_times = np.asarray(measured_times, dtype=float).reshape(-1)
    measured_positions = np.asarray(measured_positions, dtype=float)
    if simulated_positions.ndim != 3:
        raise ValueError('simulated_positions must be [time, particle, dimension]')
    if measured_positions.ndim != 2 or measured_positions.shape[0] != measured_times.size:
        raise ValueError('measured trajectory shapes are inconsistent')
    if measured_positions.shape[1] != simulated_positions.shape[2]:
        raise ValueError('measured and simulated position dimensions differ')
    if sampling_period <= 0.0:
        raise ValueError('sampling_period must be positive')

    finite = np.isfinite(measured_times) & np.all(np.isfinite(measured_positions), axis=1)
    measured_times = measured_times[finite]
    measured_positions = measured_positions[finite]
    if measured_times.size == 0:
        raise ValueError('measured trajectory has no finite samples')
    order = np.argsort(measured_times)
    measured_times = measured_times[order]
    measured_positions = measured_positions[order]
    measured_times, unique_indices = np.unique(measured_times, return_index=True)
    measured_positions = measured_positions[unique_indices]

    simulated_times = np.arange(simulated_positions.shape[0], dtype=float) * sampling_period
    common_start = max(float(simulated_times[0]), float(measured_times[0]))
    common_end = min(float(simulated_times[-1]), float(measured_times[-1]))
    valid_sim = (simulated_times >= common_start - 1e-12) & (
        simulated_times <= common_end + 1e-12)
    aligned_times = simulated_times[valid_sim]
    if aligned_times.size == 0:
        raise ValueError('measured and simulated trajectories do not overlap')
    measured_aligned = np.column_stack([
        np.interp(aligned_times, measured_times, measured_positions[:, axis])
        for axis in range(measured_positions.shape[1])
    ])
    return simulated_positions[valid_sim], measured_aligned, aligned_times


def trajectory_energy_score(simulated_positions, sampling_period,
                            measured_times, measured_positions):
    """Energy score after flattening each full aligned position trajectory."""
    simulated_aligned, measured_aligned, _ = align_position_trajectories(
        simulated_positions, sampling_period, measured_times, measured_positions
    )
    # [time, particle, xyz] -> [particle, time * xyz]
    particle_vectors = np.transpose(simulated_aligned, (1, 0, 2)).reshape(
        simulated_aligned.shape[1], -1
    )
    return energy_score(particle_vectors, measured_aligned.reshape(-1))
