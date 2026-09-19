# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic projectile priors used by the tossing GP models."""

import torch


def tossing_delta_theta_dot(X, active_dims_x, active_dims_u, active_dims_x_dot, out_dim, T_sampling):
    """Return the gravity-only prior for one velocity-increment dimension."""
    gravity = torch.as_tensor(9.81, dtype=X.dtype, device=X.device)
    delta_velocity = torch.zeros_like(X[:, active_dims_x_dot])
    delta_velocity[:, 2] = -gravity * T_sampling
    return delta_velocity[:, out_dim].reshape(-1, 1)


def tossing_delta_state(X, active_dims_x, active_dims_pos, active_dims_vel, out_dim, T_sampling):
    """Return the gravity-only prior for one state-increment dimension."""
    gravity = torch.as_tensor(9.81, dtype=X.dtype, device=X.device)
    delta_state = torch.zeros_like(X)
    delta_state[:, active_dims_vel[2]] = -gravity * T_sampling
    delta_state[:, active_dims_pos] = X[:, active_dims_vel] * T_sampling
    return delta_state[:, out_dim].reshape(-1, 1)
