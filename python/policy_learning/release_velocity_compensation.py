# SPDX-License-Identifier: AGPL-3.0-or-later
import torch


def apply_radial_velocity_compensation(
        cartesian_velocity, speed, yaw, coefficients):
    """Add a polynomial radial correction without assuming column vectors.

    Policy implementations return speed as either ``(N,)`` or ``(N, 1)`` and
    yaw as ``(N,)``.  Flattening only these scalar-per-particle quantities
    keeps the Cartesian velocity shape fixed at ``(N, D)`` and preserves the
    autograd path through speed.
    """
    if cartesian_velocity.ndim != 2 or cartesian_velocity.shape[1] < 2:
        raise ValueError(
            'cartesian_velocity must have shape (N, D) with D >= 2'
        )

    speed_flat = speed.reshape(-1)
    yaw_flat = yaw.reshape(-1)
    batch_size = cartesian_velocity.shape[0]
    if speed_flat.numel() != batch_size or yaw_flat.numel() != batch_size:
        raise ValueError(
            'speed and yaw must contain one value per Cartesian velocity'
        )

    delta_v = torch.zeros_like(speed_flat)
    for power, coefficient in enumerate(coefficients, start=1):
        delta_v = delta_v + float(coefficient) * torch.pow(speed_flat, power)

    radial_xy = torch.stack(
        (delta_v * torch.cos(yaw_flat), delta_v * torch.sin(yaw_flat)),
        dim=1,
    )
    if cartesian_velocity.shape[1] == 2:
        correction = radial_xy
    else:
        correction = torch.cat(
            (radial_xy, torch.zeros_like(cartesian_velocity[:, 2:])),
            dim=1,
        )
    return cartesian_velocity + correction
