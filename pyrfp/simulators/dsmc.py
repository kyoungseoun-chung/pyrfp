#!/usr/bin/env python3
"""Particle simulator for the homogeneous relaxation.
It includes
    - DSMC
        - Most of implementation can be found in `pystops`. Here, only single species gridless DSMC is implemented to demonstrate our data-driven approach
        - I rewrote here (from `pystops`) since I want this package to be decoupled from `pystops`
    - FP
"""
import torch
from pymytools.constants import PI
from torch import Tensor
from math import ceil, sqrt


def dsmc_nanbu_homogeneous(p_vel: Tensor, dt: float) -> Tensor:
    """Homogeneous relaxation using the Nanbu's DSMC method."""

    n_part = p_vel.shape[0]
    # number of collision pair
    # if number of particles in a cell is odd number,
    # n_pair*2 is larger than number of particles
    n_pair = int(ceil(n_part / 2))

    idx_shuffle = torch.randperm(n_part)
    # pair selection
    pair_a = p_vel[idx_shuffle[:n_pair], :].clone()
    pair_b = p_vel[idx_shuffle[n_pair:], :].clone()

    vel_buffer = torch.zeros_like(p_vel)

    assert (
        pair_a.shape[0] == pair_b.shape[0]
    ), "DSMC: homogeneous case should come with even number of particles."

    # collide particle according to K.Nanbu (1997)
    binary_collision(pair_a, pair_b, dt)

    # update velocity
    vel_buffer[idx_shuffle[:n_pair], :] = pair_a
    vel_buffer[idx_shuffle[n_pair:], :] = pair_b

    return vel_buffer


def binary_collision(vel_a: Tensor, vel_b: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """Collision according to elastic and binary collision."""

    dim = vel_a.size(0)

    # Relative velocity
    g = vel_a - vel_b

    # Magnitude of relative velocity and its perpendicular component
    g_mag = torch.sqrt(torch.sum(g**2, dim=1))
    g_perp = torch.sqrt(g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2])

    s = 8.0 * sqrt(2.0) / (g_mag**3) * dt

    cos_xi = _nanbu_scattering(s)
    sin_xi = torch.sqrt(1 - cos_xi**2)

    h = torch.zeros_like(vel_a)
    rnd = 2 * PI * torch.rand(dim, dtype=vel_a.dtype)

    # Binary collision
    h[:, 0] = g_perp * torch.cos(rnd)
    h[:, 1] = (
        -(g[:, 1] * g[:, 0] * torch.cos(rnd) + g_mag * g[:, 2] * torch.sin(rnd))
        / g_perp
    )
    h[:, 2] = (
        -(g[:, 2] * g[:, 0] * torch.cos(rnd) - g_mag * g[:, 1] * torch.sin(rnd))
        / g_perp
    )

    vel_a -= 0.5 * (g * (1 - cos_xi.view(dim, 1)) + h * sin_xi.view(dim, 1))
    vel_b += 0.5 * (g * (1 - cos_xi.view(dim, 1)) + h * sin_xi.view(dim, 1))

    return (vel_a, vel_b)


def _nanbu_scattering(s_in: Tensor) -> Tensor:
    """Calculate scattering angle based on K.Nanbu's scheme (1997).

    Note:
        - Polynomial approximation is adopted from J. Derouillat et al. (2019)
    """

    cos_xi = torch.zeros_like(s_in)
    rand = torch.rand_like(cos_xi)

    # Make sure cos(xi) > 0
    rand[rand.le(0.001)] = 0.001

    mask_1 = s_in.lt(0.1)
    mask_2 = s_in.ge(0.1) & s_in.lt(3.0)
    mask_3 = s_in.ge(3.0) & s_in.lt(6.0)
    mask_4 = s_in.ge(6.0)

    # s < 0.1
    cos_xi[mask_1] = 1.0 + s_in[mask_1] * torch.log(rand[mask_1])

    # s >= 0.1 and s < 3.0
    inv_a = (
        0.00569578
        + (
            0.95602
            + (
                -0.508139
                + (0.479139 + (-0.12789 + 0.0238957 * s_in[mask_2]) * s_in[mask_2])
                * s_in[mask_2]
            )
            * s_in[mask_2]
        )
        * s_in[mask_2]
    )
    a = 1.0 / inv_a
    cos_xi[mask_2] = inv_a * torch.log(
        torch.exp(-a) + 2.0 * rand[mask_2] * torch.sinh(a)
    )

    # s >= 0.3 and s < 0.6
    a = 3.0 * torch.exp(-s_in[mask_3])
    inv_a = 1.0 / a
    cos_xi[mask_3] = inv_a * torch.log(
        torch.exp(-a) + 2.0 * rand[mask_3] * torch.sinh(a)
    )

    # s >= 0.6
    cos_xi[mask_4] = 2.0 * rand[mask_4] - 1.0

    return cos_xi
