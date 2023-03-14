#!/usr/bin/env python3
"""Particle simulator for the homogeneous relaxation.
It includes
    - DSMC
        - Most of implementation can be found in `pystops`. Here, only single species gridless DSMC is implemented to demonstrate our data-driven approach
        - I rewrote here (from `pystops`) since I want this package to be decoupled from `pystops`
    - FP
"""
import torch
from pymytools.constants import EPS0
from pymytools.constants import PI
from torch import Tensor


def binary_nanbu(
    vel_a: torch.Tensor,
    vel_b: torch.Tensor,
    m_a: float,
    m_b: float,
    q_a: float,
    q_b: float,
    n_b: float,
    ln_lambda: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collision according to elastic and binary collision."""

    dim = vel_a.size(0)

    # Relative velocity
    g = vel_a - vel_b

    # Magnitude of relative velocity and its perpendicular component
    g_mag = torch.sqrt(torch.sum(g**2, dim=1))
    g_perp = torch.sqrt(g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2])

    mu = m_a * m_b / (m_a + m_b)

    s = (
        ln_lambda
        / (4 * PI)
        * ((q_a * q_b / (EPS0 * mu)) ** 2)
        * n_b
        / (g_mag**3)
        * dt
    )

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

    vel_a -= mu / m_a * (g * (1 - cos_xi.view(dim, 1)) + h * sin_xi.view(dim, 1))
    vel_b += mu / m_a * (g * (1 - cos_xi.view(dim, 1)) + h * sin_xi.view(dim, 1))

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
