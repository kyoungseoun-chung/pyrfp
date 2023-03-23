#!/usr/bin/env python3
"""Test R-potential related functions."""
import time
from math import sqrt

import numpy as np
import torch
from pyapes.core.geometry import Cylinder
from pyapes.core.mesh import Mesh
from pymytools.logger import Report
from scipy.special import ellipe as s_ellipe
from scipy.special import ellipk as s_ellipk
from torch import Tensor
from torch import vmap
from torch.testing import assert_close


def test_naive_loops() -> None:
    nx = 8
    ny = 16

    x = torch.linspace(0, 2, nx, dtype=torch.float64)
    y = torch.linspace(-2, 1, ny, dtype=torch.float64)

    grid = torch.meshgrid(x, y, indexing="ij")

    tic = time.perf_counter()
    res_naive = _naive_for_loop(grid)
    t_naive = time.perf_counter() - tic

    tic = time.perf_counter()
    res_vector_1 = _vectorized_for_loop_1(grid)
    t_vector_1 = time.perf_counter() - tic

    tic = time.perf_counter()
    res_vector_2 = _vectorized_for_loop_2(grid)
    t_vector_2 = time.perf_counter() - tic

    tic = time.perf_counter()
    res_vector_3 = _vectorized_for_loop_3(grid)
    t_vector_3 = time.perf_counter() - tic

    assert_close(res_naive, res_vector_1)
    assert_close(res_naive, res_vector_2)
    assert_close(res_naive, res_vector_3)

    data = {
        "Schemes": ["Naive sum", "Loop sum", "Copied sum", "Vmap sum"],
        "Elapsed time (s)": [t_naive, t_vector_1, t_vector_2, t_vector_3],
    }

    report = Report("Time comparison", data, ["center", "left"], ["blue", "red"])
    report.display()


def _vectorized_for_loop_3(grid: tuple[Tensor, Tensor]) -> Tensor:
    x = grid[0]
    y = grid[1]
    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    v_func_x = vmap(lambda t: torch.sum((t + x) ** 2))
    v_func_y = vmap(lambda t: torch.sum((t - y) ** 2))

    return (
        (v_func_x(x.flatten()).view(*x.shape) + v_func_y(y.flatten()).view(*y.shape))
        * dx
        * dy
    )


def _vectorized_for_loop_2(grid: tuple[Tensor, Tensor]) -> Tensor:
    x = grid[0]
    y = grid[1]
    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    nx = grid[0].shape[0]
    ny = grid[0].shape[1]

    x_field = x.flatten().repeat(nx, ny, 1)
    y_field = y.flatten().repeat(nx, ny, 1)

    dim = 2
    re_dim = [1 for _ in range(dim)]
    # Add one dimension to the last and repeat it
    x = x.unsqueeze(dim).repeat(*re_dim, nx * ny)
    y = y.unsqueeze(dim).repeat(*re_dim, nx * ny)

    return torch.sum(((x_field + x) ** 2 + (y_field - y) ** 2) * dx * dy, dim=dim)


def _vectorized_for_loop_1(grid: tuple[Tensor, Tensor]) -> Tensor:
    x = grid[0]
    y = grid[1]
    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    nx = grid[0].shape[0]
    ny = grid[0].shape[1]

    res = torch.zeros_like(x)

    for i in range(nx):
        for j in range(ny):
            res[i, j] = torch.sum((x[i, j] + x) ** 2 + (y[i, j] - y) ** 2) * dx * dy

    return res


def _naive_for_loop(grid: tuple[Tensor, Tensor]) -> Tensor:
    x = grid[0]
    y = grid[1]
    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    nx = grid[0].shape[0]
    ny = grid[0].shape[1]

    res = torch.zeros_like(x)

    for i in range(nx):
        for j in range(ny):
            for n in range(nx):
                for m in range(ny):
                    res[i, j] += (
                        ((x[i, j] + x[n, m]) ** 2 + (y[i, j] - y[n, m]) ** 2) * dx * dy
                    )

    return res


def naive_analytic_potential_numpy(
    grid: tuple[Tensor, ...], pdf: Tensor
) -> tuple[np.ndarray, np.ndarray]:
    hr = grid[0][1, 0] - grid[0][0, 0]
    hz = grid[1][0, 1] - grid[1][0, 0]

    # r and z direction has same number of grid points
    mr = grid[0].shape[0]
    mz = grid[0].shape[1]

    H = np.zeros((mr, mz))
    G = np.zeros((mr, mz))

    for n in range(mr):
        for m in range(mz):
            u_r = grid[0][n, m]
            u_z = grid[1][n, m]

            h_pot = 0.0
            g_pot = 0.0
            for i in range(mr):
                for j in range(mz):
                    u_r_p = grid[0][i, j]
                    u_z_p = grid[1][i, j]

                    # |v - v'| in cylindrical coordinate
                    # => sqrt( v_r^2 + v_r'^2
                    #         + (v_z - v_z')^2 - 2v_r*v_z*cos(theta - theta') )
                    # and since radial symmetry,
                    # => sqrt ( (v_r + v_r')^2 + (v_z - v_z')^2 )
                    inner = np.power(u_r + u_r_p, 2) + np.power(u_z - u_z_p, 2)

                    if inner == 0:
                        # singular point. Ignore
                        pass
                    else:
                        # Chacon's expression is wrong (see below)
                        # -> k = 4*u_r*u_r_p/np.sqrt(inner)
                        # use corrected one
                        # G.P.Lennon et al., (1979)
                        k = 4 * u_r * u_r_p / inner
                        if k == 1.0:
                            # singular point. Ignore
                            pass
                        else:
                            # complete elliptic integrals of the first kind
                            ek = s_ellipk(k)
                            # second kind
                            ee = s_ellipe(k)

                            # integration according to L. Chacon et al., (2000)
                            h_pot += (
                                8 * u_r_p * pdf[i, j] * ek / np.sqrt(inner) * hr * hz
                            )
                            g_pot += (
                                4 * u_r_p * pdf[i, j] * ee * np.sqrt(inner) * hr * hz
                            )

            H[n, m] = h_pot
            G[n, m] = g_pot

    return H, G


def rayleigh_pdf(grid: tuple[Tensor, ...]) -> Tensor:
    pdf = (
        grid[0]
        * torch.exp(-grid[0] ** 2 / 2)
        * torch.exp(-grid[1] ** 2 / 2)
        / (sqrt(torch.pi))
    )

    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    density = torch.sum(pdf * dx * dy)

    # Make sure density is 1
    pdf /= density
    return pdf


def test_potentials() -> None:
    from pyrfp.training_data import analytic_potentials_rz

    nx = 5
    ny = 10

    x = torch.linspace(0, 5, nx, dtype=torch.float64)
    y = torch.linspace(-5, 5, ny, dtype=torch.float64)
    grid = torch.meshgrid(x, y, indexing="ij")

    pdf = rayleigh_pdf(grid)

    tic = time.perf_counter()
    np_H, np_G = naive_analytic_potential_numpy(grid, pdf)
    t_naive = time.perf_counter() - tic

    tic = time.perf_counter()
    t_H = analytic_potentials_rz(grid, grid, pdf, "H")
    t_G = analytic_potentials_rz(grid, grid, pdf, "G")
    t_torch = time.perf_counter() - tic

    assert_close(torch.from_numpy(np_H).to(dtype=t_H.dtype, device=t_H.device), t_H)
    assert_close(torch.from_numpy(np_G).to(dtype=t_G.dtype, device=t_G.device), t_G)

    data = {"Scheme": ["Naive", "Vectorized"], "Elapsed Time (s)": [t_naive, t_torch]}

    table = Report("Computational Costs", data, style=["green", "red"])
    table.display()


def test_potential_boundary() -> None:
    from pyrfp.training_data import get_analytic_bcs
    from pymytools.logger import timer

    n_grid = [16, 32]

    if torch.backends.mps.is_available():  # type: ignore
        device = "mps"
        dtype = "single"

        mesh = Mesh(Cylinder[0:5, -5:5], None, n_grid, device=device, dtype=dtype)
        pdf = rayleigh_pdf(mesh.grid)

        timer.start("mps")
        get_analytic_bcs(mesh, pdf, "H")
        get_analytic_bcs(mesh, pdf, "G")
        timer.end("mps")

    device = "cpu"
    dtype = "double"

    mesh = Mesh(Cylinder[0:5, -5:5], None, n_grid, device=device, dtype=dtype)
    pdf = rayleigh_pdf(mesh.grid)

    timer.start("cpu")
    get_analytic_bcs(mesh, pdf, "H")
    get_analytic_bcs(mesh, pdf, "G")
    timer.end("cpu")


def test_potential_field_solver() -> None:
    from pyrfp.training_data import RosenbluthPotentials_RZ, analytic_potentials_rz

    mesh = Mesh(Cylinder[0:5, -5:5], None, [32, 64])
    pdf = rayleigh_pdf(mesh.grid)

    RP_rz = RosenbluthPotentials_RZ(
        mesh,
        solver_config={
            "method": "bicgstab",
            "tol": 1e-6,
            "max_it": 1000,
            "report": False,
        },
    )
    res = RP_rz.from_pdf(pdf)
    t_H = analytic_potentials_rz(mesh.grid, mesh.grid, pdf, "H")
    t_G = analytic_potentials_rz(mesh.grid, mesh.grid, pdf, "G")

    assert res["pots"] is not None

    assert_close(res["pots"]["H"], t_H, atol=1e-1, rtol=1e-1)
    assert_close(res["pots"]["G"], t_G, atol=1e-1, rtol=1e-1)

    # 32 x 64 is too coarse for the gradient
    # jac_tH = torch.gradient(t_H, spacing=mesh.dx.tolist(), edge_order=2)
    # assert_close(res["pots"]["jacH"].r, jac_tH[0], atol=1e-1, rtol=1e-1)
