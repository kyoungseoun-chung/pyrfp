#!/usr/bin/env python3
"""Test R-potential related functions."""
import time
from math import sqrt

import numpy as np
import torch
from pyapes.geometry import Cylinder
from pyapes.mesh import Mesh
from pymytools.diagnostics import DataLoader
from pymytools.diagnostics import DataSaver
from pymytools.logger import Report
from scipy.special import ellipe as s_ellipe
from scipy.special import ellipk as s_ellipk
from torch import Tensor
from torch.testing import assert_close


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
    from pyrfp.training_data import analytic_potentials_rz, analytic_potentials_rz_cpu

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

    tic = time.perf_counter()
    t_H_cpu = analytic_potentials_rz_cpu(grid, grid, pdf, "H")
    t_G_cpu = analytic_potentials_rz_cpu(grid, grid, pdf, "G")
    t_torch_cpu = time.perf_counter() - tic

    assert_close(torch.from_numpy(np_H).to(dtype=t_H.dtype, device=t_H.device), t_H)
    assert_close(torch.from_numpy(np_G).to(dtype=t_G.dtype, device=t_G.device), t_G)

    assert_close(
        torch.from_numpy(np_H).to(dtype=t_H_cpu.dtype, device=t_H_cpu.device), t_H_cpu
    )
    assert_close(
        torch.from_numpy(np_G).to(dtype=t_G_cpu.dtype, device=t_G_cpu.device), t_G_cpu
    )

    data = {
        "Scheme": ["Naive", "Vectorized", "CPU only"],
        "Elapsed Time (s)": [t_naive, t_torch, t_torch_cpu],
    }

    table = Report("Computational Costs", data, style=["green", "red"])
    table.display()


def test_potential_boundary() -> None:
    from pyrfp.training_data import get_analytic_bcs
    from pymytools.logger import timer

    n_grid = [32, 64]

    if torch.backends.mps.is_available():  # type: ignore
        device = "mps"
        dtype = "single"

        mesh = Mesh(Cylinder[0:5, -5:5], None, n_grid, device=device, dtype=dtype)
        pdf = rayleigh_pdf(mesh.grid)

        timer.start("mps")
        get_analytic_bcs(mesh, pdf, "H", cpu=False)
        get_analytic_bcs(mesh, pdf, "G", cpu=False)
        timer.end("mps")

        timer.start("mps_vec")
        get_analytic_bcs(mesh, pdf, "H")
        get_analytic_bcs(mesh, pdf, "G")
        timer.end("mps_vec")

    device = "cpu"
    dtype = "double"

    mesh = Mesh(Cylinder[0:5, -5:5], None, n_grid, device=device, dtype=dtype)
    pdf = rayleigh_pdf(mesh.grid)

    timer.start("cpu_vec")
    get_analytic_bcs(mesh, pdf, "H", cpu=False)
    get_analytic_bcs(mesh, pdf, "G", cpu=False)
    timer.end("cpu_vec")

    timer.start("cpu")
    get_analytic_bcs(mesh, pdf, "H")
    get_analytic_bcs(mesh, pdf, "G")
    timer.end("cpu")

    dl = DataLoader()
    target = dl.read_hdf5(
        "./tests/test_data/pot_bcs.h5",
        ["h_top", "h_bottom", "h_right", "g_top", "g_bottom", "g_right", "pdf"],
    )
    h_bcs = get_analytic_bcs(mesh, target["pdf"], "H")
    g_bcs = get_analytic_bcs(mesh, target["pdf"], "G")

    assert_close(h_bcs["zu"], target["h_top"])
    assert_close(h_bcs["zl"], target["h_bottom"])
    assert_close(h_bcs["ru"][1:-1], target["h_right"])

    assert_close(g_bcs["zu"], target["g_top"])
    assert_close(g_bcs["zl"], target["g_bottom"])
    assert_close(g_bcs["ru"][1:-1], target["g_right"])


def test_mnts_to_potential() -> None:
    from pyrfp.training_data import RosenbluthPotentials_RZ, analytic_potentials_rz_cpu

    dl = DataLoader()

    mesh = Mesh(Cylinder[0:5, -5:5], None, [32, 64])

    mnts_eq = torch.tensor(
        [1, 0, 1, 0, 3, 2, 8, 0, 2], dtype=mesh.dtype.float, device=mesh.device
    )

    RP_rz = RosenbluthPotentials_RZ(
        mesh,
        optimizer_config={
            "lr": 1.0,
            "max_iter": 200,
            "gtol": 1e-6,
            "xtol": 1e-6,
        },
        solver_config={
            "method": "bicgstab",
            "tol": 1e-8,
            "max_it": 1000,
            "report": False,
        },
    )

    pots_mnts = RP_rz.from_moments(mnts_eq)["pots"]

    assert pots_mnts is not None

    target = dl.read_hdf5(
        "./tests/test_data/mnts2pot.h5",
        ["mnts", "coeffs", "H", "jacH", "G", "jacG", "hessG", "pdf"],
    )
    H_exact = analytic_potentials_rz_cpu(mesh.grid, mesh.grid, target["pdf"], "H")
    G_exact = analytic_potentials_rz_cpu(mesh.grid, mesh.grid, target["pdf"], "G")

    assert_close(pots_mnts["H"], H_exact, atol=1e-1, rtol=1e-1)
    assert_close(pots_mnts["G"], G_exact, atol=1e-1, rtol=1e-1)

    assert_close(pots_mnts["H"], target["H"], atol=1e-1, rtol=1e-1)
    assert_close(pots_mnts["G"], target["G"], atol=1e-1, rtol=1e-1)

    ds = DataSaver("./tests/test_data/")
    ds.save_hdf5({"H": pots_mnts["H"], "G": pots_mnts["G"]}, "pots.h5")

    torch_jacG = torch.gradient(pots_mnts["G"], spacing=mesh.dx.tolist(), edge_order=2)[
        0
    ]

    assert_close(pots_mnts["jacH"].r, target["jacH"][0], atol=1e-1, rtol=1e-1)
    assert_close(pots_mnts["jacG"].r, target["jacG"][0], atol=1e-1, rtol=1e-1)

    assert_close(pots_mnts["hessG"].rr, target["hessG"][0], atol=1e-1, rtol=1e-1)

    from pymyplot import myplt as plt

    _, ax = plt.subplots(1, 3)

    ax[0].plot(mesh.grid[0][:, 16], target["G"][:, 16], "ro")
    ax[0].plot(mesh.grid[0][:, 16], pots_mnts["G"][:, 16], "b-")

    ax[1].plot(mesh.grid[0][:, 16], target["jacG"][0][:, 16], "ro")
    ax[1].plot(mesh.grid[0][:, 16], pots_mnts["jacG"].r[:, 16], "b-")
    ax[1].plot(mesh.grid[0][:, 16], torch_jacG[:, 16], "g:")

    ax[2].plot(mesh.grid[0][:, 16], target["hessG"][0][:, 16], "ro")
    ax[2].plot(mesh.grid[0][:, 16], pots_mnts["hessG"].rr[:, 16], "b-")

    plt.show()


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
