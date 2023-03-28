#!/usr/bin/env python3
"""Test simple simulators"""
from pyapes.mesh import Mesh
from pyapes.geometry import Cylinder
from pyapes.variables import Field
from pyapes.solver.ops import Solver
from pyapes.solver.fdm import FDM
from pyapes.solver.fdc import FDC, jacobian, hessian
from pymaxed.maxed import Maxed, Vec
from pymytools.constants import PI
import torch
from torch.testing import assert_close

from pyrfp.training_data import get_analytic_bcs, set_bc_rz
from scipy.stats import skewnorm


def test_dsmc_homogeneous() -> None:

    from pyrfp.simulators.dsmc import dsmc_nanbu_homogeneous

    n_particle = 10000

    vel = torch.zeros((n_particle, 3), dtype=torch.float64, device=torch.device("cpu"))

    vel[:, 0] = torch.randn(n_particle)
    vel[:, 1] = torch.randn(n_particle)

    s_norm = skewnorm(1)
    vel[:, 2] = torch.from_numpy(s_norm.rvs(size=n_particle)).to(
        dtype=torch.float64, device=torch.device("cpu")
    )

    # mean correction
    vel -= torch.mean(vel, dim=0)
    # energy correction
    vel /= torch.std(vel, dim=0)

    heat = []

    for i in range(100):

        heat.append(torch.mean(vel[:, 2] ** 3))

        vel = dsmc_nanbu_homogeneous(vel, 0.01)

    from pymyplot import plt

    plt.plot(heat)
    plt.show()


def test_rfp() -> None:

    mesh = Mesh(Cylinder[0:5, -5:5], None, [32, 64])

    mnts_eq = torch.tensor(
        [1, 0, 1, 0, 3, 2, 8, 0, 2], dtype=mesh.dtype.float, device=mesh.device
    )
    vec = Vec(mesh, mnts_eq, 4, [50, 100])
    maxed_eq = Maxed(vec, lr=1.0, max_itr=200, gtol=1e-6, xtol=1e-6, disp=False)
    maxed_eq.solve()

    assert maxed_eq.dist is not None

    density = torch.sum(maxed_eq.dist * 2.0 * PI * mesh.dx[0] * mesh.dx[1] * mesh.R)
    maxed_eq.dist /= density

    energy = torch.sum(
        0.5
        * (mesh.R**2 + mesh.Z**2)
        * maxed_eq.dist
        * 2.0
        * PI
        * mesh.dx[0]
        * mesh.dx[1]
        * mesh.R
    )

    maxed_eq.dist /= torch.sqrt(energy / 1.5)
    energy = torch.sum(
        0.5
        * (mesh.R**2 + mesh.Z**2)
        * maxed_eq.dist
        * 2.0
        * PI
        * mesh.dx[0]
        * mesh.dx[1]
        * mesh.R
    )

    mnts_target = torch.tensor(
        [1, 0, 1, -0.27, 1.7178, 2, 8, 0, 2], dtype=mesh.dtype.float, device=mesh.device
    )
    vec = Vec(mesh, mnts_target, 4, [50, 100])
    maxed = Maxed(vec, lr=1.0, max_itr=200, gtol=1e-6, xtol=1e-6, disp=False)
    maxed.solve()
    assert maxed.dist is not None

    density = torch.sum(maxed.dist * 2.0 * PI * mesh.dx[0] * mesh.dx[1] * mesh.R)
    maxed.dist /= density

    from pymyplot import plt

    pdf = Field("pdf", 1, mesh, None)
    pdf.set_var_tensor(maxed.dist)
    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-5,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdc = FDC({"div": {"limiter": "none", "edge": True}})
    fdm = FDM()

    var = Field("container", 1, mesh, None)

    jacPDF = torch.gradient(pdf[0], spacing=mesh.dx.tolist(), edge_order=2)

    bc_vals = get_analytic_bcs(mesh, pdf[0], "H")
    bc_H = set_bc_rz(bc_vals)
    H_pot = Field("H", 1, mesh, {"domain": bc_H(), "obstacle": None})
    solver.set_eq(fdm.laplacian(H_pot) == -8 * PI * pdf())
    solver.solve()
    jacH = jacobian(var.set_var_tensor(H_pot[0]))

    bc_vals = get_analytic_bcs(mesh, pdf[0], "G")
    bc_G = set_bc_rz(bc_vals)
    G_pot = Field("G", 1, mesh, {"domain": bc_G(), "obstacle": None})
    solver.set_eq(fdm.laplacian(G_pot) == H_pot())
    solver.solve()
    hessG = hessian(var.set_var_tensor(G_pot[0]))

    diffFlux = []
    diffFlux.append(hessG.rr * jacPDF[0] + hessG.rz * jacPDF[1])
    diffFlux.append(hessG.rz * jacPDF[0] + hessG.zz * jacPDF[1])

    friction = (
        torch.gradient(jacH.r * pdf[0], spacing=mesh.dx.tolist(), edge_order=2)[0]
        + torch.nan_to_num(
            jacH.r * pdf[0] / mesh.grid[0], nan=0.0, posinf=0.0, neginf=0.0
        )
        + torch.gradient(jacH.z * pdf[0], spacing=mesh.dx.tolist(), edge_order=2)[1]
    )

    diffusion = (
        torch.nan_to_num(
            torch.gradient(
                diffFlux[0] * mesh.grid[0], spacing=mesh.dx.tolist(), edge_order=2
            )[0]
            / mesh.grid[0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        + torch.gradient(diffFlux[1], spacing=mesh.dx.tolist(), edge_order=2)[1]
    )

    test_direct = friction + diffusion
    test_pyapes = fdc.div(jacH, pdf)[0] + fdc.div(1.0, fdc.diffFlux(hessG, pdf))[0]

    assert_close(test_direct, test_pyapes, atol=1, rtol=1)

    # pdf.set_var_tensor(pdf()[0] + 0.01 * test)
    # _, ax = plt.subplots(1, 3)
    # ax[0].contourf(mesh.R, mesh.Z, friction, cmap="jet")
    # ax[1].contourf(mesh.R, mesh.Z, test_direct, cmap="jet")
    # ax[2].contourf(mesh.R, mesh.Z, test_pyapes, cmap="jet")
    # plt.show()

    from pyrfp.simulators.rfp import RFP_RZ

    rfp_rz = RFP_RZ(mesh, 0.01, 100, maxed.dist)
    pdf_final = rfp_rz.run()

    _, ax = plt.subplots(1, 3)
    ax[0].contourf(mesh.R, mesh.Z, maxed.dist, cmap="jet")
    ax[1].contourf(mesh.R, mesh.Z, pdf_final[0], cmap="jet")
    ax[2].contourf(mesh.R, mesh.Z, maxed_eq.dist, cmap="jet")

    plt.show()
