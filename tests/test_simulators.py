#!/usr/bin/env python3
"""Test simple simulators"""
import torch
from pyapes.geometry import Cylinder
from pyapes.mesh import Mesh
from pymyplot import plt
from pymytools.constants import PI
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

    plt.plot(heat)
    plt.show()


def test_rfp() -> None:
    mesh = Mesh(Cylinder[0:5, -5:5], None, [256, 512])

    dist = 1.0 / (2.0 * PI) ** 1.5 * torch.exp(-0.5 * (mesh.R**2 + mesh.Z**2))

    density = torch.sum(dist * 2.0 * PI * mesh.dx[0] * mesh.dx[1] * mesh.R)
    dist /= density

    from pyrfp.simulators.rfp import RFP_RZ

    rfp_rz = RFP_RZ(mesh, 0.0001, 10, dist)
    pdf_final = rfp_rz.run()

    _, ax = plt.subplots(1, 2)
    ax[0].contourf(mesh.R, mesh.Z, pdf_final[0])
    ax[1].contourf(mesh.R, mesh.Z, dist)
    plt.show()
