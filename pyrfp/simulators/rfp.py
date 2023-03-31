#!/usr/bin/env python3
"""Homogeneous relaxation of the RFP equation.
Note:
    - Gave up
"""
from dataclasses import dataclass

import torch
from pyapes.mesh import Mesh
from pyapes.solver.fdc import FDC
from pyapes.solver.fdc import hessian
from pyapes.solver.fdc import jacobian
from pyapes.solver.fdm import FDM
from pyapes.solver.ops import Solver
from pyapes.solver.rfp import RFP
from pyapes.variables import Field
from pyapes.variables.container import Hess
from pyapes.variables.container import Jac
from pymaxed.maxed import Maxed
from pymaxed.maxed import Vec
from pymytools.constants import PI
from torch import Tensor

from pyrfp.training_data import get_analytic_bcs
from pyrfp.training_data import set_bc_rz


@dataclass
class RFP_RZ:
    mesh: Mesh
    """Mesh object."""
    dt: float
    """Time step."""
    n_itr: int
    """Number of iterations."""
    init_val: Tensor | None = None
    """Initial PDF."""

    def set_initial_pdf(self, mnts: list[float] | Tensor) -> None:
        vec = Vec(self.mesh, mnts, 4, [50, 100])
        maxed = Maxed(vec, lr=1.0, max_itr=200, gtol=1e-6, xtol=1e-6, disp=False)
        maxed.solve()

        if maxed.success:
            assert maxed.dist is not None
            self.init_val = maxed.dist
        else:
            raise RuntimeError(
                f"RFP_RZ: Failed to solve for initial PDF from the given set of moments ({mnts=})."
            )

    def run(self) -> Field:
        assert self.init_val is not None, "RFP_RZ: Initial PDF is not set."

        pdf = Field("pdf", 1, self.mesh, None)
        pdf.set_var_tensor(self.init_val)
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
        rfp = RFP()

        for itr in range(self.n_itr):
            bc_vals = get_analytic_bcs(self.mesh, pdf[0], "H")
            bc_H = set_bc_rz(bc_vals)
            H_pot = Field("H", 1, self.mesh, {"domain": bc_H(), "obstacle": None})
            solver.set_eq(fdm.laplacian(H_pot) == -8 * PI * pdf())
            solver.solve()
            jacH = jacobian(H_pot)

            bc_vals = get_analytic_bcs(self.mesh, pdf[0], "G")
            bc_G = set_bc_rz(bc_vals)
            G_pot = Field("G", 1, self.mesh, {"domain": bc_G(), "obstacle": None})
            solver.set_eq(fdm.laplacian(G_pot) == H_pot())
            solver.solve()
            hessG = hessian(G_pot)

            # pdf[0] = pdf[0] + self.dt * (
            #     -rfp.friction(jacH, pdf) + 0.5 * rfp.diffusion(hessG, pdf)
            # )
            pdf[0] = pdf[0] + self.dt * (
                -fdc.div(jacH, pdf)[0] + fdc.div(1.0, fdc.diffFlux(hessG, pdf))[0]
            )
            # friction = rfp.friction(jacH, pdf)
            # diffusion = rfp.diffusion(hessG, pdf)

            # f_fdc = fdc.div(jacH, pdf)[0]
            # d_dfc = fdc.div(1.0, fdc.diffFlux(hessG, pdf))[0]

            # from pymyplot import plt
            # from pymyplot.colors import TOLCmap

            # _, ax = plt.subplots(2, 3)
            # ax[0][0].contourf(pdf.mesh.R, pdf.mesh.Z, H_pot[0], cmap=TOLCmap.sunset())
            # ax[0][1].contourf(pdf.mesh.R, pdf.mesh.Z, G_pot[0], cmap=TOLCmap.sunset())
            # ax[0][2].contourf(pdf.mesh.R, pdf.mesh.Z, pdf[0], cmap=TOLCmap.sunset())
            # ax[1][0].contourf(pdf.mesh.R, pdf.mesh.Z, friction, cmap=TOLCmap.sunset())
            # ax[1][1].contourf(pdf.mesh.R, pdf.mesh.Z, jacH.r, cmap=TOLCmap.sunset())
            # ax[1][2].contourf(pdf.mesh.R, pdf.mesh.Z, jacH.z, cmap=TOLCmap.sunset())

            # _, ax = plt.subplots(2, 4, subplot_kw={"projection": "3d"})
            # ax[0][0].contourf(pdf.mesh.R, pdf.mesh.Z, -friction, cmap=TOLCmap.sunset())
            # ax[0][1].contourf(pdf.mesh.R, pdf.mesh.Z, diffusion, cmap=TOLCmap.sunset())
            # ax[0][2].contourf(
            #     pdf.mesh.R, pdf.mesh.Z, -friction + diffusion, cmap=TOLCmap.sunset()
            # )
            # ax[0][3].contourf(
            #     pdf.mesh.R,
            #     pdf.mesh.Z,
            #     pdf[0] + self.dt * (-friction + diffusion),
            #     cmap=TOLCmap.sunset(),
            # )
            # ax[1][0].contourf(pdf.mesh.R, pdf.mesh.Z, -f_fdc, cmap=TOLCmap.sunset())
            # ax[1][1].contourf(
            #     pdf.mesh.R, pdf.mesh.Z, 0.5 * d_dfc, cmap=TOLCmap.sunset()
            # )
            # ax[1][2].contourf(
            #     pdf.mesh.R, pdf.mesh.Z, -f_fdc + 0.5 * d_dfc, cmap=TOLCmap.sunset()
            # )
            # ax[1][3].contourf(
            #     pdf.mesh.R,
            #     pdf.mesh.Z,
            #     pdf[0] + self.dt * (-f_fdc + 0.5 * d_dfc),
            #     cmap=TOLCmap.sunset(),
            # )
            # pass

            pdf[0][pdf[0].le(1e-10)] = 0.0
            # den = pdf.volume_integral()
            # pdf.set_var_tensor(pdf[0] / den)

        return pdf


def get_friction(jacH: Jac, pdf: Field) -> Tensor:
    friction = torch.zeros_like(pdf[0])

    dx = pdf.mesh.dx
    r = pdf.mesh.grid[0]

    r_rm = (r[:-2, 1:-1] + r[1:-1, 1:-1]) / 2.0
    r_rp = (r[2:, 1:-1] + r[1:-1, 1:-1]) / 2.0

    pdf_rm = (pdf[0][:-2, 1:-1] + pdf[0][1:-1, 1:-1]) / 2.0
    pdf_rp = (pdf[0][2:, 1:-1] + pdf[0][1:-1, 1:-1]) / 2.0

    pdf_zm = (pdf[0][1:-1, :-2] + pdf[0][1:-1, 1:-1]) / 2.0
    pdf_zp = (pdf[0][1:-1, 2:] + pdf[0][1:-1, 1:-1]) / 2.0

    A_rm = (jacH.r[1:-1, 1:-1] - jacH.r[:-2, 1:-1]) / (0.5 * dx[0])
    A_rp = (jacH.r[2:, 1:-1] - jacH.r[1:-1, 1:-1]) / (0.5 * dx[0])

    A_zm = (jacH.z[1:-1, 1:-1] - jacH.z[1:-1, :-2]) / (0.5 * dx[1])
    A_zp = (jacH.z[1:-1, 2:] - jacH.z[1:-1, 1:-1]) / (0.5 * dx[1])

    inner = (A_zp * pdf_zp - A_zm * pdf_zm) / (dx[1]) + (
        r_rp * A_rp * pdf_rp - r_rm * A_rm * pdf_rm
    ) / (r[1:-1, 1:-1] * dx[0])

    friction[1:-1, 1:-1] = inner

    # Only has positive flux in r

    A_zm0 = (jacH.z[0, 1:-1] - jacH.z[0, :-2]) / (0.5 * dx[1])
    A_zp0 = (jacH.z[0, 2:] - jacH.z[0, 1:-1]) / (0.5 * dx[1])

    pdf_zm0 = (pdf[0][0, :-2] + pdf[0][0, 1:-1]) / 2.0
    pdf_zp0 = (pdf[0][0, 2:] + pdf[0][0, 1:-1]) / 2.0

    friction[0, 1:-1] = (A_zp0 * pdf_zm0 - A_zm0 * pdf_zp0) / (dx[1])

    return friction


def get_diffusion(hessG: Hess, pdf: Field) -> Tensor:
    dx = pdf.mesh.dx
    r = pdf.mesh.grid[0]

    DrrPr_p = (
        (hessG.rr[2:, 1:-1] + hessG.rr[1:-1, 1:-1])
        / (2.0)
        * (pdf[0][2:, 1:-1] - pdf[0][1:-1, 1:-1])
        / (0.5 * dx[0])
    )
    DrrPr_m = (
        (hessG.rr[1:-1, 1:-1] + hessG.rr[:-2, 1:-1])
        / (2.0)
        * (pdf[0][1:-1, 1:-1] - pdf[0][:-2, 1:-1])
        / (0.5 * dx[0])
    )

    DzzPz_p = (
        (hessG.zz[1:-1, 2:] + hessG.zz[1:-1, 1:-1])
        / (2.0)
        * (pdf[0][1:-1, 2:] - pdf[0][1:-1, 1:-1])
        / (0.5 * dx[1])
    )

    DzzPz_m = (
        (hessG.zz[1:-1, 1:-1] + hessG.zz[1:-1, :-2])
        / (2.0)
        * (pdf[0][1:-1, 1:-1] - pdf[0][1:-1, :-2])
        / (0.5 * dx[1])
    )

    DrzPz_p = hessG.rz

    diffusion = torch.zeros_like(pdf[0])

    return diffusion
