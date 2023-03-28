#!/usr/bin/env python3
"""Homogeneous relaxation of the RFP equation.
Note:
    - Gave up
"""


from dataclasses import dataclass
from pyapes.mesh import Mesh
from pyapes.variables import Field
from pyapes.solver.ops import Solver
from pyapes.solver.fdm import FDM
from pyapes.solver.fdc import FDC
from pyapes.solver.fdc import jacobian, hessian
from torch import Tensor
from pymaxed.maxed import Maxed, Vec
from pymytools.constants import PI

from pyrfp.training_data import get_analytic_bcs, set_bc_rz
import torch
from pyapes.variables.container import Jac, Hess


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

            new_pdf = pdf()[0] + self.dt * (
                fdc.div(jacH, pdf)[0] + fdc.div(1.0, fdc.diffFlux(hessG, pdf))[0]
            )
            # friction = get_friction(jacH, pdf)
            # my_friction = fdc.div(jacH, pdf)[0]

            # from pymyplot import plt

            # _, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
            # ax[0].plot_surface(pdf.mesh.grid[0], pdf.mesh.grid[1], friction, cmap="jet")
            # ax[1].plot_surface(
            #     pdf.mesh.grid[0], pdf.mesh.grid[1], my_friction, cmap="jet"
            # )

            # plt.show()
            pdf()[0][pdf()[0].le(0.0)] = 0.0
            density = torch.sum(
                new_pdf * 2.0 * PI * self.mesh.R * self.mesh.dx[0] * self.mesh.dx[1]
            )

            new_pdf /= density

            energy = torch.sum(
                0.5
                * (self.mesh.R**2 + self.mesh.Z**2)
                * new_pdf
                * 2.0
                * PI
                * self.mesh.R
                * self.mesh.dx[0]
                * self.mesh.dx[1]
            )

            new_pdf /= torch.sqrt(energy / 1.5)

            pdf.set_var_tensor(new_pdf / density)

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
