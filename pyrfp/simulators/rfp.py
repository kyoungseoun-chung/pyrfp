#!/usr/bin/env python3
"""Homogeneous relaxation of the RFP equation.
Note:
    - Gave up
"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from pyapes.mesh import Mesh
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
from pymytools.logger import Timer
from torch import Tensor

from pyrfp.training_data import get_analytic_bcs
from pyrfp.training_data import set_bc_rz


@dataclass
class RFP_Base(ABC):
    mesh: Mesh
    dist: Tensor | None

    def __post_init__(self) -> None:
        # Set timer
        self.timer = Timer()
        # Set solver
        self.solver = Solver(
            {
                "fdm": {
                    "method": "bicgstab",
                    "tol": 1e-5,
                    "max_it": 1000,
                    "report": False,
                }
            }
        )
        self.fdm = FDM()
        self.rfp = RFP()

    @abstractmethod
    def run(self, *_) -> None:
        """Run simulation."""
        ...

    def potentials_from_pdf(self, pdf: Field) -> tuple[Field, Field, Jac, Hess]:
        """Obtain the Rosenbluth potentials from the given PDF."""

        self.timer.start("H_bc")
        bc_vals = get_analytic_bcs(self.mesh, pdf[0], "H")
        self.timer.end("H_bc")
        bc_H = set_bc_rz(bc_vals)
        H_pot = Field("H", 1, self.mesh, {"domain": bc_H(), "obstacle": None})
        self.solver.set_eq(self.fdm.laplacian(H_pot) == -8 * PI * pdf())
        self.timer.start("H_sol")
        self.solver.solve()
        self.timer.end("H_sol")
        jacH = jacobian(H_pot)

        self.timer.start("G_bc")
        bc_vals = get_analytic_bcs(self.mesh, pdf[0], "G")
        self.timer.end("G_bc")
        bc_G = set_bc_rz(bc_vals)
        G_pot = Field("G", 1, self.mesh, {"domain": bc_G(), "obstacle": None})
        self.solver.set_eq(self.fdm.laplacian(G_pot) == H_pot())
        self.timer.start("G_sol")
        self.solver.solve()
        self.timer.end("G_sol")
        hessG = hessian(G_pot)

        return H_pot, G_pot, jacH, hessG


@dataclass
class RFP_RZ(RFP_Base):
    dt: float
    """Time step."""
    n_itr: int
    """Number of iterations."""

    def set_initial_pdf(self, mnts: list[float] | Tensor) -> None:
        vec = Vec(self.mesh, mnts, 4, [50, 100])
        maxed = Maxed(vec, lr=1.0, max_itr=200, gtol=1e-6, xtol=1e-6, disp=False)
        maxed.solve()

        if maxed.success:
            assert maxed.dist is not None
            self.dist = maxed.dist
        else:
            raise RuntimeError(
                f"RFP_RZ: Failed to solve for initial PDF from the given set of moments ({mnts=})."
            )

    def run(self, no_update: bool = False) -> Field:
        """Run the RFP simulation.

        Args:
            no_update (bool, optional): If True, the initial PDF is not updated. Defaults to False. This options is intended for obtaining the computational cost without updating the PDF.
        """
        assert self.dist is not None, "RFP_RZ: Initial PDF is not set."

        pdf = Field("pdf", 1, self.mesh, None)
        pdf.set_var_tensor(self.dist)

        for _ in range(self.n_itr):
            _, _, jacH, hessG = self.potentials_from_pdf(pdf)

            if not no_update:
                pdf[0] = pdf[0] + self.dt * (
                    -self.rfp.friction(jacH, pdf) + 0.5 * self.rfp.diffusion(hessG, pdf)
                )

                pdf[0][pdf[0].le(1e-10)] = 0.0
                den = pdf.volume_integral()
                pdf.set_var_tensor(pdf[0] / den)

        return pdf
