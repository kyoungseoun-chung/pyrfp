#!/usr/bin/env python3
"""Training data generation for the Rosenbluth Fokker Planck equation.

Here, we compute the Rosenbluth potentials using the maximum entropy distribution and the Green's function (for the boundary conditions).

In this case, the computation of the boundary condition (due to the evaluation of the Greens' function) is the most time consuming part.

To have a solution of the Green's function, we can take several approaches:
    - Naive summation
    - Loop over vectorized sum
    - Reshape all tensors and use the vectorized sum
    - Use vmap

The performance of the abovementioned approaches are following:
    - For 32 x 64 grid
    ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃  Schemes   ┃ Elapsed time (s)     ┃
    ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ Naive sum  │ 55.80528954183683    │
    │  Loop sum  │ 0.027860709000378847 │
    │Reshaped sum│ 0.03574174991808832  │
    │  Vmap sum  │ 0.009342916077002883 │
    └────────────┴──────────────────────┘

Therefore, we decided to use the vmap approach for the computation of the boundary conditions.
"""
from dataclasses import dataclass
from math import pi
from typing import Callable

import torch
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.solver.tools import FDMSolverConfig
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import CylinderBoundary
from pyapes.tools.spatial import ScalarOP
from pymaxed.maxed import Maxed
from pymaxed.vectors import Vec
from pymytools.special import ellipe
from pymytools.special import ellipk
from torch import Tensor
from torch import vmap

from pyrfp.types import OptimizerConfig
from pyrfp.types import PotentialReturnType


@dataclass
class RosenbluthPotentials_RZ:
    """Compute the Rosenbluth potentials.

    Note:
        - This object is only designed to be worked in the `rz` coordinate for the moment order up to 4th order.
        - More Generalized case will be a future plan.

    Args:
        mesh (Mesh): Mesh object.
    """

    mesh: Mesh
    optimizer_config: OptimizerConfig | None = None
    solver_config: FDMSolverConfig | None = None

    def __post_init__(self):
        """Post initialization."""

        assert (
            self.mesh.domain.type == "cylinder"
        ), "RosenbluthPotential_RZ only works for cylinder domain."

    def from_mnts(self, mnts: Tensor) -> PotentialReturnType:
        """From moment, obtain the maximum entropy distribution (maxed)"""

        assert (
            self.optimizer_config is not None
        ), "RosenbluthPotential_rz: optimizer config is not provided."

        assert (
            self.solver_config is not None
        ), "RosenbluthPotential_rz: solver config is not provided."

        vec = Vec(self.mesh, mnts, 4, [50, 100])
        maxed = Maxed(
            vec,
            lr=self.optimizer_config["lr"],
            max_itr=self.optimizer_config["max_iter"],
            gtol=self.optimizer_config["gtol"],
            xtol=self.optimizer_config["xtol"],
            disp=False,
        )

        maxed.solve()

        if not maxed.success:
            return {"pots": None, "success": False}

        pdf = maxed.dist_from_coeffs()

        return self.from_pdf(pdf)

    def from_pdf(self, pdf: Tensor) -> PotentialReturnType:
        assert (
            self.solver_config is not None
        ), "RosenbluthPotential_rz: solver config is not provided."

        solver = Solver({"fdm": self.solver_config})
        fdm = FDM()

        # Our Scalar field has additional dimension other than the mesh dimension in the leading position.
        # Therefore, if pdf is just like mesh dimension, add one as for the leading dimension
        if pdf.shape[0] != 1 or pdf.shape == self.mesh.nx:
            pdf = pdf.unsqueeze(0)

        bc_H = _set_bc_rz(pdf, bc_H_pots)
        H_pot = Field("H", 1, self.mesh, {"domain": bc_H(), "obstacle": None})

        solver.set_eq(fdm.laplacian(H_pot) == -8 * pi * pdf)

        bc_G = _set_bc_rz(pdf, bc_G_pots)
        G_pot = Field("G", 1, self.mesh, {"domain": bc_G(), "obstacle": None})

        solver.set_eq(fdm.laplacian(G_pot) == H_pot())

        return {
            "pots": {
                "H": H_pot()[0],
                "jacH": ScalarOP.jac(H_pot),
                "G": G_pot()[0],
                "jacG": ScalarOP.jac(G_pot),
                "hessG": ScalarOP.hess(G_pot),
            },
            "success": True,
        }


def _set_bc_rz(pdf: Tensor, bc_func: Callable) -> CylinderBoundary:
    return CylinderBoundary(
        rl={"bc_type": "neumann", "bc_val": 0.0},
        ru={"bc_type": "dirichlet", "bc_val": bc_func, "bc_val_opt": {"pdf": pdf}},
        zl={"bc_type": "dirichlet", "bc_val": bc_func, "bc_val_opt": {"pdf": pdf}},
        zu={"bc_type": "dirichlet", "bc_val": bc_func, "bc_val_opt": {"pdf": pdf}},
    )


def bc_H_pots(
    grid: tuple[Tensor, ...], mask: Tensor, _, opt: dict[str, Tensor]
) -> Tensor:
    """Dirichlet boundary condition for the H potential computed by the analytic solution of the Rosenbluth potential."""

    pdf = opt["pdf"]

    target = (grid[0][mask], grid[1][mask])

    return analytic_potentials_rz(target, grid, pdf, "H")


def bc_G_pots(
    grid: tuple[Tensor, ...], mask: Tensor, _, opt: dict[str, Tensor]
) -> Tensor:
    """Dirichlet boundary condition for the G potential computed by the analytic solution of the Rosenbluth potential."""

    pdf = opt["pdf"]

    target = (grid[0][mask], grid[1][mask])

    return analytic_potentials_rz(target, grid, pdf, "G")


def analytic_potentials_rz(
    target: tuple[Tensor, ...], grid: tuple[Tensor, ...], pdf: Tensor, potential: str
) -> Tensor:
    r"""Compute analytic solution of the Rosenbluth potential using the Green's formulation in the axisymmetric phase space.
    The integration is done with the elliptic integral of the first and second kind.

    The H potential is computed by:

    .. math::

        \left. H(u_\perp, u_\parallel) \right|_{\Omega_{BC}} = 8\int_{0}^{u_\perp^{max}}
        \int_{-u_\parallel^{max}}^{u_\parallel^{max}}
        u_\perp \frac{f^{MED}_{\vec{\lambda}}(u_\perp', u_\parallel')K[k(u_\perp, u_\parallel;u_\perp', u_\parallel')]}
        {\sqrt{(u_\perp + u_\perp')^2 + (u_\parallel - u_\parallel')^2}}du_\perp'du_\parallel' \quad \text{and}

    And the G potential is:

    .. math::

        \left. G(u_\perp, u_\parallel) \right|_{\Omega_{BC}} &=& 4\int_{0}^{u_\perp^{max}}
        \int_{-u_\parallel^{max}}^{u_\parallel^{max}}
        u_\perp f^{MED}_{\vec{\lambda}}(u_\perp', u_\parallel')E[k(u_\perp, u_\parallel;u_\perp', u_\parallel')] \nonumber\\
        &&\times \sqrt{(u_\perp + u_\perp')^2 + (u_\parallel - u_\parallel')^2}du_\perp'du_\parallel'

    Examples:

        >>> r = torch.linspace(0, 1, 32)
        >>> z = torch.linspace(-1, 1, 64)
        >>> grid = torch.meshgrid(r, z, indexing="ij")
        >>> pdf = torch.tensor(...)
        >>> H_pot = analytic_potentials_rz(grid, grid, pdf, "H") # H potential in the whole domain, target == grid
        >>> G_pot = analytic_potentials_rz(grid, grid, pdf, "G") # G potential in the whole domain, target == grid

    Args:
        target (tuple[Tensor, ...]): The target phase space to evaluate the potentials.
        grid (tuple[Tensor, ...]): The grid of the phase space.
        pdf (Tensor): The maximum entropy distribution.
        potential (str): The potential to compute. Either `H` or `G`.

    Returns:
        Tensor: The potential evaluated at the grid in the domain.

    Note:
        - The target and grid are separated and `target` $\in$ `grid`. This is intended to utilize this function for the evaluation of the boundary condition.
        - If you want to solve the potential in the whole domain, you can use the `target` and `grid` as the same.
    """

    ur = grid[0]
    uz = grid[1]

    hr = ur[1, 0] - ur[0, 0]
    hz = uz[0, 1] - uz[0, 0]

    if potential.lower() == "h":
        # Compute H potential
        vmap_func = vmap(
            lambda x, y: torch.sum(
                torch.nan_to_num(
                    ellipk(
                        4 * x * ur / ((x + ur) ** 2 + (y - uz) ** 2),
                    )
                    * 8
                    * ur
                    * pdf
                    / (torch.sqrt((x + ur) ** 2 + (y - uz) ** 2))
                    * hr
                    * hz,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
        )

        return vmap_func(target[0].flatten(), target[1].flatten()).view(
            *target[0].shape
        )

    elif potential.lower() == "g":
        # Compute G potential
        vmap_func = vmap(
            lambda x, y: torch.sum(
                torch.nan_to_num(
                    ellipe(
                        4 * x * ur / ((x + ur) ** 2 + (y - uz) ** 2),
                    )
                    * 4
                    * ur
                    * pdf
                    * (torch.sqrt((x + ur) ** 2 + (y - uz) ** 2))
                    * hr
                    * hz,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
        )

        return vmap_func(target[0].flatten(), target[1].flatten()).view(
            *target[0].shape
        )

    else:
        raise ValueError("Potential must be either 'H' or 'G'.")
