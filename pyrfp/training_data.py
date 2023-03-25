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

Note:
    - Above performance results are illusion. `vmap` works very well for the native torch functions but not for my `ellipe` and `ellipk` functions (from `pymytools`).
    - Therefore, we will use `Reshaped sum` approach for the computation of the potential boundary conditions.

Therefore, we decided to use the vmap approach for the computation of the boundary conditions.
"""
from dataclasses import dataclass
from math import pi

import pymaxed
import torch
from pyapes.mesh import Mesh
from pyapes.solver.fdc import ScalarOP
from pyapes.solver.fdm import FDM
from pyapes.solver.ops import Solver
from pyapes.solver.tools import FDMSolverConfig
from pyapes.variables import Field
from pyapes.variables.bcs import CylinderBoundary
from pymaxed.maxed import Maxed
from pymaxed.vectors import Vec
from pymytools.logger import logging
from pymytools.logger import markup
from pymytools.logger import Timer
from pymytools.special import ellipe
from pymytools.special import ellipk
from scipy.special import ellipe as s_ellipe
from scipy.special import ellipk as s_ellipk
from torch import Tensor
from torch import vmap

from pyrfp import __version__
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

        self.timer = Timer()

    def from_moments(self, mnts: Tensor) -> PotentialReturnType:
        """From moment, obtain the maximum entropy distribution (maxed)"""

        assert (
            self.optimizer_config is not None
        ), "RosenbluthPotential_rz: optimizer config is not provided."

        assert (
            self.solver_config is not None
        ), "RosenbluthPotential_rz: solver config is not provided."

        logging.info(
            markup(
                f"🚀 Computing Maximum Entropy Distribution  🚀",
                "red",
                "italic",
            )
            + f" (pymaxed v {pymaxed.__version__})"
        )
        logging.info("Target moment: " + markup(f" {mnts}", "blue"))
        logging.info(
            "Domain: "
            + markup(
                f" {self.mesh.lower.tolist()} x {self.mesh.upper.tolist()}", "blue"
            )
        )
        logging.info("Grid: " + markup(f" {list(self.mesh.nx)}", "blue"))

        vec = Vec(self.mesh, mnts, 4, [50, 100])
        maxed = Maxed(
            vec,
            lr=self.optimizer_config["lr"],
            max_itr=self.optimizer_config["max_iter"],
            gtol=self.optimizer_config["gtol"],
            xtol=self.optimizer_config["xtol"],
            disp=False,
        )

        logging.info(markup("Solving for MaxEd...", "yellow"))
        self.timer.start("maxed")
        maxed.solve()
        self.timer.end("maxed")
        logging.info(
            "🔥Done in " + markup(f"{self.timer.elapsed('maxed'):.2f} s", "blue")
        )
        if not maxed.success:
            return {"pots": None, "success": False, "timer": None}

        pdf = maxed.dist_from_coeffs()

        # Density correction
        dr = self.mesh.grid[0][1, 0] - self.mesh.grid[0][0, 0]
        dz = self.mesh.grid[1][0, 1] - self.mesh.grid[1][0, 0]

        density = torch.sum(2.0 * torch.pi * self.mesh.grid[0] * pdf) * dr * dz
        pdf /= density

        return self.from_pdf(pdf)

    def from_pdf(self, pdf: Tensor) -> PotentialReturnType:
        """Compute the Rosenbluth potentials and their derivatives (Jacobian and Hessian) from the given PDF by solving the Poisson equation using the iterative solver (bicgstab scheme).

        Args:
            pdf (Tensor): a given PDF tensor.

        Returns:
            PotentialReturnType: a dictionary containing the potentials and their derivatives.

        Note:
            - The components of the Jacobian and Hessian can be accessed by member attribute of each object, e.g. `PotentialReturnType["jacH"].r` for the radial component of the Jacobian.
            - The `PotentialReturnType["success"]` is `True` only H and G potential both are converged.
        """

        assert (
            self.solver_config is not None
        ), "RosenbluthPotential_rz: solver config is not provided."

        solver = Solver({"fdm": self.solver_config})
        fdm = FDM()

        # Our Scalar field has additional dimension other than the mesh dimension in the leading position.
        # Therefore, if pdf is just like mesh dimension, add one as for the leading dimension
        if pdf.shape[0] != 1 or pdf.shape == self.mesh.nx:
            pdf = pdf.unsqueeze(0)

        logging.info(
            markup(
                f"🚀 Computing Rosenbluth potentials 🚀",
                "red",
                "italic",
            )
            + f" (pyrfp v {__version__})"
        )
        logging.info(
            "Domain: "
            + markup(
                f" {self.mesh.lower.tolist()} x {self.mesh.upper.tolist()}", "blue"
            )
        )
        logging.info("Grid: " + markup(f" {list(self.mesh.nx)}", "blue"))
        logging.info(markup("Evaluating H potential boundary...", "yellow"))
        self.timer.start("H_bc")
        bc_vals = get_analytic_bcs(self.mesh, pdf[0], "H")
        self.timer.end("H_bc")
        logging.info(
            "🔥Done in " + markup(f"{self.timer.elapsed('H_bc'):.2f} s", "blue")
        )

        bc_H = _set_bc_rz(bc_vals)
        H_pot = Field("H", 1, self.mesh, {"domain": bc_H(), "obstacle": None})

        logging.info(markup("Solving H potential...", "yellow"))
        self.timer.start("H_pot")
        solver.set_eq(fdm.laplacian(H_pot) == -8 * pi * pdf)
        solver.solve()
        self.timer.end("H_pot")
        logging.info(
            "🔥Done in " + markup(f"{self.timer.elapsed('H_pot'):.2f} s", "blue")
        )
        h_success = solver.report["converge"]

        logging.info(markup("Evaluating G potential boundary...", "yellow"))
        self.timer.start("G_bc")
        bc_vals = get_analytic_bcs(self.mesh, pdf[0], "G")
        self.timer.end("G_bc")
        logging.info(
            "🔥Done in " + markup(f"{self.timer.elapsed('G_bc'):.2f} s", "blue")
        )

        bc_G = _set_bc_rz(bc_vals)
        G_pot = Field("G", 1, self.mesh, {"domain": bc_G(), "obstacle": None})

        logging.info(markup("Solving G potential...", "yellow"))
        self.timer.start("G_pot")
        solver.set_eq(fdm.laplacian(G_pot) == H_pot())
        solver.solve()
        self.timer.end("G_pot")
        logging.info(
            "🔥Done in " + markup(f"{self.timer.elapsed('G_pot'):.2f} s", "blue")
        )
        g_success = solver.report["converge"]

        logging.info(
            "H: "
            + markup(
                f"{'successful' if h_success else 'fail'}",
                "green" if h_success else "red",
            )
            + ", G: "
            + markup(
                f"{'successful' if g_success else 'fail'}",
                "green" if g_success else "red",
            )
        )
        logging.info(markup("🎉 Finish! 🎉 ", "red"))

        var = Field("container", 1, self.mesh, None)

        return {
            "pots": {
                "H": H_pot()[0],
                "jacH": ScalarOP.jac(var.set_var_tensor(H_pot[0])),
                "G": G_pot()[0],
                "jacG": ScalarOP.jac(var.set_var_tensor(G_pot[0])),
                "hessG": ScalarOP.hess(var.set_var_tensor(G_pot[0])),
                "pdf": pdf[0],
            },
            "success": h_success & g_success,
            "timer": self.timer,
        }


def get_analytic_bcs(
    mesh: Mesh, pdf: Tensor, pot: str, cpu: bool = True
) -> dict[str, Tensor]:
    bcs: dict[str, Tensor] = {}

    for k, v in mesh.d_mask.items():
        if k == "rl":
            pass
        if cpu:
            bcs[k] = analytic_potentials_rz_cpu(
                (mesh.grid[0][v], mesh.grid[1][v]), mesh.grid, pdf, pot
            )

        else:
            bcs[k] = analytic_potentials_rz(
                (mesh.grid[0][v], mesh.grid[1][v]), mesh.grid, pdf, pot
            )

    return bcs


def _set_bc_rz(vals: dict[str, Tensor]) -> CylinderBoundary:
    return CylinderBoundary(
        rl={"bc_type": "neumann", "bc_val": 0.0},
        ru={"bc_type": "dirichlet", "bc_val": vals["ru"]},
        zl={"bc_type": "dirichlet", "bc_val": vals["zl"]},
        zu={"bc_type": "dirichlet", "bc_val": vals["zu"]},
    )


def analytic_potentials_rz_cpu(
    target: tuple[Tensor, ...], grid: tuple[Tensor, ...], pdf: Tensor, potential: str
) -> Tensor:
    """Since vectorized version is too slow, lets try CPU only version using `scipy.special`

    Note:
        - Literal translation of the vectorized version, `analytic_potentials_rz`
    """
    ur = grid[0]
    uz = grid[1]

    hr = ur[1, 0] - ur[0, 0]
    hz = uz[0, 1] - uz[0, 0]

    # What if target[0].shape = torch.Size([64]) and grid[0].shape = torch.Size([64, 128?
    # I need to make 64 x 1 x (64 * 128)
    ur_target = target[0].flatten().unsqueeze(1).repeat(1, ur.numel())
    uz_target = target[1].flatten().unsqueeze(1).repeat(1, ur.numel())

    ur = ur.flatten().repeat(target[0].numel(), 1)
    uz = uz.flatten().repeat(target[0].numel(), 1)
    pdf = pdf.flatten().repeat(target[0].numel(), 1)

    inner = (ur_target + ur) ** 2 + (uz_target - uz) ** 2
    k = 4 * ur_target * ur / inner

    if potential.lower() == "h":
        ek = s_ellipk(k.to(device=torch.device("cpu"))).to(
            device=ur.device, dtype=ur.dtype
        )
        ek[k.eq(1.0)] = 0.0
        return torch.sum(
            torch.nan_to_num(
                8 * ur * pdf * ek / torch.sqrt(inner) * hr * hz,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dim=1,
        ).view(*target[0].shape)
    elif potential.lower() == "g":
        ee = s_ellipe(k.to(device=torch.device("cpu"))).to(
            device=ur.device, dtype=ur.dtype
        )
        ee[k.eq(1.0)] = 0.0
        return torch.sum(
            torch.nan_to_num(
                4 * ur * pdf * ee * torch.sqrt(inner) * hr * hz,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dim=1,
        ).view(*target[0].shape)
    else:
        raise ValueError("Potential must be either H or G")


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
