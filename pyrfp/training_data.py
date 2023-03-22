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

import torch
from pyapes.core.mesh import Mesh
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import CylinderBoundary
from pymaxed.maxed import Maxed
from pymaxed.vectors import Vec
from pymytools.special import ellipe
from pymytools.special import ellipk
from torch import Tensor
from torch import vmap

from pyrfp.types import PotentialConfig
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
    config: PotentialConfig

    def __post_init__(self):
        """Post initialization."""

        assert (
            self.mesh.domain.type == "cylinder"
        ), "RosenbluthPotential_RZ only works for cylinder domain."

    def from_mnts(self, mnts: Tensor) -> PotentialReturnType:
        """From moment, obtain the maximum entropy distribution (maxed)"""

        vec = Vec(self.mesh, mnts, 4, [50, 100])
        maxed = Maxed(
            vec,
            lr=self.config["optimizer"]["lr"],
            max_itr=self.config["optimizer"]["max_iter"],
            gtol=self.config["optimizer"]["gtol"],
            xtol=self.config["optimizer"]["xtol"],
            disp=False,
        )

        maxed.solve()

        if not maxed.success:
            return {"pots": None, "success": False}

        dist = maxed.dist_from_coeffs()

        H_pot = Field("H", 1, self.mesh, ...)

        ...


def bc_pots(grid: tuple[Tensor, ...], mask: Tensor, pdf: Tensor) -> Tensor:
    ...


def analytic_potentials(
    grid: tuple[Tensor, ...], pdf: Tensor, potential: str
) -> Tensor:
    """Compute potential using the Green's function."""

    ur = grid[0]
    uz = grid[1]

    hr = ur[1, 0] - ur[0, 0]
    hz = uz[0, 1] - uz[0, 0]

    if potential.lower() == "h":
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

        return vmap_func(ur.flatten(), uz.flatten()).view(*ur.shape)

    elif potential.lower() == "g":
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

        return vmap_func(ur.flatten(), uz.flatten()).view(*ur.shape)

    else:
        raise ValueError("Potential must be either 'H' or 'G'.")
