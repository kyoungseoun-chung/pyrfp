#!/usr/bin/env python
"""Tools for training models.
"""
import math
from typing import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from pyrfp.types import LossConfig
from pyrfp.types import LRSchedularOptions


def set_act_funcs(act_f: list[str]) -> list[Callable]:
    """Setting activation function for each layer.
    sigmoid function is called from torch directly since F.sigmoid is
    deprecated.

    Supporting activation functions:
        relu, selu, elu, tanh, sigmoid

    Args:
        act_f (list[str]): list of activation functions

    Returns:
        list: list of activation functions
    """

    act_funcs = []

    for i_f in act_f:
        assert i_f in ACT_F_TYPES, f"Model: Unsupported activation function: {i_f}!"

        act_funcs.append(ACT_F_TYPES[i_f])

    return act_funcs


def diffRelu(input: Tensor) -> Tensor:
    """apply Relu activation function for diagonal diffusion matrix
    components

    This custom activation function is intended to ensure diagonal components
    of the diffusion matrix are always positive
    """
    diff_idx = [3, 5]

    result = input

    for idx in diff_idx:
        # use clone to avoid an inplace operation related error
        result[:, idx] = torch.relu(input[:, idx].clone())

    return result


def set_loss_funcs(loss_config: LossConfig) -> torch.nn.Module:
    """setting loss function used during the training"""

    # Make sure the lower case
    loss_f = loss_config["name"].lower()

    try:
        criterion = LOSS_TYPES[loss_f]
    except KeyError:
        raise KeyError(f"Unsupported loss function: {loss_f}!")

    if loss_f in LOSS_CUSTOM:
        criterion.set_options(loss_config["options"])

    return criterion


def set_lr_schedular(
    schedular: str,
    optimizer: torch.optim.Optimizer,
    opts: LRSchedularOptions | None = None,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Setting learning rate schedular.

    Note:
        - Currently only available with ExponentialLR
    """
    from torch.optim.lr_scheduler import ExponentialLR

    if opts is None:
        lr_schedular = None
    else:
        verbose = opts["verbose"] if "verbose" in opts else False

        assert isinstance(verbose, bool)

        if schedular == "exp_lr":
            gamma = opts["gamma"]

            assert gamma is not None, "Model tools: gamma is not set!"

            lr_schedular = ExponentialLR(
                optimizer=optimizer, gamma=gamma, verbose=verbose
            )
        elif schedular == "cosine_lr":
            """CosineAnnealingLR"""
            t_max = int(opts["t_max"])
            eta_min = opts["eta_min"]

            lr_schedular = CosineAnnealingLR(
                optimizer=optimizer, T_max=t_max, eta_min=eta_min
            )

        elif schedular == None:
            lr_schedular = None
        else:
            msg = f"Unsupported schedular: {schedular}!"
            raise NameError(msg)

    return lr_schedular


class signedMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return _signed_MSELoss(pred, target)


class ALMSE(torch.nn.Module):
    """Asymmetric linear MSE loss function."""

    def __init__(self):
        super().__init__()

    def set_options(self, opts: Optional[dict] = None) -> None:
        """Update loss options."""

        if opts is None:
            self.n_slope = 5
        else:
            self.n_slope = opts["n_slope"]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return _leakyRelu(pred, target, self.n_slope)

    def extra_repr(self) -> str:
        msg = f"n_slope: {self.n_slope}"

        return msg


class AEPMSE(torch.nn.Module):
    """Asymmetric energy preserving loss function."""

    def __init__(self):
        super().__init__()

    def set_options(self, opts: Optional[dict] = None) -> None:
        """Update loss options."""

        if opts is None:
            self.n_slope = 5
            self.weight = 0.01
        else:
            self.n_slope = opts["n_slope"]
            self.weight = opts["weight"]

    def forward(
        self,
        input: Tensor,
        pred: Tensor,
        target: Tensor,
        wiener: Tensor,
    ) -> Tensor:
        # Normal MSE
        loss_1 = _leakyRelu(pred, target, self.n_slope)
        loss_de_dt = ensemble_de_dt(input, pred, wiener)

        return loss_1 + loss_de_dt * self.weight

    def extra_repr(self) -> str:
        msg = f"  n_slope: {self.n_slope}\n" f"  weight: {self.weight}"

        return msg


class EPVIB(torch.nn.Module):
    """Energy preserving VIB loss function.
    Same energy preserving loss for the variational information bottleneck (VIB) model.
    """

    def __init__(self):
        super().__init__()

    def set_options(self, opts: Optional[dict] = None) -> None:
        """Update loss options."""

        if opts is None:
            self.weight_info = 0.01
            self.weight_de_dt = 0.01
        else:
            self.weight_info = opts["weight_info"]
            self.weight_de_dt = opts["weight_de_dt"]

    def extra_repr(self) -> str:
        msg = (
            f"  weight info: {self.weight_info}\n"
            f"  weight de_dt: {self.weight_de_dt}"
        )

        return msg

    def forward(
        self,
        input: Tensor,
        pred: Tensor,
        target: Tensor,
        wiener: Tensor,
        mu: Tensor,
        std: Tensor,
    ) -> Tensor:
        loss_mse = _mse_loss(pred, target)
        loss_info = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(
            1
        ).mean().div(math.log(2))

        loss_de_dt = ensemble_de_dt(input, pred, wiener)

        return loss_mse + self.weight_info * loss_info + self.weight_de_dt * loss_de_dt


class EPMSE(torch.nn.Module):
    """Energy preserving MSE loss."""

    def __init__(self):
        super().__init__()

    def set_options(self, opts: Optional[dict] = None) -> None:
        """Update loss options."""

        if opts is None:
            self.weight = 0.01
        else:
            self.weight = opts["weight"]

    def forward(
        self,
        input: Tensor,
        pred: Tensor,
        target: Tensor,
        wiener: Tensor,
    ) -> Tensor:
        # MSE
        loss_mse = _mse_loss(pred, target)

        # de_dt
        loss_de_dt = ensemble_de_dt(input, pred, wiener)

        return loss_mse + self.weight * loss_de_dt

    def extra_repr(self) -> str:
        msg = f"weight: {self.weight}"

        return msg


class DEPMSE(torch.nn.Module):
    """Deterministic Energy preserving MSE loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: Tensor,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        loss_mse = _mse_loss(pred, target)
        loss_ep = ensemble_e(input, pred)
        loss_mp = ensemble_m(input, pred)

        return loss_mse + (loss_ep + loss_mp) * self.weight

    def set_options(self, opts: dict) -> None:
        """Update loss options."""

        self.weight = opts["weight"]
        self.factor = opts["factor"]
        self.update_every = opts["update_every"]

    def update_weight(self) -> None:
        """Update weight. Multiply by the factor."""

        self.weight *= self.factor

    def extra_repr(self) -> str:
        msg = (
            f"  weight: {self.weight}\n"
            f"  factor: {self.factor}\n"
            f"  weight update every: {self.update_every} epoch"
        )

        return msg


class DMEPMSE(torch.nn.Module):
    """Deterministic momentum and energy preserging MSE loss function."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: Tensor,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        loss_mse = _mse_loss(pred, target)
        loss_ep = ensemble_e(input, pred)
        loss_mp = ensemble_m(input, pred)

        return loss_mse + (loss_ep + loss_mp) * self.weight

    def set_options(self, opts: dict) -> None:
        """Update loss options."""

        self.weight = opts["weight"]
        self.factor = opts["factor"]
        self.update_every = opts["update_every"]

    def update_weight(self) -> None:
        """Update weight. Multiply by the factor."""

        self.weight *= self.factor

    def extra_repr(self) -> str:
        msg = (
            f"  weight: {self.weight}\n"
            f"  factor: {self.factor}\n"
            f"  weight update every: {self.update_every} epoch"
        )

        return msg


def ensemble_e(input: Tensor, pred: Tensor) -> Tensor:
    """Ensemble average of the kinetic energy. Exclude the effect from
    The random noise.
    """

    idx_set = [3, 4, 5, 4, 6, 7, 5, 7, 8]

    dt = 0.01
    dt2 = dt * dt

    uA = 2 * torch.sum(torch.mul(input[:, -3::], pred[:, :3]), dim=1) * dt

    A = torch.sum(torch.mul(pred[:, :3], pred[:, :3]), dim=1) * dt2

    D = torch.sum(torch.mul(pred[:, idx_set], pred[:, idx_set]), dim=1) * dt

    return torch.abs(uA + A + D).mean()


def _mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Custom MSE loss. Nothing special."""

    return torch.mean((pred - target) ** 2)


def ensemble_m(input: Tensor, pred: Tensor):
    dt = 0.01

    uA = torch.sum(torch.mul(input[:, -3::], pred[:, :3]), dim=1) * dt

    # Return ensemble of de/dt
    return torch.abs(uA).mean()


def ensemble_de_dt(
    input: Tensor,
    pred: Tensor,
    wiener: Tensor,
) -> Tensor:
    r"""Calculate mean(de/dt).

    Which is calculated by

    .. math::

        \vert \frac{d e_{kin}}{dt} \vert
        = \vert u_{input} A_{pred}
        + u_{input}\sqrt{D_{pred}}\mathcal{N}(0, \frac{1}{dt}) \vert = 0 \\
        \text{where,} e = \frac{1}{2} u_{input}^2

    Note:
        - Here, \(\mathcal{N}(0, \frac{1}{dt})\) is
          pre-calculated and stored in the target files. -> target[:, -3::]

    """

    idx_set = [[3, 4, 5], [4, 6, 7], [5, 7, 8]]

    diff = torch.zeros_like(wiener)
    # pred 3, 4, 5, 6, 7, 8 => 11, 12, 13, 22, 23, 33
    for i, idx in enumerate(idx_set):
        diff[:, i] = torch.sum(torch.mul(pred[:, idx], wiener), dim=1)

    de_dt = torch.sum(torch.mul(input[:, -3::], pred[:, :3]), dim=1) + torch.sum(
        torch.mul(input[:, -3::], diff), dim=1
    )

    # Return ensemble of de/dt
    return torch.abs(de_dt.mean())


def _signed_MSELoss(pred: Tensor, target: Tensor) -> Tensor:
    """enforce prediction to have same sign with target."""

    # if pred.sign() and target.sign() are different, assign zero
    # then multiply target.sign() to make sure have same sign with the target
    signed_pred = F.relu(pred * target.sign()) * target.sign()

    return torch.mean((signed_pred - target) ** 2)


def _leakyRelu(pred: Tensor, target: Tensor, n_slope: float) -> Tensor:
    check_sign = pred * target.sign() < 0

    # assign strong gradient on negative side
    strong_leakyRelu = torch.nn.LeakyReLU(n_slope)

    pred[check_sign] = strong_leakyRelu(pred[check_sign])

    return torch.mean((pred - target) ** 2)


LOSS_TYPES = {
    "mse": torch.nn.MSELoss(),
    "l1": torch.nn.L1Loss(),
    "sl1": torch.nn.SmoothL1Loss(),
    "signedmse": signedMSE(),
    "almse": ALMSE(),
    "epmse": EPMSE(),
    "aepmse": AEPMSE(),
    "epvib": EPVIB(),
    "depmse": DEPMSE(),
    "dmepmse": DMEPMSE(),
}
"""Interface for the loss function."""

LOSS_CUSTOM = ["epmse", "aepmse", "depmse", "dmepmse"]
"""List of custom loss function."""

ACT_F_TYPES: dict[str, Callable] = {
    "relu": F.relu,
    "selu": F.selu,
    "elu": F.elu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "diffrelu": diffRelu,
}
"""Interface for the activation function."""
