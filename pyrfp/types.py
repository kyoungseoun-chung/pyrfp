#!/usr/bin/env python3
"""Collection of custom types used in `pyrfp`."""
from typing import TypedDict

import torch
from pyapes.variables.container import Hess
from pyapes.variables.container import Jac
from pymytools.logger import Timer
from torch import Tensor


class Potentials(TypedDict):
    """Collection of potentials and their derivatives."""

    H: Tensor
    """Rosenbluth H potential."""
    jacH: Jac
    """Jacobian of the H potential."""
    G: Tensor
    """Rotational G potential."""
    jacG: Jac
    """Jacobian of the G potential."""
    hessG: Hess
    """Hessian of the H potential."""
    pdf: Tensor
    """Target distribution."""


class PotentialReturnType(TypedDict):
    pots: Potentials | None
    success: bool
    timer: Timer | None


class OptimizerConfig(TypedDict):
    lr: float
    max_iter: int | None
    gtol: float
    xtol: float


class LRSchedularOptions(TypedDict):
    t_max: int
    eta_min: float
    gamma: float | None
    verbose: bool
    start_after: int
    update_every: int


class LRSchedularConfig(TypedDict):
    name: str
    options: LRSchedularOptions


class LossOptions(TypedDict):
    weight: float
    factor: float
    update_every: int


class LossConfig(TypedDict):
    name: str
    options: LossOptions


class TrainingConfig(TypedDict):
    """Training configs"""

    data_dir: str
    """Training data directory"""
    model_dir: str
    """Model output directory"""
    n_epochs: int
    """Number of epochs"""
    layers: list[int]
    """Layer configuration (a number of neurons per layer). e.g. [10, 20, 30]"""
    act_funcs: list[str]
    """Activation function configuration (a name of activation functions per layer). e.g. ['relu', 'elu', 'relu']"""
    lr: float
    """Learning rate"""
    lr_schedular: LRSchedularConfig
    """Learning rate schedular config"""
    loss: LossConfig
    """Loss function"""
    dropout: float
    """Dropout rate"""
    n_batch: int
    """Number of batches"""
    batch_norm: bool
    """If true, perform batch normalization."""
    device: torch.device
    """It true, use GPU"""
    restart: str | None
    """If string is given, restart is the name of pre-trained model. Start training from that model. If None, start from the beginning."""
    data_chunk_size: int
    """The amount of data to be trained at once."""
    tracker_overwrite: bool
    """If true, Tensorboard will overwrite tracker."""
    num_workers: int
    """Number of CPUs for trainloader."""
    save_every: int
    """Save model at every `save_every` epochs."""
    dtype: torch.dtype
    """Data type for training."""
