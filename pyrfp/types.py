#!/usr/bin/env python3
"""Collection of custom types used in `pyrfp`."""
from typing import TypedDict


class LRSchedularOptions(TypedDict):
    t_max: int
    eta_min: float
    gamma: float
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
    cuda: bool
    """It true, use GPU"""
    restart: str | None
    """If string is given, restart is the name of pre-trained model. Start training from that model. If None, start from the beginning."""
    data_chunk_size: int
    """The amount of data to be trained at once."""
    tracker_overwrite: bool
    """If true, Tensorboard will overwrite tracker."""
    num_workers: int
    """Number of CPUs for trainloader."""
