#!/usr/bin/env python3
"""Tools used in training"""
import random
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from enum import Enum
from typing import Protocol

import torch
from pymytools.diagnostics import DataLoader
from pymytools.diagnostics import file_list_by_pattern
from pymytools.logger import draw_rule
from pymytools.logger import logging
from pymytools.logger import markup
from torch import Tensor
from torch.utils.data.dataloader import DataLoader as TorchDataLoader

from pyrfp.types import TrainingConfig


@dataclass
class Metric:
    """Contains metrics for the trained model."""

    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int):
        """Update metrics."""

        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates


class Stages(Enum):
    """Representation of training stages. Either TRAIN or VAL (validation)."""

    TRAIN = auto()
    VAL = auto()
    NORMAL = auto()


class TrainingTracker(Protocol):
    """Protocol for the Tensorboard tracker `pymytools.diagnostics.DataTracker`."""

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        """Implements logging a epoch-level metric."""

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        """Add batch metric to tensorboard."""

    def add_hparams(
        self, hparams: dict[str, str | float], metrics: dict[str, float]
    ) -> None:
        """Implements logging hyperparamters."""

    def set_stage(self, stage: Stages) -> None:
        """Sets the stage of the tracker."""

    def flush(self) -> None:
        """Flushes the stage of the tracker."""


def training_summary(config: TrainingConfig) -> None:
    """Print out training summary"""

    logging.info(markup("Summary", "red", "bold"))
    draw_rule(character="-")
    for k, v in config.items():
        logging.info(markup(f"{k}: {v}", "blue", "bold"))
    draw_rule(character="-")


@dataclass
class HDFTrainignDataHandler:
    device: torch.device = torch.device("cpu")
    """Device to use for data loading."""
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dl = DataLoader(device=self.device, dtype=self.dtype)
        """Data loader. Dtype of the returned data is `torch.float64` and device is `self.device`."""

    @property
    def is_cuda(self) -> bool:
        """Cuda identifier"""
        return self.device.type == "cuda"

    def get_data_size(self, data_dir: str, addr: str) -> torch.Size:
        """Return data size (`idx`'s column length) of the training data.

        Args:
            data_dir (str): Directory where the training data is stored.
            addr (str): Address used in HDF5 format. Should be posix style.
            idx (int): Column index of the data to be loaded.
        """

        # If address comes with `/`, split it and take the last part.
        dataset_name = addr.rsplit("/", 1)[-1]
        loaded_data = self.dl.read_hdf5(data_dir, dataset_name)[dataset_name]

        return loaded_data.shape

    def get_data_by_chunk(
        self, file_path: str, addr: str, chunk: torch.Tensor
    ) -> Tensor:
        """Load HDF5 data by chunk (`torch.Tensor`)."""

        return self.dl.read_hdf5(file_path, addr)[addr][chunk, :]

    def get_file_list(self, data_dir: str) -> tuple[list[str], list[str]]:
        """Return input and target file list (by pattern)."""

        return file_list_by_pattern(data_dir, "input_*.h5"), file_list_by_pattern(
            data_dir, "target_*.h5"
        )

    def get_input_target_from_dir(
        self,
        input_list: list[str],
        target_list: list[str],
        chunk_size: int,
        is_validation: bool = False,
        ratio: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """From the given input and target file list, load each data by a chunk size/len(input_list) and construct torch trainingloader.

        Args:
            input_list (list[str]): list contains input data pathes
            target_list (list[str]): list contains target data pathes
            chunk_size (int): data chunk size to be returned. If negative, load all data in a file.
            is_validation (bool, optional): If true, this is validation data. Defaults to False.
            ratio (float, optional): Ratio of data to be loaded. Used to adjust total number of files to be loaded. Defaults to 1.0.

        Returns:
            tuple[Tensor, Tensor]: loaded tensor of input and target
        """

        assert len(input_list) == len(
            target_list
        ), "DataHandler: input and target list should have the same length."

        assert ratio <= 1.0, "DataHandler: ratio should be less than or equal to 1.0."

        n_files = int(len(input_list) * ratio)

        if n_files == 0:
            n_files = 1

        chunk_size = int(chunk_size / n_files)

        # Shuffle input and target list to avoid bias
        random.shuffle(input_list)
        random.shuffle(target_list)

        if is_validation:
            prefix = "LOAD DATA - VALID: "
        else:
            prefix = "LOAD DATA - TRAIN: "

        msg = markup(
            prefix + f"from {n_files} files, loading chunk size of ({chunk_size}) each",
            "blue",
            "bold",
        )
        logging.info(msg)

        inputs: list[Tensor] = []
        targets: list[Tensor] = []

        for i, t in zip(input_list[:n_files], target_list[:n_files]):
            column_size = self.get_data_size(i, "inputs")

            assert (
                chunk_size <= column_size[0]
            ), f"DataHandler: chunk size ({chunk_size}) should be less than or equal to the column size of the given data ({column_size[0]})."

            if chunk_size > 0:
                # chunk = torch.randperm(column_size[0], device=self.device)[:chunk_size]
                chunk = torch.randperm(column_size[0], device=torch.device("cpu"))[
                    :chunk_size
                ]
                chunk.to(self.device)

            else:
                # chunk = torch.randperm(column_size[0], device=self.device)
                chunk = torch.randperm(column_size[0], device=torch.device("cpu"))
                chunk.to(self.device)

            inputs.append(self.get_data_by_chunk(i, "inputs", chunk))
            targets.append(self.get_data_by_chunk(t, "targets", chunk))

        return torch.vstack(inputs), torch.vstack(targets)

    def get_trainlaoder(
        self, inputs: Tensor, targets: Tensor, batch_size: int, num_workers: int
    ) -> TorchDataLoader:
        """From inputs and targets, construct torch Dataloader.

        Args:
            inputs (Tensor): Input data
            targets (Tensor): Target data
            batch_size (int): Batch size
            num_workers (int): Number of workers. Currently, only used when device is cuda. (in cpu mode, utilize only 1 processor)
        """

        td = torch.utils.data.TensorDataset(inputs, targets)  # type: ignore

        # Set pin memory to True if device is cuda
        pm = True if self.is_cuda else False
        dl = TorchDataLoader(
            td,
            batch_size=batch_size,
            pin_memory=pm,
            shuffle=True,
            num_workers=num_workers if self.is_cuda else 0,
        )

        return dl
