#!/usr/bin/env python3
"""Training runner"""
from typing import Any

import torch
from pymytools.diagnostics import DataTracker
from pymytools.progress import ProgressBar
from torch.utils.data.dataloader import DataLoader

from pyrfp.model_tools import ensemble_de_dt
from pyrfp.model_tools import ensemble_e
from pyrfp.model_tools import LOSS_CUSTOM
from pyrfp.training_tools import Metric


class Runner:
    def __init__(
        self,
        loader: DataLoader,
        model: torch.nn.Module,
        criterion: Any,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
    ):
        """Run training.

        Args:
            loader (DataLoader): torch data loader
            model (torch.nn.Module): torch neural net architecture
            criterion (Any): loss function
            optimizer (Optional[torch.optim.Optimizer]): torch optimizer
                If it is None, runner runs for the validation
            device (torch.device): device to run the model
        """

        self.run_count = 0
        self.loader = loader
        self.loader_size = len(self.loader)
        self.model = model
        self.loss_function_name = criterion.__class__.__name__.lower()
        self.compute_loss = criterion
        self.optimizer = optimizer
        self.device = device
        self.accuracy_metric = Metric()

        self.target_batches: list[list[Any]] = []
        self.pred_batches: list[list[Any]] = []

        self.tol = 1e-1

    def run(self, desc: str, tracker: DataTracker) -> None:
        """Run training or validation.

        Args:
            desc (str): training description.
            tracker (Tracker): training tracker.
        """

        pbar = ProgressBar(self.loader, desc=desc, mininterval=0)

        for data, target in pbar:
            # forward
            loss, acc_abs, acc_rel, acc_eng = self._run_single(data, target)

            # Compute metrics
            tracker.add_batch_metric("loss", loss, self.run_count)
            tracker.add_batch_metric("accuracy_abs", acc_abs, self.run_count)
            tracker.add_batch_metric("accuracy_rel", acc_rel, self.run_count)

            if acc_eng is not None:
                tracker.add_batch_metric("accuracy_eng", acc_eng, self.run_count)
                desc_mod = desc + f" Acc-eng: {acc_eng:.2e} "
            else:
                desc_mod = desc + f" Acc-rel: {acc_rel:.2e} "

            if self.optimizer is not None:
                desc_mod += f" lr: {self.optimizer.param_groups[0]['lr']:.2e}"  # type: ignore
            else:
                desc_mod += f"             "
            pbar.set_description(desc_mod)

    def _run_single(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[float, float, float, float | None]:
        """Feed forward process. This part is separated to handle duplication
        both in training and validation process.

        """
        self.run_count += 1

        if self.device == "cuda":
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
        else:
            data, target = data.to(self.device), target.to(self.device)

        batch_size = target.numel()

        # Inference
        prediction = self.model(data)

        target_size = target.shape[1]

        if self.loss_function_name in LOSS_CUSTOM:
            if target_size <= 9:
                msg = f"Training: target size has to be 12. But received {target_size}!"

                raise RuntimeError(msg)

            target_coeffs = target[:, :-3]
            wiener = target[:, -3::]

            # NOTE: Need to add dm/dt
            if self.loss_function_name == "depmse":
                loss = self.compute_loss(data, prediction, target_coeffs)
            else:
                loss = self.compute_loss(data, prediction, target_coeffs, wiener)

            if self.loss_function_name == "depmse":
                batch_accuracy_eng = ensemble_e(data, prediction).cpu().detach().numpy()
            else:
                batch_accuracy_eng = (
                    ensemble_de_dt(data, prediction, wiener).cpu().detach().numpy()
                )
        else:
            if target_size == 12:
                target_coeffs = target[:, :-3]
            elif target_size == 9:
                target_coeffs = target
            else:
                msg = (
                    "Wrong target size!. Should be either 9 or 12 "
                    f"but received {target_size}"
                )
                raise RuntimeError(msg)

            loss = self.compute_loss(prediction, target_coeffs)
            batch_accuracy_eng = None

        data_size = target_coeffs.numel()

        # percentage of diff below tolerance...
        diff = torch.abs(torch.div((target_coeffs - prediction), target_coeffs))
        batch_accuracy_abs = diff.sum().cpu().detach().numpy() / data_size

        batch_accuracy_rel = (diff < self.tol).sum().cpu().detach().numpy() / data_size

        self.accuracy_metric.update(batch_accuracy_rel, batch_size)

        # Back propagation
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        batch_loss = loss.cpu().detach().numpy()

        return (
            batch_loss,
            batch_accuracy_abs,
            batch_accuracy_rel,
            batch_accuracy_eng,
        )
