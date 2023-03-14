#!/usr/bin/env python3
"""Potential data training using the MLP architecture.
Refactored version of `pystops_ml.training.rps_training_slim`
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from pymytools.diagnostics import DataLoader
from pymytools.diagnostics import DataSaver
from pymytools.diagnostics import DataTracker
from pymytools.diagnostics import is_dir
from pymytools.logger import draw_rule
from pymytools.logger import logging
from pymytools.logger import markup
from pymytools.logger import timer

from pyrfp.model_tools import DEPMSE
from pyrfp.model_tools import DMEPMSE
from pyrfp.model_tools import set_loss_funcs
from pyrfp.model_tools import set_lr_schedular
from pyrfp.training_model import NLinearLayerModel
from pyrfp.training_runner import Runner
from pyrfp.training_tools import HDFTrainignDataHandler
from pyrfp.training_tools import Stages
from pyrfp.training_tools import training_summary
from pyrfp.types import TrainingConfig


@dataclass
class TrainingBasis:
    """Training basis class"""

    config: TrainingConfig
    """Training configuration data"""

    def __post_init__(self):
        # Print summary
        training_summary(self.config)

        # ------------------------------------------------------------------ #
        # Check training data directory.
        self.data_dir = Path(self.config["data_dir"]).as_posix()
        self.model_dir = Path(self.config["model_dir"]).as_posix()
        if not is_dir(self.data_dir, create=False):
            raise NotADirectoryError(
                f"Training: a given data directory ({self.data_dir}) not found."
            )

        # Check model save location
        # If not exists, create folder
        is_dir(self.model_dir, create=True, verbose=1)

        # ------------------------------------------------------------------ #
        # Device setup
        # Cuda status
        # Set device accordingly
        self.device = self.config["device"]
        self.dtype = self.config["dtype"]

        # Add data handler
        self.dh = HDFTrainignDataHandler(device=self.device, dtype=self.dtype)
        # Add data saver
        self.ds = DataSaver(self.model_dir)
        # Add data loader
        self.dl = DataLoader(self.dtype, self.device)

        # Check number of workers for the trainloader
        self.num_workers = self.config["num_workers"]

        # ------------------------------------------------------------------ #
        # Training preps
        # A number of batchs actually used in the training
        self.n_batchs = self.config["n_batch"]
        # Set a number of main epoch
        self.n_epochs = self.config["n_epochs"]
        # A number of data chunk size loaded at once
        self.n_data_chunk = self.config["data_chunk_size"]
        # Restart flag
        self.tr_restart = self.config["restart"]

        # ------------------------------------------------------------------ #
        # Architecture setups
        self.tr_lr = self.config["lr"]
        self.tr_lr_schedular = self.config["lr_schedular"]

        self.act_funcs = self.config["act_funcs"]
        self.layers = self.config["layers"]
        self.dropout = self.config["dropout"]
        self.batch_norm = self.config["batch_norm"]
        # Defining loss is bit changed!
        self.loss = self.config["loss"]

        # ------------------------------------------------------------------ #
        # Data tracker
        self.tracker_log_dir = Path(self.model_dir, Path("./run/"))
        self.tracker_overwrite = self.config["tracker_overwrite"]

        self.tracker = DataTracker(
            str(self.tracker_log_dir), overwrite=self.tracker_overwrite
        )

        # Timer
        self.timer = timer


class ParticleTraining(TrainingBasis):
    """Training only runs at the local system. Therefore, intended to work
    one CUDA device or local CPUs.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def get_file_lists(
        self, is_validation: bool, verbose: int = 0
    ) -> tuple[list[str], list[str]]:
        """Get data file lists with a certain pattern.

        Args:
            is_validation (bool): flag to decide which file list returned.
            verbose (int): report level flag

        Returns:
            tuple[List[str], List[str]]: input and target file lists
        """

        if is_validation is False:
            data_dir = Path(self.data_dir, "training/").as_posix()
        else:
            data_dir = Path(self.data_dir, "validation/").as_posix()

        input_list, target_list = self.dh.get_file_list(data_dir)

        if verbose > 0:
            msg = markup(
                f"from {data_dir}, loading {len(input_list)} input files and {len(target_list)} target files",
                "blue",
                "bold",
            )
            logging.info(msg)

        return input_list, target_list

    def build_architecture(self, input_size: int, target_size: int):
        """Build MLP architecture based on `pyrfp.training_model.NLinearLayerModel`

        Args:
            input_size (int): input size
            target_size (int): target size

        Returns:
            torch.nn.Module: Neural net architecture
        """

        # construct SNN architecture

        logging.info(markup("configuring ann architecture...", "blue", "bold"))
        architecture = NLinearLayerModel(
            target_size,
            input_size,
            self.layers,
            self.act_funcs,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout,
            dtype=self.dtype,
        )

        if self.tr_restart is not None:
            # This only work with state dict. No entire model load

            assert isinstance(
                self.tr_restart, str
            ), "Training: restart option must be a model path"

            state_dict_loc = Path(self.model_dir, Path(self.tr_restart)).as_posix()

            logging.info(
                markup(
                    f"ATTENTION! restart training from {state_dict_loc}",
                    "yellow",
                    "bold",
                )
            )

            architecture = self.dl.read_state_dict(architecture, state_dict_loc)

        architecture.to(self.device)
        architecture.train()

        return architecture

    def set_schedular(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.optim.lr_scheduler._LRScheduler | None, float, int]:
        """Set lr schedular."""

        # Learning-rate schedular
        schedular = set_lr_schedular(
            self.tr_lr_schedular["name"],
            optimizer,
            self.tr_lr_schedular["options"],
        )

        if schedular is not None:
            # Enable schedular after schedular_start_after*n_epochs.
            # If option is not given, use 30% as default
            try:
                start_after = self.tr_lr_schedular["options"]["start_after"]
            except KeyError:
                start_after = 0.3

                msg = markup(
                    f"Schedular start_after value entry not given! Will use default value {start_after}!",
                    "magenta",
                    "bold",
                )
                logging.warning(msg)

            try:
                update_every = self.tr_lr_schedular["options"]["update_every"]
            except KeyError:
                update_every = 10

                msg = markup(
                    f"Schedular update_every value entry not given! Will use default value {update_every}!",
                    "magenta",
                    "bold",
                )
                logging.warning(msg)

        else:
            start_after = 1
            update_every = 1

        return schedular, start_after, update_every

    def excute(self) -> torch.nn.Module:
        """Excute training.

        Returns:
            torch.nn.Module: trained model
        """

        logging.info(markup("Preparing training..."), "blue", "bold")

        # Training data list
        tr_input_list, tr_target_list = self.get_file_lists(
            is_validation=False, verbose=1
        )
        # Validation data list
        val_input_list, val_target_list = self.get_file_lists(
            is_validation=True, verbose=1
        )

        # Get data size to construct architecture
        n_tr_files = len(tr_input_list)

        # Check one of data's size
        tr_input_size = self.dh.get_data_size(tr_input_list[0], "inputs")
        tr_target_size = self.dh.get_data_size(tr_target_list[0], "targets")

        tr_in_size = tr_input_size[1]
        tr_in_rows = tr_input_size[0]
        tr_out_size = tr_target_size[1]

        data_chunk = tr_in_rows if self.n_data_chunk is None else self.n_data_chunk

        # Validation data chunk size
        val_data_chunk = int(len(val_input_list) / n_tr_files * data_chunk)

        # Construct architecture
        architecture = self.build_architecture(tr_in_size, 9)

        # Optimizer
        optimizer = torch.optim.Adam(architecture.parameters(), lr=self.tr_lr)

        # Loss function
        criterion = set_loss_funcs(self.loss)

        (
            schedular,
            schedular_start_after,
            schedular_update_every,
        ) = self.set_schedular(optimizer)

        draw_rule(character="-")

        # Check nn architecture.
        logging.info(markup("checking MLP architecture...\n", "blue", "bold"))
        # Neural Net layer info
        print(architecture)
        # Optimizer info
        print(optimizer)
        # Scheduler info (if exists)
        if schedular is not None:
            schedular_name = self.tr_lr_schedular["name"]
            print(schedular_name + " (")
            for k, v in self.tr_lr_schedular["options"].items():
                print(f"    {k} : {v}")
            print(")")
        # Loss function info
        print(criterion)
        draw_rule(character="-")
        logging.info(markup("training starts! \n", "blue", "bold"))

        logging.info(markup(f"{str(self.device)} mode in use.", "blue", "bold"))

        self.timer.start("ann")

        saved = False

        try:
            for epoch in range(self.n_epochs):
                print("")

                tr_inputs, tr_targets = self.dh.get_input_target_from_dir(
                    tr_input_list, tr_target_list, data_chunk, False
                )
                val_inputs, val_targets = self.dh.get_input_target_from_dir(
                    val_input_list, val_target_list, val_data_chunk, True
                )

                # Training loader
                tr_tl = self.dh.get_trainlaoder(
                    tr_inputs, tr_targets, self.n_batchs, self.num_workers
                )

                # Validation loader
                val_tl = self.dh.get_trainlaoder(
                    val_inputs, val_targets, val_data_chunk, True
                )

                # Construct runners with loader
                training_runner = Runner(
                    tr_tl, architecture, criterion, optimizer, self.device
                )
                # Validation process doesn't require optimizer
                validation_runner = Runner(
                    val_tl, architecture, criterion, None, self.device
                )

                # Need better idea...
                if self.loss["name"] == "depmse" or self.loss["name"] == "dmepmse":
                    # Update factors (and skip epoch == 0)
                    if epoch % self.loss["options"]["update_every"] == 0 and epoch > 1:
                        assert isinstance(criterion, DEPMSE | DMEPMSE)
                        criterion.update_weight()

                    # Double check..
                    logging.info(
                        markup(
                            f"using loss_ep weight of {training_runner.compute_loss.weight}",
                            "yellow",
                            "bold",
                        )
                    )

                # Training starts!
                self.tracker.set_stage(Stages.TRAIN)
                training_runner.run(f"TRAIN |{epoch+1}/{self.n_epochs}|", self.tracker)

                self.tracker.set_stage(Stages.VAL)
                validation_runner.run(
                    f"VALID |{epoch+1}/{self.n_epochs}|", self.tracker
                )

                self.tracker.set_stage(Stages.NORMAL)
                self.tracker.add_epoch_metric(
                    "tr_count", training_runner.run_count, epoch
                )
                self.tracker.add_epoch_metric(
                    "val_count", validation_runner.run_count, epoch
                )

                # schedular resides outside since this is nested training
                # after 30% of n_epochs, decay learning rate
                if epoch > int(self.n_epochs * schedular_start_after):
                    # decay every 10 epochs
                    if epoch % schedular_update_every == 0 and schedular is not None:
                        print("")
                        schedular.step()

                # Save model during epoch
                # If self.config["save_every"] < 0, model will not be saved
                if (
                    epoch % self.config["save_every"] == 0
                    and epoch != 0
                    and self.config["save_every"] > 0
                ):
                    self.ds.save_model(architecture, f"model_{epoch}")

        except KeyboardInterrupt:
            saved = self.finalize(architecture)

        if saved is False:
            self.finalize(architecture)

        return architecture

    def finalize(self, architecture: torch.nn.Module) -> bool:
        """Finalize the training."""

        # Save model in case of keyboard interruption
        # In this case, model will be saved as "model.pth"
        self.ds.save_model(architecture, "model")

        self.timer.end("ann")
        elapsed_time = self.timer.elapsed("ann")

        logging.info(
            markup(rf"done in {elapsed_time} [s] :kissing_heart:", "green", "bold")
        )

        return True
