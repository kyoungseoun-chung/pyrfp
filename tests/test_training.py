#!/usr/bin/env python3
"""Test training module"""
import torch
from pymytools.diagnostics import DataLoader

from pyrfp.training import ParticleTraining
from pyrfp.training_model import NLinearLayerModel
from pyrfp.types import TrainingConfig

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()  # type: ignore
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

CONFIG: TrainingConfig = {
    "data_dir": "./tests/test_data",
    "model_dir": "./tests/test_data",
    "n_epochs": 1,
    "layers": [10, 10, 10],
    "act_funcs": ["relu", "tanh", "elu"],
    "lr": 0.001,
    "lr_schedular": {
        "name": "cosine_lr",
        "options": {
            "gamma": None,
            "t_max": 30,
            "eta_min": 1e-6,
            "verbose": True,
            "start_after": 0,
            "update_every": 1,
        },
    },
    "loss": {
        "name": "depmse",
        "options": {"weight": 1e-2, "factor": 1.0, "update_every": 1},
    },
    "dropout": 0.0,
    "n_batch": 1,
    "batch_norm": False,
    "device": DEVICE,
    "restart": "model.pth",
    "data_chunk_size": 10,
    "tracker_overwrite": False,
    "num_workers": 0,
    "save_every": -1,
    "dtype": torch.float32,
}


def test_architecture() -> None:
    """Just functionality test. No meaning here."""
    architecture = NLinearLayerModel(
        9,
        8,
        CONFIG["layers"],
        CONFIG["act_funcs"],
        CONFIG["batch_norm"],
        CONFIG["dropout"],
    )

    assert architecture.n_hidden == 3
    assert architecture.act_func == CONFIG["act_funcs"]
    assert architecture.layers == [8] + CONFIG["layers"]

    pred = architecture(torch.rand(10, 8))

    assert pred.shape == torch.Size([10, 9])

    # Load state dict
    dl = DataLoader(
        dtype=CONFIG["dtype"],
        device=CONFIG["device"],
    )
    architecture = dl.read_state_dict(architecture, "./tests/test_data/model.pth")

    pred = architecture(torch.rand(10, 8))

    assert pred.shape == torch.Size([10, 9])


def test_training() -> None:
    training = ParticleTraining(CONFIG)
    training.excute()

    # Run once for for CPU
    CONFIG["device"] = torch.device("cpu")
    training = ParticleTraining(CONFIG)
    training.excute()
