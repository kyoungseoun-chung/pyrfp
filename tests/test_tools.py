#!/usr/bin/env python3
"""Test tools"""
import torch
from pymytools.diagnostics import DataLoader
from torch.testing import assert_close  # type: ignore

from pyrfp.training_tools import HDFTrainignDataHandler


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_training_tools() -> None:
    """Test data is located in `tests/test_data`. Each test file has the data size of 10x8 for input and 10x12 for target."""

    dh = HDFTrainignDataHandler(DEVICE)

    assert dh.is_cuda == torch.cuda.is_available()

    input_0_size = dh.get_data_size("./tests/test_data/input_0.h5", "inputs")
    target_0_size = dh.get_data_size("./tests/test_data/target_0.h5", "targets")

    assert input_0_size == torch.Size([10, 8])
    assert target_0_size == torch.Size([10, 12])

    input_list, target_list = dh.get_file_list("./tests/test_data")

    assert len(input_list) == len(target_list) == 2

    input_list_target = [f"input_{i}.h5" for i in range(2)]
    target_list_target = [f"target_{i}.h5" for i in range(2)]

    for i, t in zip(input_list, target_list):
        i_file = i.rsplit("/", 1)[-1]
        t_file = t.rsplit("/", 1)[-1]

        assert i_file in input_list_target
        assert t_file in target_list_target

    dl = DataLoader(device=DEVICE)
    chunk_size = 5
    chunk = torch.randperm(10, device=DEVICE)[:chunk_size]

    input_t = dl.read_hdf5("./tests/test_data/input_0.h5", "inputs")["inputs"]

    data_loaded = dh.get_data_by_chunk("./tests/test_data/input_0.h5", "inputs", chunk)

    assert_close(data_loaded, input_t[chunk, :])

    input_loaded, target_loaded = dh.get_input_target_from_dir(
        input_list, target_list, -1
    )

    assert input_loaded.shape[0] == 20
    assert target_loaded.shape[0] == 20

    input_loaded, target_loaded = dh.get_input_target_from_dir(
        input_list, target_list, 15
    )

    assert input_loaded.shape[0] == int(15 / 2) * 2
    assert target_loaded.shape[0] == int(15 / 2) * 2
