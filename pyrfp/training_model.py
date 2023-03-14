#!/usr/bin/env python3
import torch
from torch import Tensor

from pyrfp.model_tools import set_act_funcs


class NLinearLayerModel(torch.nn.Module):
    """Class for neural network (nn) model which consisted of N number of
    hidden layers.
    Therefore, total number of layer is, N + 2 including input layer and
    prediction layer.

    Note:
        I implemented batch normalization here, but no need to use
        (actually not working in our case) since our prediction process
        only requires 1D data.
    """

    def __init__(
        self,
        d_output: int = 12,
        d_input: int = 8,
        d_hidden: list[int] = [128, 128],
        act_func: list[str] = ["tanh", "tanh"],
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        """Construct NLinearLayerModel.

        Args:
            torch (object): base of class
            d_output (int): data output size
            d_input (int): data input size
            d_hidden (list): number of hidden layers
                e.g.) [30, 40, 50, 20]
            act_func (list, optional): activation function.
                Defaults to relu.
                e.g.) ['relu', 'elu', 'sigmoid', 'relu', 'tanh']
            batch_norm (bool, optional): batch normalization.
            dropout_rate (float, optional): dropout rate.
        """
        super().__init__()

        self.n_hidden = len(d_hidden)

        self.act_func = act_func
        self.batch_norm = batch_norm
        self.hidden = []
        self.hidden_bn = []
        self.act_funcs = set_act_funcs(act_func)
        self.dropout_rate = dropout_rate

        self.layers = [d_input] + d_hidden

        # construct hidden layers
        for i_h in range(self.n_hidden):
            self.hidden.append(
                torch.nn.Linear(self.layers[i_h], self.layers[i_h + 1], dtype=dtype)
            )

        # convert to ModuleList. python list is not working here.
        self.hidden = torch.nn.ModuleList(self.hidden)

        if self.batch_norm:
            # including input batch normalization
            for i_bn in range(self.n_hidden + 1):
                self.hidden_bn.append(torch.nn.BatchNorm1d(self.layers[i_bn]))
            self.hidden_bn = torch.nn.ModuleList(self.hidden_bn)

        # prediction layer
        self.predict = torch.nn.Linear(d_hidden[-1], d_output, dtype=dtype)

        if self.dropout_rate > 0:
            # dropout
            self.dropout = torch.nn.Dropout(p=self.dropout_rate)

    def forward(self, x: Tensor):
        """Feedforward process"""

        if self.batch_norm:
            x = self.hidden_bn[0](x)

        for i_h in range(self.n_hidden):
            # activation function
            if self.batch_norm:
                x = self.act_funcs[i_h](self.hidden_bn[i_h + 1](self.hidden[i_h](x)))
            else:
                x = self.act_funcs[i_h](self.hidden[i_h](x))

            if self.dropout_rate > 0:
                x = self.dropout(x)

        if self.n_hidden == len(self.act_func):
            x = self.predict(x)
        elif self.n_hidden < len(self.act_func):
            # apply custom activation function for the prediction layer
            x = self.act_funcs[-1](self.predict(x))

        return x

    def extra_repr(self) -> str:
        """Custom repr"""
        msg = "(act_funcs): (\n"
        for i, func in enumerate(self.act_func):
            msg += f"  ({i}): {func}"
            if i + 1 > self.n_hidden:
                msg += " <- custom activation function for the (predict) layer"
            msg += "\n"
        msg += ")"

        return msg
