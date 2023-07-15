import scipy
import torch
import torch.nn.utils.parametrize as P
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_size: int = 128,
        out_size: int = 1,
        activation: str = "linear",
        bias: bool = True,
    ):
        super().__init__()

        self.U = nn.Linear(in_size, out_size, bias=bias, dtype=torch.float64)
        self.in_size = in_size
        self.out_size = out_size

        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh
        elif activation == "softmax":
            self.activation = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        in_size = x.size()[-1]
        assert in_size == self.in_size

        # print("Linear", x.device, self.U.weight.device)
        x = self.U(x)
        x = self.activation(x)

        return x
