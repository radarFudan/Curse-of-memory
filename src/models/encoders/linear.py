import scipy
import torch
import torch.nn.utils.parametrize as P
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_size: int = 1,
        out_size: int = 128,
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
        # print("in linear encoder", x.shape)
        in_size = x.shape[-1]

        # print("in linear encoder", in_size, self.in_size)
        assert in_size == self.in_size

        # print("In linear.py from encoder")
        # print(x.dtype, self.U.weight.dtype)
        x = self.U(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    test_model = Linear(2, 3)

    inputs = torch.randn(1, 10, 2)
    outputs = test_model(inputs)
    assert outputs.shape == (1, 10, 3)
    print("Test passed.")
