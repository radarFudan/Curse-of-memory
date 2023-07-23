# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
import torch
from torch import nn


class GRU(nn.Module):
    def __init__(
        self,
        rec1_size: int = 128,
        batch_first: bool = True,
        bias: bool = True,
        return_sequences: bool = True,
    ):
        super().__init__()

        self.model = torch.nn.GRU(
            input_size=rec1_size,
            hidden_size=rec1_size,
            batch_first=batch_first,
            bias=bias,
            dtype=torch.float64,
        )
        self.rec1_size = rec1_size
        self.return_sequences = return_sequences

    def forward(self, x):
        rec1_size = x.size()[-1]
        assert rec1_size == self.rec1_size

        y, h = self.model(x)

        if self.return_sequences:
            return y
        else:
            return y[:, -1, :]


if __name__ == "__main__":
    test_model = GRU(10)

    inputs = torch.randn(1, 10, 10)
    outputs = test_model(inputs)
    assert outputs.shape == (1, 10, 10)

    test_model = GRU(10, return_sequences=False)

    inputs = torch.randn(1, 10, 10)
    outputs = test_model(inputs)
    assert outputs.shape == (1, 10)

    print("Test passed.")
