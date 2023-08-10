import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LieRNN(nn.Module):
    def __init__(
        self,
        rec1_size: int = 256,
        hidden_dim: int = None,
        return_sequences: bool = True,
        activation: str = "relu",
        training: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(np.ceil((rec1_size) ** 0.5))

        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh
        elif activation == "relu":
            self.activation = torch.relu

        self.encode_weight = nn.Parameter(
            torch.randn((rec1_size, hidden_dim, hidden_dim), dtype=torch.float64)
            / np.sqrt(hidden_dim * rec1_size)
        )
        self.decode_weight = nn.Parameter(
            torch.randn((rec1_size, hidden_dim, hidden_dim), dtype=torch.float64)
            / np.sqrt(hidden_dim * rec1_size)
        )

        self.rec1_size = rec1_size
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences

    def forward(self, x):
        batch_size, input_length, rec1_size = x.size()
        assert rec1_size == self.rec1_size

        # Maybe we need to bound the matrix's frobenius norm
        # https://math.stackexchange.com/questions/1393301/frobenius-norm-of-product-of-matrix

        x = torch.einsum("bti,ijk->btjk", x, self.encode_weight)
        x = self.activation(x)

        hiddens = []
        hidden_state = torch.eye(self.hidden_dim, self.hidden_dim, dtype=x.dtype, device=x.device)

        # Residual RNN
        for i in range(input_length):
            hidden_state = torch.matmul(hidden_state, x[:, i : i + 1, :, :])
            hiddens.append(hidden_state)
        hiddens = torch.cat(hiddens, dim=1)

        outputs = torch.einsum("btjk,ojk->bto", hiddens, self.decode_weight)
        outputs = self.activation(outputs)

        # returned sequence or not
        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]

    def stability_margin(self):
        raise NotImplementedError

    @torch.no_grad()
    def perturb_weight_initialization(self):
        raise NotImplementedError


if __name__ == "__main__":
    d = 12

    # SimpleDiagonalRNN
    train_model = LieRNN(d, training=True)
    test_model = LieRNN(d, training=False)

    # Shape check
    inputs = torch.randn(1, 100, d * d, dtype=torch.float64)
    outputs = test_model(inputs)
    print(outputs.shape)
    assert outputs.shape == (1, 100, d * d)

    print("Test passed.")
