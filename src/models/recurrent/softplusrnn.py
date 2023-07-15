import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinearLayer(nn.Module):
    def __init__(
        self,
        size,
        dtype,
        training: bool = True,
    ):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.randn(size, dtype=dtype))
        self.training = training

        # if self.training:
        #     self.activation = F.softplus
        # else:
        #     self.activation = torch.nn.Identity()

        self.activation = F.softplus

    def forward(self, x):
        # Create a diagonal matrix
        weight = self.activation(self.weight)

        return x * weight

    def stability_margin(self):
        """Return the stability margin of the weight matrix.

        Positive means stable, the larger the better. Negative means unstable.
        """

        smallest_positive = F.softplus(self.weight.min())

        return smallest_positive

    @torch.no_grad()
    def perturb_weight_initialization(self):
        self.weight.data = F.softplus(self.weight.data)


class CustomOrthogonalLayer(nn.Module):
    def __init__(
        self,
        size,
        dtype,
    ):
        super().__init__()
        self.size = size

        matrix = torch.randn((size, size), dtype=dtype)
        q, _ = torch.linalg.qr(matrix)
        self.weight = nn.Parameter(q)  # Now orthogonal

    def forward(self, x, transpose=False):
        if transpose:
            return F.linear(x, self.weight.t())
        else:
            return F.linear(x, self.weight)


class SoftplusRNN(nn.Module):
    def __init__(
        self,
        rec1_size: int = 128,
        activation: str = "linear",
        dt: float = 1.0,
        return_sequences: bool = True,
        training: bool = True,
    ):
        super().__init__()

        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh

        self.W = CustomLinearLayer(rec1_size, dtype=torch.float64, training=training)
        self.P = CustomOrthogonalLayer(rec1_size, dtype=torch.float64)

        self.rec1_size = rec1_size
        self.dt = dt
        self.return_sequences = return_sequences

    def forward(self, x):
        batch_size, input_length, rec1_size = x.size()
        assert rec1_size == self.rec1_size

        hidden = []
        hidden.append(torch.zeros(1, 1, self.rec1_size, dtype=x.dtype, device=x.device))

        # Residual RNN
        # Expect W1 to be positive definite to ensure stability
        for i in range(input_length):
            h_next = hidden[i] + self.dt * self.activation(
                x[:, i : i + 1, :]
                - self.P(self.W(self.P(hidden[i], transpose=False)), transpose=True)
            )
            hidden.append(h_next)
        hidden = torch.cat(hidden[1:], dim=1)

        # returned sequence or not
        if self.return_sequences:
            return hidden
        else:
            return hidden[:, -1, :]

    def stability_margin(self):
        """Return the stability margin of the model."""
        return self.W.stability_margin()

    @torch.no_grad()
    def perturb_weight_initialization(self):
        """Perturb the weight initialization to make the model unstable."""
        # self.W.perturb_weight_initialization()
        pass


if __name__ == "__main__":
    d = 6

    # CustomLinearLayer
    train_linear_model = CustomLinearLayer(d, dtype=torch.float64, training=True)
    test_linear_model = CustomLinearLayer(d, dtype=torch.float64, training=False)

    # Shape check
    inputs = torch.randn(1, 100, d, dtype=torch.float64)
    outputs = train_linear_model(inputs)
    assert outputs.shape == (1, 100, d)

    # Stability margin check
    print("Stability margin", train_linear_model.stability_margin(), "expected > 0")
    print("Stability margin", test_linear_model.stability_margin())

    train_model = SoftplusRNN(d, training=True)
    test_model = SoftplusRNN(d, training=False)

    # Shape check
    inputs = torch.randn(1, 100, d, dtype=torch.float64)
    outputs = test_model(inputs)
    assert outputs.shape == (1, 100, d)

    # Stability margin check
    print("Stability margin", train_model.stability_margin(), "expected > 0")
    print("Stability margin", test_model.stability_margin())

    print(
        "P's L2 norm",
        torch.linalg.norm(train_model.P.weight, ord=2),
        "expected approximately 1",
    )

    # Stability margin check, inside the model should be the same as the train_model
    # print("Stability margin", train_model.W.stability_margin(), "expected > 0")
    # print("Stability margin", test_model.W.stability_margin())

    print("Test passed.")
