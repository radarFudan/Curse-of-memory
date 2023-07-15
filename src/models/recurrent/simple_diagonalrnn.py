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

        if self.training:
            self.activation = F.relu
        else:
            self.activation = torch.nn.Identity()

    def forward(self, x):
        # Create a diagonal matrix
        weight = self.activation(self.weight)

        return x * weight

    def stability_margin(self):
        """Return the stability margin of the weight matrix.

        Positive means stable, the larger the better. Negative means unstable.
        """

        if self.training:
            # Mask for positive weights
            positive_weights_mask = self.weight > 0

            # Evaluate the smallest positive value in the weight
            smallest_positive = self.weight[positive_weights_mask].min()
        else:
            smallest_positive = self.weight.min()

        return smallest_positive

    @torch.no_grad()
    def perturb_weight_initialization(self):
        self.weight.data = F.relu(self.weight.data)


class SimpleDiagonalRNN(nn.Module):
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
            h_next = hidden[i] + self.dt * self.activation(x[:, i : i + 1, :] - self.W(hidden[i]))
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
        self.W.perturb_weight_initialization()


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

    # SimpleDiagonalRNN
    train_model = SimpleDiagonalRNN(d, training=True)
    test_model = SimpleDiagonalRNN(d, training=False)

    # Shape check
    inputs = torch.randn(1, 100, d, dtype=torch.float64)
    outputs = test_model(inputs)
    assert outputs.shape == (1, 100, d)

    # Stability margin check
    print("Stability margin", train_model.stability_margin(), "expected > 0")
    print("Stability margin", test_model.stability_margin())

    print("Test passed.")
