import torch
import torch.nn as nn
import torch.nn.functional as F
from activation import activation_name_to_function
from parameterization import CustomLinearLayer

# from src.models.recurrent.activation import activation_name_to_function
# from src.models.recurrent.parameterization import CustomLinearLayer


class SimpleDiagonalRNN(nn.Module):
    def __init__(
        self,
        rec1_size: int = 128,
        activation: str = "linear",
        parameterization: str = "linear",
        diagonal: bool = True,
        return_sequences: bool = True,
        training: bool = True,
    ):
        super().__init__()

        # Kept for standalone purpose
        # if activation == "linear":
        #     self.activation = torch.nn.Identity()
        self.activation = activation_name_to_function(activation)

        # Parameterization of recurrent weight matrix.
        self.W = CustomLinearLayer(
            rec1_size,
            parameterization=parameterization,
            diagonal=diagonal,
            dtype=torch.float64,
            training=training,
        )

        self.rec1_size = rec1_size
        self.parameterization = parameterization
        self.activation_name = activation
        self.diagonal = diagonal
        self.return_sequences = return_sequences

    def forward(self, x):
        batch_size, input_length, rec1_size = x.size()
        assert rec1_size == self.rec1_size

        hidden = []
        hidden.append(torch.zeros(batch_size, 1, self.rec1_size, dtype=x.dtype, device=x.device))

        # SSMs
        for i in range(input_length):
            h_next = self.W(hidden[i]) + x[:, i : i + 1, :]
            hidden.append(h_next)
        hidden = torch.cat(hidden[1:], dim=1)

        # Nonlinearity
        hidden = self.activation(hidden)

        # returned sequence or not
        if self.return_sequences:
            return hidden
        else:
            return hidden[:, -1, :]

    def unify_weight_initialization(self):
        """TODO, when compare weight initialization, we need to unify the initialization of the
        weight matrix."""
        self.W.unify_weight_initialization()

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
    train_linear_model = CustomLinearLayer(
        d, parameterization="linear", diagonal=True, dtype=torch.float64, training=True
    )
    test_linear_model = CustomLinearLayer(
        d, parameterization="linear", diagonal=True, dtype=torch.float64, training=False
    )

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
