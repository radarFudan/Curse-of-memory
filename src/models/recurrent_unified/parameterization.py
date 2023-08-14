import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinearLayer(nn.Module):
    def __init__(
        self,
        size,
        parameterization: str = "linear",
        diagonal: bool = True,
        dtype=torch.float64,
        training: bool = True,
        epsilon: float = 1e-7,
    ):
        super().__init__()

        # Uniform initialization in [-1, 1]
        self.weight = nn.Parameter(2 * torch.rand(size, dtype=dtype) - 1)

        if diagonal:
            pass
        else:
            matrix = torch.randn((size, size), dtype=dtype)
            q, _ = torch.linalg.qr(matrix)
            self.Q = nn.Parameter(q)  # Now orthogonal

        self.size = size
        self.parameterization = parameterization
        self.diagonal = diagonal
        self.training = training
        self.epsilon = epsilon

    def forward(self, x):
        W = self.parameterize()

        if self.diagonal:
            return x * W
        else:
            W_matrix = self.Q.T @ torch.diag(W) @ self.Q
            return x @ W_matrix

    def parameterize(self):
        epsilon = 1e-7

        if self.parameterization == "linear":
            W = self.weight
            if self.training:
                W = torch.clamp(
                    self.weight, -1 + self.epsilon, 1 - self.epsilon
                )  # Stability require [-1, 1]
        elif self.parameterization == "lru":
            W = torch.exp(-torch.exp(self.weight))
        elif self.parameterization == "sigmoid":
            W = 1 - torch.exp(-self.weight) / (1 + torch.exp(-self.weight))
        elif self.parameterization == "inverse":
            W = 1 - 1 / self.weight  # Stability require (0.5, inf]
            if self.training:
                W = 1 - 1 / torch.clamp(self.weight, 0.5 + self.epsilon, float("inf"))
        else:
            raise ValueError("Unknown parameterization")

        return W

    def unify_weight_initialization(self):
        """TODO, when compare weight initialization, we need to unify the initialization of the
        weight matrix."""
        pass

    def stability_margin(self):
        W = self.parameterize()

        # Now it's assumed the recurrent weight matrix is Hurwitz
        stability_margin_value = min(1 - W.max(), W.min() - (-1))

        return stability_margin_value

    @torch.no_grad()
    def perturb_weight_initialization(self):
        if self.parameterization == "linear":
            print("Perturbing weight initialization, project the weights to [-1, 1].")
            self.weight.data = torch.clamp(self.weight.data, -1 + self.epsilon, 1 - self.epsilon)
        elif self.parameterization == "inverse":
            print("Perturbing weight initialization, project the weights to [0.5, inf].")
            self.weight.data = torch.clamp(self.weight.data, 0.5 + self.epsilon, float("inf"))
        else:
            print("Stable parameterization, no need to perturb weight initialization")
