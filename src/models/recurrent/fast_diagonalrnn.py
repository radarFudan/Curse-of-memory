import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper, reinterpret


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
            h_next = hidden[i] + self.dt * (x[:, i : i + 1, :] - self.W(hidden[i]))
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


def to_triton(x: np.ndarray, device="cuda", dst_type=None):
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip("u")  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device).contiguous(), getattr(tl, t))
    else:
        if dst_type and "float8" in dst_type:
            return reinterpret(torch.tensor(x, device=device).contiguous(), getattr(tl, dst_type))
        if t == "float32" and dst_type == "bfloat16":
            return torch.tensor(x, device=device).contiguous().bfloat16()
        return torch.tensor(x, device=device).contiguous()


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


@triton.jit
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.

    Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


class Assoc_scan_DiagonalRNN(nn.Module):
    def __init__(
        self,
        rec1_size: int = 128,
        activation: str = "linear",
        dt: float = 1.0,
        return_sequences: bool = True,
        training: bool = True,
    ):
        super().__init__()

        self.W = CustomLinearLayer(rec1_size, dtype=torch.float64, training=training)

        self.rec1_size = rec1_size
        self.dt = dt
        self.return_sequences = return_sequences

    @triton.jit
    def forward(self, x):
        batch_size, input_length, rec1_size = x.size()
        assert rec1_size == self.rec1_size

        hidden = []
        hidden.append(torch.zeros(batch_size, 1, self.rec1_size, dtype=x.dtype, device=x.device))

        state = torch.zeros(batch_size, 1, self.rec1_size, dtype=x.dtype, device=x.device)

        Ws = self.W * torch.ones((input_length, rec1_size))
        _, hidden = tl.associative_scan((Ws, x), 0, binary_operator)

        # Residual RNN
        # Expect W1 to be positive definite to ensure stability
        for i in range(input_length):
            h_next = hidden[i] + self.dt * (x[:, i : i + 1, :] - self.W(hidden[i]))

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

    # Repeat
    repeats = 10

    # Step 1, evaluate current model evaluation speed.
    start_time = time.time_ns()
    # Evaluate model
    for _ in range(repeats):
        outputs = test_model(inputs)
    # Stop the timer
    end_time = time.time_ns()
    # Compute and print elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time for {repeats} evaluations: {elapsed_time:.2f} nano seconds")

    # Step 2, evaluate the compiled model evaluation speed.
    # It would be very hard for the model to get results from compiled case.
    compiled_test_model = torch.compile(
        test_model, mode="reduce-overhead", fullgraph=True, dynamic=False
    )
    start_time = time.time_ns()
    # Evaluate model
    for _ in range(repeats):
        outputs = compiled_test_model(inputs)
    # Stop the timer
    end_time = time.time_ns()
    # Compute and print elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time for {repeats} evaluations (Compiled): {elapsed_time:.2f} nano seconds")

    # Step 3, evaluate the model's speed when the triton is used.
    triton_test_model = Assoc_scan_DiagonalRNN(d, training=False)
    start_time = time.time_ns()
    # Evaluate model
    for _ in range(repeats):
        outputs = triton_test_model(inputs)
    # Stop the timer
    end_time = time.time_ns()
    # Compute and print elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time for {repeats} evaluations (Triton): {elapsed_time:.2f} nano seconds")
