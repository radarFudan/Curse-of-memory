import torch


def activation_name_to_function(activation):
    if activation == "linear":
        return torch.nn.Identity()
    elif activation == "tanh":
        return torch.tanh
    elif activation == "hardtanh":
        return torch.nn.functional.hardtanh
    elif activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == "relu":
        return torch.relu
    elif activation == "sigmoid":
        torch.sigmoid
