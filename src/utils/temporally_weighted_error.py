import torch
import torch.nn.functional as F


def twe_MSE(outs, y, criterion, p=2, time_dim=1, overflow_cap=1e7):
    assert time_dim == 1

    shape_len = len(outs.shape)
    shape = tuple(1 if i != 1 else outs.shape[1] for i in range(shape_len))

    weight = torch.ones(shape, device=outs.device)
    for i in range(shape[1]):
        try:
            weight[:, i] = ((i + 1) / (shape[1] + 1)) ** p
        except OverflowError:
            print("Power operation resulted in overflow.")
            weight[:, i] = overflow_cap
    weight /= weight.sum()
    weight *= shape[1]

    return criterion(outs * weight, y * weight)


def twe_CE(outs, y, criterion, p=2, time_dim=1, overflow_cap=1e7):
    assert time_dim == 1

    shape_len = len(outs.shape)
    shape = tuple(1 if i != 1 else outs.shape[1] for i in range(shape_len))

    weight = torch.ones(shape, device=outs.device)
    for i in range(shape[1]):
        try:
            weight[:, i] = ((i + 1) / (shape[1] + 1)) ** p
        except OverflowError:
            print("Power operation resulted in overflow.")
            weight[:, i] = overflow_cap
    weight /= weight.sum()
    weight *= shape[1]

    return criterion(outs * weight, y)
