"""Independent PyTorch arithmetic reference for the public AddExample contract."""


def add_example(x1, x2):
    import torch

    return torch.add(x1, x2)
