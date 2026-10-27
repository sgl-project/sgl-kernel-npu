import torch


def l1_norm(input):
    output = input / input.sum(dim=-1, keepdim=True)
    return output.to(torch.float32)
