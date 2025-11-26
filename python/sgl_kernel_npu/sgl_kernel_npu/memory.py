import torch


def weak_ref_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return torch.ops.npu.weak_ref_tensor(tensor)
    return tensor
