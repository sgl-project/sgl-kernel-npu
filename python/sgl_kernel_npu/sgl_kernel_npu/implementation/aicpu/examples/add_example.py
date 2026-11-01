"""Dispatch the packaged CpuKernel sample; never fall back to another backend."""


def add_example(x1, x2):
    import torch

    return torch.ops.npu.sgl_add_example(x1, x2)
