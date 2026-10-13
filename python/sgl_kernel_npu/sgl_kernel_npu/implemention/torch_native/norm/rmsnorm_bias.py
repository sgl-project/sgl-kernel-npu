"""PyTorch reference implementation of the non-quantized operation."""

import torch


def rmsnorm_bias(
    input, norm_weight, norm_bias, eps, quant_scale=None, quant_offset=None
):
    if quant_scale is not None or quant_offset is not None:
        raise NotImplementedError(
            "torch_native rmsnorm_bias does not support quantization yet; "
            "use implementation='triton'."
        )

    values = input.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * (1 / torch.sqrt(variance + eps))
    return (normalized * norm_weight.float() + norm_bias.float()).to(input.dtype)
