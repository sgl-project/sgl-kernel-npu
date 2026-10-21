"""PyTorch RMSNorm+bias with optional saturating int8 quantization."""

import torch


def rmsnorm_bias(
    input, norm_weight, norm_bias, eps, quant_scale=None, quant_offset=None
):
    values = input.float()
    variance = values.square().sum(dim=-1, keepdim=True) / input.shape[1]
    normalized = values * (1 / torch.sqrt(variance + eps))
    output = normalized * norm_weight.float() + norm_bias.float()
    # Match the Triton branch condition: offset alone does not enable quantization.
    if quant_scale is not None:
        output = output * quant_scale.float() + quant_offset.float()
        # Float-to-int conversion truncates toward zero; clamp before casting
        # to implement overflow_mode="saturate" instead of integer wraparound.
        return output.trunc().clamp(-128, 127).to(torch.int8)
    return output.to(input.dtype)
