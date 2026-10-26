"""Public RMSNorm+bias input metadata checks and flat-vector preparation."""


def prepare_inputs(input, norm_weight, norm_bias, quant_scale, quant_offset):
    if input.ndim != 2 or input.shape[1] <= 0:
        raise ValueError("input must have shape [rows, positive hidden]")
    hidden = input.shape[1]
    vectors = [norm_weight, norm_bias]
    if quant_scale is not None:
        if quant_offset is None:
            raise ValueError("quant_offset is required when quant_scale is provided")
        vectors.extend((quant_scale, quant_offset))
    for value in vectors:
        if value is None or value.numel() != hidden or value.device != input.device:
            raise ValueError(
                "affine/quantization vectors must contain hidden elements on the input device"
            )
    norm_weight = _flat_vector(norm_weight)
    norm_bias = _flat_vector(norm_bias)
    if quant_scale is not None:
        quant_scale = _flat_vector(quant_scale)
        quant_offset = _flat_vector(quant_offset)
    return input.contiguous(), norm_weight, norm_bias, quant_scale, quant_offset


def _flat_vector(value):
    return value.contiguous() if value.ndim == 1 else value.reshape(-1).contiguous()
