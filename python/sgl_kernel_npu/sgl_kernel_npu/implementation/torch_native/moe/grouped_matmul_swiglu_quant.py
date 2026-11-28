"""Independent eager reference for grouped matmul, SwiGLU and quantization."""

import torch


def grouped_matmul_swiglu_quant(
    x, weight, weight_scale, x_scale, group_list, swiglu_limit, weight_assist_matrix
):
    # Eager reference: host expert boundaries are intentionally not graph-safe.
    import torch_npu

    if weight_assist_matrix is None:
        weight = torch_npu.npu_format_cast(weight, 2).float()
    else:
        packed = torch_npu.npu_format_cast(weight.view(torch.int8), 2)
        nibbles = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).flatten(-2)
        weight = (nibbles ^ 8).float() - 8
        experts, k, n = weight.shape
        scales = (
            weight_scale.to(torch.int32).view(torch.float32).reshape(experts, -1, n)
        )
        weight = (
            weight.reshape(experts, scales.shape[1], -1, n) * scales.unsqueeze(-2)
        ).reshape(experts, k, n)

    outputs = []
    begin = 0
    for expert, end in enumerate(group_list.cpu().tolist()):
        tokens = x[begin:end].float()
        if weight_assist_matrix is None:
            z = (tokens @ weight[expert]) * weight_scale[expert].float()
        else:
            # MSD reconstructs the signed activation with its low nibble shifted by 8.
            z = (tokens - 8) @ weight[expert] + weight_assist_matrix[expert]
        z = z * x_scale[begin:end, None]
        gate, up = z.chunk(2, dim=-1)
        if swiglu_limit > 0:
            gate, up = gate.clamp(max=swiglu_limit), up.clamp(
                -swiglu_limit, swiglu_limit
            )
        value = torch.nn.functional.silu(gate) * up
        scale = value.abs().amax(-1) / 127
        quant = torch.floor(value / scale.clamp_min(1e-30)[:, None] + 0.5)
        outputs.append((quant.clamp(-128, 127).to(torch.int8), scale))
        begin = end
    quant = torch.zeros(
        (x.shape[0], weight.shape[-1] // 2), dtype=torch.int8, device=x.device
    )
    scale = torch.zeros(x.shape[0], device=x.device)
    if begin:
        quant[:begin], scale[:begin] = torch.cat([v[0] for v in outputs]), torch.cat(
            [v[1] for v in outputs]
        )
    return quant, scale
