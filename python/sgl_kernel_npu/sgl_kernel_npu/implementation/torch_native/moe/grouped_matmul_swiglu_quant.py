"""Independent eager reference for grouped matmul, SwiGLU and quantization."""

import torch


def grouped_matmul_swiglu_quant(
    x, weight, weight_scale, x_scale, group_list, swiglu_limit
):
    # Eager reference: host expert boundaries are intentionally not graph-safe.
    import torch_npu

    weight = torch_npu.npu_format_cast(weight, 2).float()
    outputs = []
    begin = 0
    for expert, end in enumerate(group_list.cpu().tolist()):
        z = (x[begin:end].float() @ weight[expert]) * weight_scale[expert].float()
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
