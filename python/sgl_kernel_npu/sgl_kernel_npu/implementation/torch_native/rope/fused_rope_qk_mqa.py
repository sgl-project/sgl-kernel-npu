"""Independent PyTorch Q/K rotary embeddings with an unchanged head tail."""

import torch

from ...triton.utils.rope.rope_utils import validate_rope_inputs


def fused_rope_qk_mqa(query, key, cos_sin, rotary_dim, is_neox_style):
    """Rotate Q and K using preselected cos/sin rows; never modify the inputs."""
    validate_rope_inputs(query, key, cos_sin, rotary_dim, is_neox_style)
    # Triton prefers float16 for mixed float16/bfloat16 operands, whereas
    # torch.promote_types would choose float32 for that particular pair.
    if {query.dtype, cos_sin.dtype} == {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.promote_types(query.dtype, cos_sin.dtype)
    cos, sin = cos_sin[: query.shape[0]].chunk(2, dim=-1)
    cos = cos.to(compute_dtype).unsqueeze(1)
    sin = sin.to(compute_dtype).unsqueeze(1)

    def rotate(tensor):
        rotary = tensor[..., :rotary_dim].to(compute_dtype)
        if is_neox_style:
            first, second = rotary.chunk(2, dim=-1)
        else:
            first, second = rotary[..., ::2], rotary[..., 1::2]

        out_first = first * cos - second * sin
        out_second = first * sin + second * cos
        if is_neox_style:
            rotated = torch.cat((out_first, out_second), dim=-1)
        else:
            rotated = torch.stack((out_first, out_second), dim=-1).flatten(-2)
        return torch.cat((rotated.to(tensor.dtype), tensor[..., rotary_dim:]), dim=-1)

    return rotate(query), rotate(key)
