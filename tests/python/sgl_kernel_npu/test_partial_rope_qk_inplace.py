import torch
from sgl_kernel_npu.norm.partial_rope_qk_inplace import partial_rope_qk_inplace


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style=False,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def rope_native(query, key, cos_sin, rotary_dim, is_neox_style=False):
    head_dim = query.shape[-1]
    cos, sin = cos_sin.chunk(2, dim=-1)
    q_pe, q_nope = torch.split(query, [rotary_dim, head_dim - rotary_dim], dim=-1)
    k_pe, k_nope = torch.split(key, [rotary_dim, head_dim - rotary_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, cos, sin, is_neox_style=is_neox_style)
    k_pe = apply_rotary_emb(k_pe, cos, sin, is_neox_style=is_neox_style)
    q = torch.cat((q_pe, q_nope), dim=-1)
    k = torch.cat((k_pe, k_nope), dim=-1)
    return q, k


def test_partial_rope_qk_inplace():
    dtype = torch.float32
    shapes = [
        [64, 4, 1, 256, 64],  # partial, HQ_IN_GRID
        [64, 4, 1, 64, 64],   # no partial
        [1, 4, 1, 256, 64],   # HK_IN_GRID
        [1, 4, 1, 64, 64], 
    ]
    for T, Hq, Hk, D, D_ROPE in shapes:
        for is_neox_style in [True, False]:
            query = torch.randn((T, Hq, D), dtype=dtype, device="npu")
            key = torch.randn((T, Hk, D), dtype=dtype, device="npu")
            cos_sin = torch.randn((T, D_ROPE), dtype=dtype, device="npu")
            _query = query.clone()
            _key = key.clone()
            # triton
            res_q, res_k = partial_rope_qk_inplace(query, key, cos_sin, rotary_dim=D_ROPE, is_neox_style=is_neox_style)
            # native
            ans_q, ans_k = rope_native(_query, _key, cos_sin, rotary_dim=D_ROPE, is_neox_style=is_neox_style)
            assert torch.allclose(res_q, ans_q)
            assert torch.allclose(res_k, ans_k)


if __name__ == "__main__":
    test_partial_rope_qk_inplace()
