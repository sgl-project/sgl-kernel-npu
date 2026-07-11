import torch
from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
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


def forward_native(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin,
    head_size,
    rotary_dim,
    is_neox_style,
):
    num_tokens = query.shape[0]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_shape = query.shape

    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb(key_rot, cos, sin, is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def test_fused_rope_qk_mqa():
    device = "npu"
    test_cases = [
        # token, q_heads, kv_heads, head_size, rotary_dim
        (32, 8, 1, 64, 32),
        (32, 8, 1, 32, 32),
        (17, 16, 1, 128, 64),
        (64, 32, 1, 128, 64),
    ]

    for is_neox_style in [
        True,
        False,
    ]:

        for (
            num_tokens,
            q_heads,
            kv_heads,
            head_size,
            rotary_dim,
        ) in test_cases:

            print(
                f"test "
                f"neox={is_neox_style}, "
                f"tokens={num_tokens}, "
                f"heads={q_heads}, "
                f"head={head_size}, "
                f"rotary={rotary_dim}"
            )

            query = torch.randn(
                num_tokens,
                q_heads,
                head_size,
                device=device,
                dtype=torch.float16,
            )

            key = torch.randn(
                num_tokens,
                kv_heads,
                head_size,
                device=device,
                dtype=torch.float16,
            )

            cos = torch.randn(
                num_tokens,
                rotary_dim // 2,
                device=device,
                dtype=torch.float16,
            )

            sin = torch.randn(
                num_tokens,
                rotary_dim // 2,
                device=device,
                dtype=torch.float16,
            )

            cos_sin = torch.cat(
                [
                    cos,
                    sin,
                ],
                dim=-1,
            )

            # reference
            q_ref, k_ref = forward_native(
                query.clone(),
                key.clone(),
                cos_sin,
                head_size,
                rotary_dim,
                is_neox_style,
            )

            # kernel
            q_out, k_out = fused_rope_qk_mqa(
                query.clone(),
                key.clone(),
                cos_sin,
                rotary_dim,
                is_neox_style,
            )

            torch.testing.assert_close(
                q_out,
                q_ref,
            )

            torch.testing.assert_close(
                k_out,
                k_ref,
            )


if __name__ == "__main__":
    test_fused_rope_qk_mqa()
