import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.norm.split_qkv_rmsnorm_mrope import triton_split_qkv_rmsnorm_mrope


def _select_mrope_cos_sin(
    cos_sin: torch.Tensor,
    mrope_section: list[int],
    is_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_dim = cos_sin.shape[-1]
    half_rope_dim = rope_dim // 2
    offsets = torch.arange(half_rope_dim, device=cos_sin.device)

    if is_interleaved:
        h_mask = (offsets % 3 == 1) & (offsets <= 3 * mrope_section[1])
        w_mask = (offsets % 3 == 2) & (offsets <= 3 * mrope_section[2])
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section[0]
        h_end = t_end + mrope_section[1]
        w_end = h_end + mrope_section[2]
        t_mask = offsets < t_end
        h_mask = (offsets >= t_end) & (offsets < h_end)
        w_mask = (offsets >= h_end) & (offsets < w_end)

    cos = torch.where(
        t_mask,
        cos_sin[0, :, :half_rope_dim],
        torch.where(
            h_mask,
            cos_sin[1, :, :half_rope_dim],
            cos_sin[2, :, :half_rope_dim],
        ),
    )
    sin = torch.where(
        t_mask,
        cos_sin[0, :, half_rope_dim:],
        torch.where(
            h_mask,
            cos_sin[1, :, half_rope_dim:],
            cos_sin[2, :, half_rope_dim:],
        ),
    )
    return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    x = x * weight.float()
    if bias is not None:
        x = x + bias.float()
    return x


def _apply_mrope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    rotary = x[..., :rope_dim]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    rotary = rotary * cos[:, None, :] + rotated * sin[:, None, :]
    return torch.cat((rotary, x[..., rope_dim:]), dim=-1)


def _golden(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int,
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    has_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    if has_gate:
        q_gate, k, v = qkv.split((q_size * 2, kv_size, kv_size), dim=-1)
        q, gate = q_gate.view(num_tokens, num_q_heads, head_size * 2).chunk(2, dim=-1)
        gate = gate.reshape(num_tokens, q_size)
    else:
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(num_tokens, num_q_heads, head_size)
        gate = qkv.new_empty((num_tokens, 0))

    k = k.view(num_tokens, num_kv_heads, head_size)
    q = _rms_norm(q, q_weight, eps, q_bias)
    k = _rms_norm(k, k_weight, eps, k_bias)
    cos, sin = _select_mrope_cos_sin(cos_sin, mrope_section, is_interleaved)
    q = _apply_mrope(q, cos.float(), sin.float(), rope_dim)
    k = _apply_mrope(k, cos.float(), sin.float(), rope_dim)
    return q.flatten(1), k.flatten(1), v, gate


@pytest.mark.parametrize(
    "num_tokens,rope_dim,mrope_section,is_interleaved,has_gate,has_bias",
    [
        pytest.param(
            7, 256, [32, 48, 48], True, True, False, id="full_interleaved_gate"
        ),
        pytest.param(
            65, 128, [48, 40, 40], False, False, True, id="partial_contiguous_bias"
        ),
    ],
)
def test_split_qkv_rmsnorm_mrope(
    num_tokens: int,
    rope_dim: int,
    mrope_section: list[int],
    is_interleaved: bool,
    has_gate: bool,
    has_bias: bool,
):
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")

    torch.manual_seed(0)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    dtype = torch.bfloat16
    num_q_heads = 2
    num_kv_heads = 1
    head_size = 256
    eps = 1e-6
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0

    qkv = torch.randn(
        num_tokens,
        q_size + gate_size + 2 * kv_size,
        device=device,
        dtype=dtype,
    )
    q_weight = torch.randn(head_size, device=device, dtype=dtype)
    k_weight = torch.randn(head_size, device=device, dtype=dtype)
    q_bias = torch.randn(head_size, device=device, dtype=dtype) if has_bias else None
    k_bias = torch.randn(head_size, device=device, dtype=dtype) if has_bias else None
    cos_sin = torch.randn(3, num_tokens, rope_dim, device=device, dtype=dtype)

    expected = _golden(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        q_bias,
        k_bias,
        has_gate,
    )
    actual = triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
        q_bias=q_bias,
        k_bias=k_bias,
        has_gate=has_gate,
    )

    for fused, golden in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            fused.float().cpu(),
            golden.float().cpu(),
            atol=5e-2,
            rtol=5e-3,
        )
