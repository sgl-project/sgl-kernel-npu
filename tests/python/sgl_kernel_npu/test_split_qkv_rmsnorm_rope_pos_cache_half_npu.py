import unittest

import torch
from sglang.srt.utils import is_npu

split_qkv_rmsnorm_rope_pos_cache_half_npu = None
if is_npu():
    try:
        from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
            split_qkv_rmsnorm_rope_pos_cache_half_npu,
        )
    except (ImportError, OSError):
        pass


def _make_cos_sin_cache(
    max_position_embeddings: int,
    rotary_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Match RotaryEmbedding._compute_cos_sin_cache (base.py)."""
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


def golden_rms_norm_forward_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match RMSNorm.forward_native (layernorm.py): fp32 variance, then weight."""
    if not x.is_contiguous():
        x = x.contiguous()
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.to(torch.float32)).to(orig_dtype)
    if bias is not None:
        x = x + bias.to(orig_dtype)
    return x


def golden_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Match apply_rotary_emb (rotary_embedding/utils.py)."""
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
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def golden_split_qkv_rmsnorm_rope_pos_cache_half(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_size: int,
    kv_size: int,
    head_dim: int,
    rope_dim: int,
    eps: float | None,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    use_qk_norm: bool = True,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch path aligned with llada2 else-branch:
    split -> apply_qk_norm (RMSNorm per head) -> RoPE forward_native.
    """
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    bsz = qkv.shape[0]

    if use_qk_norm:
        assert eps is not None and q_weight is not None and k_weight is not None
        q = q.reshape(-1, head_dim)
        k = k.reshape(-1, head_dim)
        q = golden_rms_norm_forward_native(q, q_weight, eps, q_bias)
        k = golden_rms_norm_forward_native(k, k_weight, eps, k_bias)
        q = q.reshape(bsz, q_size)
        k = k.reshape(bsz, kv_size)

    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    q_shape = q.shape
    k_shape = k.shape
    q = q.view(num_tokens, -1, head_dim)
    k = k.view(num_tokens, -1, head_dim)

    q_rot = q[..., :rope_dim]
    q_pass = q[..., rope_dim:]
    q_rot = golden_apply_rotary_emb(q_rot, cos, sin, is_neox_style)
    q = torch.cat((q_rot, q_pass), dim=-1).reshape(q_shape)

    k_rot = k[..., :rope_dim]
    k_pass = k[..., rope_dim:]
    k_rot = golden_apply_rotary_emb(k_rot, cos, sin, is_neox_style)
    k = torch.cat((k_rot, k_pass), dim=-1).reshape(k_shape)

    return q, k, v


def _assert_close_fp32(
    a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float = 5e-3
):
    torch.testing.assert_close(
        a.to(torch.float32).cpu(),
        b.to(torch.float32).cpu(),
        atol=atol,
        rtol=rtol,
    )


@unittest.skipIf(not is_npu(), "NPU not available")
@unittest.skipIf(
    split_qkv_rmsnorm_rope_pos_cache_half_npu is None,
    "sgl_kernel_npu.split_qkv_rmsnorm_rope_pos_cache_half_npu not installed",
)
class TestSplitQkvRmsnormRopePosCacheHalfNpu(unittest.TestCase):
    """Eager vs golden, then NPU graph capture/replay vs golden (see test_npu_fia_chunk_prefill_pa_and_graph)."""

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("torch.npu not available")
        self.device = torch.device("npu:0")
        self.dtype = torch.bfloat16

    def _run_case_graph(
        self,
        bsz: int,
        q_hidden_size: int,
        kv_hidden_size: int,
        head_dim: int,
        rope_dim: int,
        use_qk_norm: bool,
    ):
        eps = 1e-6
        max_pos = 2048
        rope_theta = 10000.0
        torch.manual_seed(0)

        qkv = torch.randn(
            bsz,
            q_hidden_size + kv_hidden_size * 2,
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.randint(
            0, max_pos, (bsz,), dtype=torch.int64, device=self.device
        )
        cos_sin_cache = _make_cos_sin_cache(
            max_pos, rope_dim, rope_theta, torch.float32, self.device
        )

        q_weight = torch.randn(head_dim, dtype=self.dtype, device=self.device)
        k_weight = torch.randn(head_dim, dtype=self.dtype, device=self.device)
        q_bias = k_bias = None

        gq, gk, gv = golden_split_qkv_rmsnorm_rope_pos_cache_half(
            qkv,
            positions,
            cos_sin_cache,
            q_hidden_size,
            kv_hidden_size,
            head_dim,
            rope_dim,
            eps if use_qk_norm else None,
            q_weight if use_qk_norm else None,
            k_weight if use_qk_norm else None,
            q_bias,
            k_bias,
            use_qk_norm=use_qk_norm,
        )

        # Eager fused kernel vs golden
        fq_e, fk_e, fv_e = split_qkv_rmsnorm_rope_pos_cache_half_npu(
            qkv,
            positions,
            cos_sin_cache,
            q_hidden_size,
            kv_hidden_size,
            head_dim,
            eps=eps if use_qk_norm else None,
            q_weight=q_weight if use_qk_norm else None,
            k_weight=k_weight if use_qk_norm else None,
            q_bias=q_bias,
            k_bias=k_bias,
            rope_dim=rope_dim,
        )
        _assert_close_fp32(fq_e, gq, atol=5e-2)
        _assert_close_fp32(fk_e, gk, atol=5e-2)
        _assert_close_fp32(fv_e, gv, atol=5e-2, rtol=5e-3)

        # Graph: staging inputs + fixed output buffers (same addresses across replay)
        qkv_staging = torch.empty_like(qkv)
        positions_staging = torch.empty_like(positions)
        out_q = torch.empty(bsz, q_hidden_size, dtype=self.dtype, device=self.device)
        out_k = torch.empty(bsz, kv_hidden_size, dtype=self.dtype, device=self.device)
        out_v = torch.empty(bsz, kv_hidden_size, dtype=self.dtype, device=self.device)

        def run_once():
            fq, fk, fv = split_qkv_rmsnorm_rope_pos_cache_half_npu(
                qkv_staging,
                positions_staging,
                cos_sin_cache,
                q_hidden_size,
                kv_hidden_size,
                head_dim,
                eps=eps if use_qk_norm else None,
                q_weight=q_weight if use_qk_norm else None,
                k_weight=k_weight if use_qk_norm else None,
                q_bias=q_bias,
                k_bias=k_bias,
                rope_dim=rope_dim,
            )
            out_q.copy_(fq)
            out_k.copy_(fk)
            out_v.copy_(fv)
            return out_q

        qkv_staging.copy_(qkv)
        positions_staging.copy_(positions)
        graph = torch.npu.NPUGraph()
        torch.npu.synchronize()
        capture_stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
            run_once()
        torch.npu.synchronize()

        qkv_staging.copy_(qkv)
        positions_staging.copy_(positions)
        graph.replay()
        torch.npu.synchronize()

        _assert_close_fp32(out_q, gq, atol=5e-2)
        _assert_close_fp32(out_k, gk, atol=5e-2)
        _assert_close_fp32(out_v, gv, atol=5e-2, rtol=5e-3)

        # Replay vs eager (bf16 path: looser than golden)
        _assert_close_fp32(out_q, fq_e, atol=2e-2)
        _assert_close_fp32(out_k, fk_e, atol=2e-2)
        _assert_close_fp32(out_v, fv_e, atol=2e-2, rtol=2e-2)

    def test_with_qk_norm_full_rope(self):
        """rope_dim == head_dim; fused op in NPU graph."""
        self._run_case_graph(
            bsz=12,
            q_hidden_size=6144,
            kv_hidden_size=1024,
            head_dim=128,
            rope_dim=128,
            use_qk_norm=True,
        )

    def test_with_qk_norm_partial_rope(self):
        """rope_dim < head_dim."""
        self._run_case_graph(
            bsz=8,
            q_hidden_size=4096,
            kv_hidden_size=1024,
            head_dim=128,
            rope_dim=64,
            use_qk_norm=True,
        )

    def test_no_qk_norm_full_rope(self):
        self._run_case_graph(
            bsz=12,
            q_hidden_size=6144,
            kv_hidden_size=1024,
            head_dim=128,
            rope_dim=128,
            use_qk_norm=False,
        )


if __name__ == "__main__":
    unittest.main()
