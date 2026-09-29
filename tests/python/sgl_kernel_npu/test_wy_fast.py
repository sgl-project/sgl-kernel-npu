import pytest
import torch
from sgl_kernel_npu.fla.wy_fast import recompute_w_u_fwd_npu

device = "npu"


@torch.inference_mode()
def test_recompute_w_u_fwd_equal_length():
    # Equal-length (cu_seqlens=None) regression test.
    # Covers: i_t initialization in the else branch, batch index from
    # tl.program_id(1), and the [B, H, T] pointer base for the transposed
    # beta / g_cumsum tensors. Distinct per-batch values make any batch
    # index error visible in the outputs.
    # See https://github.com/vllm-project/vllm-ascend/pull/14333
    batch_size, seq_len, key_heads, value_heads = 2, 96, 2, 4
    key_dim = value_dim = chunk_size = 64

    k = torch.randn(
        batch_size,
        seq_len,
        key_heads,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    v = torch.randn(
        batch_size,
        seq_len,
        value_heads,
        value_dim,
        dtype=torch.float32,
        device=device,
    )
    batch_values = torch.arange(
        1, batch_size + 1, dtype=torch.float32, device=device
    ).view(batch_size, 1, 1)
    beta = batch_values.expand(batch_size, seq_len, value_heads).contiguous()
    g_cumsum = batch_values.log().expand_as(beta).contiguous()

    # Identity rows: token t activates column t % chunk_size, so the
    # kernel reduces to w = k * beta * exp(g), u = v * beta.
    A = torch.zeros(
        batch_size,
        seq_len,
        value_heads,
        chunk_size,
        dtype=torch.float32,
        device=device,
    )
    token_indices = torch.arange(seq_len, device=device)
    A[:, token_indices, :, token_indices % chunk_size] = 1

    w, u = recompute_w_u_fwd_npu(
        k,
        v,
        beta,
        g_cumsum,
        A,
        cu_seqlens=None,
    )

    expected_w = (
        k.repeat_interleave(value_heads // key_heads, dim=2)
        * beta.unsqueeze(-1)
        * g_cumsum.exp().unsqueeze(-1)
    )
    expected_u = v * beta.unsqueeze(-1)

    torch.testing.assert_close(w, expected_w)
    torch.testing.assert_close(u, expected_u)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
