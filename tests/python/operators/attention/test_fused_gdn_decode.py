"""Compare both outputs and modified state with independent inputs."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gdn_decode(varlen, normalize, dtype):
    from sgl_kernel_npu.operators.attention import fused_gdn_decode

    count, heads, value_heads, dim = 4, 2, 4, 64
    shape = (1, count) if varlen else (count, 1)
    q = torch.randn(*shape, heads, dim, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(*shape, value_heads, dim, dtype=dtype)
    a = torch.randn(count, value_heads, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(value_heads, dtype=torch.float32)
    bias = torch.randn(value_heads, dtype=torch.float32)
    state = torch.randn(count, value_heads, dim, dim, dtype=torch.float32)
    indices = torch.arange(count, dtype=torch.int32)
    offsets = torch.arange(count + 1, dtype=torch.int32) if varlen else None
    actual_state, ref_state = state.clone(), state.clone()
    args = (A_log, a, bias, 1.0, 20.0, q, k, v, b)
    kwargs = {"use_qk_l2norm_in_kernel": normalize, "cu_seqlens": offsets}
    reference = fused_gdn_decode(
        *args, ref_state, indices, impl="torch_native", **kwargs
    )
    actual = fused_gdn_decode(*args, actual_state, indices, impl="triton", **kwargs)
    torch.testing.assert_close(actual, reference, rtol=0.04, atol=0.04)
    torch.testing.assert_close(actual_state, ref_state, rtol=0.04, atol=0.04)
