import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

from ._inputs import make_qkv_inputs


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "rows,head_dim,rope_dim",
    [
        pytest.param(1, 64, 64, id="decode-full"),
        pytest.param(7, 128, 64, id="prefill-partial"),
        pytest.param(0, 64, 64, id="empty"),
    ],
)
# Verify Gemma's 1+weight normalization, head-interleaved Q/gate layout and exact V/gate copies.
def test_split_qkvgate_gemma_rmsnorm_rope(rows, head_dim, rope_dim, dtype):
    from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

    data = make_qkv_inputs(dtype, rows=rows, head_dim=head_dim, rope_dim=rope_dim)
    q_width, kv_width = data["q_hidden_size"], data["kv_hidden_size"]
    x = torch.randn(rows, 2 * q_width + 2 * kv_width, dtype=dtype)
    snapshot = x.clone()
    kwargs = {
        name: data[name]
        for name in (
            "sin",
            "cos",
            "q_hidden_size",
            "kv_hidden_size",
            "head_dim",
            "eps",
            "q_weight",
            "k_weight",
        )
    }
    kwargs.update(input=x, rope_dim=rope_dim)
    executor = check_accuracy(
        split_qkvgate_gemma_rmsnorm_rope,
        kwargs=kwargs,
        impl="triton",
        impl_ref="torch_native",
    )
    q, k, v, gate = executor()
    assert q.shape == gate.shape == (rows, q_width)
    assert k.shape == v.shape == (rows, kv_width)
    expected_gate = (
        snapshot[:, : 2 * q_width]
        .reshape(rows, q_width // head_dim, 2 * head_dim)[..., head_dim:]
        .reshape(rows, q_width)
    )
    torch.testing.assert_close(gate, expected_gate, atol=0, rtol=0)
    torch.testing.assert_close(v, snapshot[:, 2 * q_width + kv_width :], atol=0, rtol=0)
    torch.testing.assert_close(x, snapshot, atol=0, rtol=0)
    check_performance(executor)
