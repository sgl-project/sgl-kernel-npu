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
    "options",
    [
        pytest.param({"rows": 1}, id="decode-gqa"),
        pytest.param({"head_dim": 192, "rope_dim": 96}, id="head192-partial"),
        pytest.param({"eps": None}, id="rope-only"),
        pytest.param({"bias": True}, id="paired-bias"),
        pytest.param(
            {"is_neox_style": False, "rope_dim": 32}, id="adjacent-pair-partial"
        ),
        pytest.param(
            {"strided": True, "cache_dtype": torch.bfloat16}, id="strided-mixed-cache"
        ),
        pytest.param({"rows": 0}, id="empty"),
    ],
)
# Compare Q/K normalization and rotary modes while requiring an exact, unmodified V copy.
def test_split_qkv_rmsnorm_rope(options, dtype):
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

    data = make_qkv_inputs(dtype, **options)
    snapshot = data["input"].clone()
    executor = check_accuracy(
        split_qkv_rmsnorm_rope, kwargs=data, impl="triton", impl_ref="torch_native"
    )
    q, k, v = executor()
    rows = data["input"].shape[0]
    assert q.shape == (rows, data["q_hidden_size"])
    assert k.shape == v.shape == (rows, data["kv_hidden_size"])
    torch.testing.assert_close(
        v, snapshot[:, data["q_hidden_size"] + data["kv_hidden_size"] :], atol=0, rtol=0
    )
    torch.testing.assert_close(data["input"], snapshot, atol=0, rtol=0)
    check_performance(executor)
