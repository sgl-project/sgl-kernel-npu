import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

from ._inputs import make_position_cache, make_qkv_inputs

# Preserve the existing position-cache test's golden tolerances. The original
# kernel rounds through BF16 even for FP32/FP16 outputs; this is not a new threshold.
POSITION_STANDARD = {
    str(dtype): AccuracyStandard(atol=5e-2, rtol=5e-3)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
}


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "options,cast,threshold,position_dtype",
    [
        pytest.param({"rows": 1}, True, 1024, torch.int64, id="decode-per-head"),
        pytest.param(
            {"rows": 7, "rope_dim": 32}, True, 1, torch.int32, id="wide-partial"
        ),
        pytest.param({"eps": None}, True, 1024, torch.int64, id="rope-only"),
        pytest.param(
            {"bias": True}, False, 1, torch.int64, id="bias-without-pre-rope-cast"
        ),
        pytest.param({"rows": 7}, False, 1024, torch.int32, id="position-clamp"),
        pytest.param({"rows": 0}, True, 1024, torch.int64, id="empty"),
    ],
)
# Compare position conversion/clamping, both grids, norm/bias and the original BF16 cast stages.
def test_split_qkv_position_cache(options, cast, threshold, position_dtype, dtype):
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    data = make_qkv_inputs(dtype, **options)
    x = data["input"]
    snapshot = x.clone()
    rows = x.shape[0]
    rope_dim = data["sin"].shape[-1]
    cache = make_position_cache(32, rope_dim)
    choices = torch.tensor([0, 1, 7, 31, -2, 35, 3], dtype=position_dtype)
    positions = choices[:rows]
    kwargs = {
        name: data[name]
        for name in (
            "q_hidden_size",
            "kv_hidden_size",
            "head_dim",
            "eps",
            "q_weight",
            "k_weight",
            "q_bias",
            "k_bias",
        )
    }
    kwargs.update(
        input_tensor=x,
        positions=positions,
        cos_sin_cache=cache,
        rope_dim=rope_dim,
        cast_norm_to_bf16=cast,
        wide_grid_min_tokens=threshold,
    )
    executor = check_accuracy(
        split_qkv_rmsnorm_rope_pos_cache_half_npu,
        kwargs=kwargs,
        impl="triton",
        impl_ref="torch_native",
        standard=POSITION_STANDARD,
    )
    q, k, v = executor()
    assert q.shape == (rows, data["q_hidden_size"])
    assert k.shape == v.shape == (rows, data["kv_hidden_size"])
    torch.testing.assert_close(
        v, snapshot[:, data["q_hidden_size"] + data["kv_hidden_size"] :], atol=0, rtol=0
    )
    torch.testing.assert_close(x, snapshot, atol=0, rtol=0)
    check_performance(executor)
