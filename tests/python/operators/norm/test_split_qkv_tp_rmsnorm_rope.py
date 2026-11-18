import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Shape: (rows, Q, KV, head, rotary). MiniMax-M2 TP=1 uses rotary=64.
# Batches are synthetic; full RoPE is additional generic coverage.
perf_shape = [(12, 6144, 1024, 128, 128), (12, 6144, 1024, 128, 64)]


def generate_data(shape, dtype):
    rows, q_width, kv_width, head_dim, rotary_dim = shape
    generator = torch.Generator(device="cpu").manual_seed(17)

    def random(*shape):
        return torch.randn(shape, dtype=dtype, device="cpu", generator=generator).npu()

    return dict(
        input=random(rows, q_width + 2 * kv_width),
        cos=torch.rand((rows, rotary_dim), device="cpu", generator=generator)
        .to(dtype)
        .npu(),
        sin=torch.rand((rows, rotary_dim), device="cpu", generator=generator)
        .to(dtype)
        .npu(),
        q_hidden_size=q_width,
        kv_hidden_size=kv_width,
        head_dim=head_dim,
        eps=1e-6,
        q_weight=random(q_width),
        k_weight=random(kv_width),
        rotary_dim=rotary_dim,
        tp_world=1,
        tp_group=None,
    )


@pytest.mark.parametrize(
    "shape,dtype",
    [
        pytest.param(
            (12, 6144, 1024, 128, 64),
            torch.bfloat16,
            id="minimax-m2-tp1-partial-rope-bf16",
        ),
        pytest.param((12, 6144, 1024, 128, 128), torch.bfloat16, id="full-rope-bf16"),
        pytest.param((1, 512, 128, 64, 32), torch.float32, id="partial-rope-fp32"),
        pytest.param((3, 512, 128, 64, 32), torch.float16, id="partial-rope-fp16"),
        pytest.param(
            (0, 512, 128, 64, 64),
            torch.bfloat16,
            marks=pytest.mark.extra_checks,
            id="empty-batch",
        ),
    ],
)
def test_single_rank(shape, dtype):
    from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

    data = generate_data(shape, dtype)
    original = data["input"].clone()
    impl, impl_ref = get_test_impls()
    executor = check_accuracy(
        split_qkv_tp_rmsnorm_rope,
        kwargs=data,
        impl=impl,
        impl_ref=impl_ref,
        standard={"torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=1e-7)},
    )
    if executor.accuracy_checked:
        rows, q_width, kv_width, _, _ = shape
        q, k, v = executor()
        assert q.shape == (rows, q_width)
        assert k.shape == v.shape == (rows, kv_width)
        torch.testing.assert_close(v, original[:, q_width + kv_width :], atol=0, rtol=0)
        torch.testing.assert_close(data["input"], original, atol=0, rtol=0)
    check_performance(executor, enable_perf=shape in perf_shape)
