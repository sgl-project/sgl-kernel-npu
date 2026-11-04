import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# (rows, local Q width, local KV width, head size, rotary width)
perf_shape = [(12, 6144, 1024, 128, 128)]


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
        ((12, 6144, 1024, 128, 128), torch.bfloat16),
        ((1, 512, 128, 64, 32), torch.float32),
        ((3, 512, 128, 64, 32), torch.float16),
        pytest.param(
            (0, 512, 128, 64, 64), torch.bfloat16, marks=pytest.mark.extra_checks
        ),
    ],
)
def test_single_rank(shape, dtype):
    from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

    data = generate_data(shape, dtype)
    original = data["input"].clone()
    executor = check_accuracy(
        split_qkv_tp_rmsnorm_rope,
        kwargs=data,
        impl_ref="cpu",
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
