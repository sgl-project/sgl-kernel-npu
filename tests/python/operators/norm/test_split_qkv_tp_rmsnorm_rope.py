import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

from ._tp_runner import run_two_ranks

# (rows, local Q width, local KV width, head size, rotary width)
perf_shape = [(12, 6144, 1024, 128, 128)]


def generate_data(shape, dtype, tp_group=None):
    rows, q_width, kv_width, head_dim, rotary_dim = shape
    generator = torch.Generator(device="cpu").manual_seed(17)

    def random(*shape):
        return torch.randn(shape, dtype=dtype, device="cpu", generator=generator).npu()

    data = dict(
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

    if tp_group is not None:
        rank = torch.distributed.get_rank(tp_group)
        data.update(tp_world=2, tp_group=tp_group)
        q, k, v = data["input"].split((q_width, kv_width, kv_width), 1)
        q.fill_(1 + 2 * rank)
        k.fill_(2 + 2 * rank)
        v.fill_(11 + rank)
        data["cos"].fill_(1)
        data["sin"].zero_()
        data["q_weight"].fill_(1)
        data["k_weight"].fill_(1)
    return data


def _check_case(shape, dtype, tp_group=None):
    from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

    data = generate_data(shape, dtype, tp_group)
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
        ref_q, ref_k, ref_v = original.split((q_width, kv_width, kv_width), dim=1)
        torch.testing.assert_close(v, ref_v, atol=0, rtol=0)
        torch.testing.assert_close(data["input"], original, atol=0, rtol=0)
        if tp_group is not None:
            # Global mean squares: Q=(1+9)/2 and K=(4+16)/2.
            torch.testing.assert_close(q, ref_q / (5 + data["eps"]) ** 0.5)
            torch.testing.assert_close(k, ref_k / (10 + data["eps"]) ** 0.5)
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((12, 6144, 1024, 128, 128), torch.bfloat16),
        ((1, 512, 128, 64, 32), torch.float32),
        ((3, 512, 128, 64, 32), torch.float16),
        ((0, 512, 128, 64, 64), torch.bfloat16),
    ],
)
def test_single_rank(shape, dtype):
    _check_case(shape, dtype)


@pytest.mark.parametrize("shape,dtype", [((1, 128, 64, 64, 64), torch.float32)])
def test_two_rank(request, tmp_path, monkeypatch, shape, dtype):
    run_two_ranks(request, tmp_path, monkeypatch, _check_case, shape, dtype)
