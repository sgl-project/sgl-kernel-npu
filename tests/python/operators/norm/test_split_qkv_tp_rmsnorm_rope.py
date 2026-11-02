import os
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard


def make_inputs(rows, q_width, kv_width, head_dim, rotary_dim, dtype):
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
    "rows,q_width,kv_width,head_dim,rotary_dim,dtype",
    [
        (12, 6144, 1024, 128, 128, torch.bfloat16),
        (1, 512, 128, 64, 32, torch.float32),
        (3, 512, 128, 64, 32, torch.float16),
        (0, 512, 128, 64, 64, torch.bfloat16),
    ],
)
def test_single_rank(rows, q_width, kv_width, head_dim, rotary_dim, dtype):
    from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

    data = make_inputs(rows, q_width, kv_width, head_dim, rotary_dim, dtype)
    original = data["input"].clone()
    executor = check_accuracy(
        split_qkv_tp_rmsnorm_rope,
        kwargs=data,
        impl_ref="cpu",
        standard={"torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=1e-7)},
    )
    q, k, v = executor()
    assert q.shape == (rows, q_width)
    assert k.shape == v.shape == (rows, kv_width)
    torch.testing.assert_close(v, original[:, q_width + kv_width :], atol=0, rtol=0)
    torch.testing.assert_close(data["input"], original, atol=0, rtol=0)
    check_performance(executor, enable_perf=rows == 12)


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2", reason="requires two torchrun ranks"
)
def test_two_rank():
    import torch.distributed as dist

    with torch.npu.device(int(os.environ["LOCAL_RANK"])):
        from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

        created = not dist.is_initialized()
        if created:
            dist.init_process_group("hccl", timeout=timedelta(seconds=120))
        try:
            rank = dist.get_rank()
            data = make_inputs(1, 128, 64, 64, 64, torch.float32)
            data.update(tp_world=2, tp_group=dist.group.WORLD)
            q, k, v = data["input"].split((128, 64, 64), 1)
            q.fill_(1 + 2 * rank)
            k.fill_(2 + 2 * rank)
            v.fill_(11 + rank)
            data["cos"].fill_(1)
            data["sin"].zero_()
            data["q_weight"].fill_(1)
            data["k_weight"].fill_(1)
            original = data["input"].clone()
            executor = check_accuracy(
                split_qkv_tp_rmsnorm_rope, kwargs=data, impl_ref="cpu"
            )
            actual_q, actual_k, actual_v = executor()
            # Across ranks, mean squares are (1+9)/2 for Q and (4+16)/2 for K.
            torch.testing.assert_close(actual_q, q / (5 + data["eps"]) ** 0.5)
            torch.testing.assert_close(actual_k, k / (10 + data["eps"]) ** 0.5)
            torch.testing.assert_close(actual_v, v, atol=0, rtol=0)
            torch.testing.assert_close(data["input"], original, atol=0, rtol=0)
        finally:
            if created:
                dist.destroy_process_group()
