import os
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

from ._inputs import make_tp_inputs


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"rows": 1}, id="decode"),
        pytest.param({"rotary_dim": 32}, id="multi-chunk-partial"),
        pytest.param(
            {
                "rows": 3,
                "q_width": 6144,
                "kv_width": 1024,
                "head_dim": 128,
                "rotary_dim": 128,
            },
            id="minimax-borrowed-local-widths",
        ),
        pytest.param({"rows": 0}, id="empty"),
    ],
)
# Compare full-local-hidden RMS statistics, intermediate casts and RoPE without a collective.
def test_split_qkv_tp_single_rank(options, dtype):
    from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

    data = make_tp_inputs(dtype, **options)
    snapshot = data["input"].clone()
    executor = check_accuracy(
        split_qkv_tp_rmsnorm_rope, kwargs=data, impl="triton", impl_ref="torch_native"
    )
    q, k, v = executor()
    assert q.shape == (snapshot.shape[0], data["q_hidden_size"])
    assert k.shape == v.shape == (snapshot.shape[0], data["kv_hidden_size"])
    torch.testing.assert_close(
        v, snapshot[:, data["q_hidden_size"] + data["kv_hidden_size"] :], atol=0, rtol=0
    )
    torch.testing.assert_close(data["input"], snapshot, atol=0, rtol=0)
    check_performance(executor)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="Run this case with torchrun on at least two NPUs",
)
# Exercise real HCCL SUM reductions for both implementations using rank-specific inputs.
def test_split_qkv_tp_multi_rank():
    import torch.distributed as dist
    from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

    local_rank = int(os.environ["LOCAL_RANK"])
    created_group = False
    with torch.npu.device(local_rank):
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="hccl", init_method="env://", timeout=timedelta(seconds=120)
                )
                created_group = True
            rank = dist.get_rank()
            data = make_tp_inputs(torch.bfloat16, seed=42 + rank)
            data.update(tp_world=dist.get_world_size(), tp_group=dist.group.WORLD)
            snapshot = data["input"].clone()
            executor = check_accuracy(
                split_qkv_tp_rmsnorm_rope,
                kwargs=data,
                impl="triton",
                impl_ref="torch_native",
            )
            q, k, v = executor()
            assert q.shape == (snapshot.shape[0], data["q_hidden_size"])
            assert k.shape == v.shape == (snapshot.shape[0], data["kv_hidden_size"])
            torch.testing.assert_close(
                v,
                snapshot[:, data["q_hidden_size"] + data["kv_hidden_size"] :],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(data["input"], snapshot, atol=0, rtol=0)
            torch.npu.synchronize()
        finally:
            if created_group:
                dist.destroy_process_group()
