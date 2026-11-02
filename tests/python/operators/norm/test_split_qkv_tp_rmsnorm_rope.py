import os
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance, performance_context
from tests.python.utils.standards import AccuracyStandard

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


def _check_outputs(executor, original, data):
    rows = original.shape[0]
    q_width, kv_width = data["q_hidden_size"], data["kv_hidden_size"]
    q, k, v = executor()
    assert q.shape == (rows, q_width)
    assert k.shape == v.shape == (rows, kv_width)
    ref_q, ref_k, ref_v = original.split((q_width, kv_width, kv_width), dim=1)
    torch.testing.assert_close(v, ref_v, atol=0, rtol=0)
    torch.testing.assert_close(data["input"], original, atol=0, rtol=0)
    if data["tp_world"] == 2:
        # Global mean squares: Q=(1+9)/2 and K=(4+16)/2.
        torch.testing.assert_close(q, ref_q / (5 + data["eps"]) ** 0.5)
        torch.testing.assert_close(k, ref_k / (10 + data["eps"]) ** 0.5)


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
        _check_outputs(executor, original, data)
    check_performance(executor, enable_perf=shape in perf_shape)


@contextmanager
def _hccl_group(rank, rendezvous):
    import torch.distributed as dist

    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            "hccl",
            init_method=rendezvous,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=120),
        )
    try:
        yield dist.group.WORLD
    finally:
        if created:
            dist.destroy_process_group()


def _check_two_rank_worker(rank, shape, dtype, rendezvous, options):
    import torch_npu

    device = int(os.environ["LOCAL_RANK"]) if rendezvous == "env://" else rank
    csv = options["csv_path"]
    if csv:
        path = Path(csv)
        csv = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
    # Spawned ranks need the pytest options and independent profiling reports.
    with torch.npu.device(device), _hccl_group(
        rank, rendezvous
    ) as group, performance_context(
        options["nodeid"],
        {"shape": shape, "dtype": dtype, "rank": rank},
        None,
        enable_perf=options["enable_perf"],
        prof_dir=options["prof_dir"],
        csv_path=csv,
    ):
        from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

        data = generate_data(shape, dtype, tp_group=group)
        original = data["input"].clone()
        executor = check_accuracy(
            split_qkv_tp_rmsnorm_rope,
            kwargs=data,
            impl=options["impl"],
            impl_ref="cpu",
            skip_acc_check=bool(options["skip_acc_check"]),
        )
        if executor.accuracy_checked:
            _check_outputs(executor, original, data)
        check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("shape,dtype", [((1, 128, 64, 64, 64), torch.float32)])
def test_two_rank(request, tmp_path, monkeypatch, shape, dtype):
    opt = request.config.getoption
    options = {
        k: opt(k)
        for k in ("impl", "skip_acc_check", "enable_perf", "prof_dir", "csv_path")
    }
    options["nodeid"] = request.node.nodeid
    if "HCCL_NPU_SOCKET_PORT_RANGE" not in os.environ:
        monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    if "RANK" in os.environ:
        if os.environ.get("WORLD_SIZE") != "2":
            pytest.skip("requires two ranks")
        _check_two_rank_worker(int(os.environ["RANK"]), shape, dtype, "env://", options)
        return
    if opt("npu_device") is not None or torch.npu.device_count() < 2:
        pytest.skip("requires two visible NPUs without --npu-device")
    torch.multiprocessing.spawn(
        _check_two_rank_worker,
        nprocs=2,
        args=(shape, dtype, (tmp_path / "tp_store").as_uri(), options),
    )
