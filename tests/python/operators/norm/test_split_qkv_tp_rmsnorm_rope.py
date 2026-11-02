import csv
import os
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from time import perf_counter_ns

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
    return q, k, v


def _check_full_tp_output(original, outputs, data, group):
    import torch.distributed as dist

    # Each rank contributes [input QKV | output Q | output K | output V].
    packed = torch.cat((original, *outputs), dim=1)
    peers = [torch.empty_like(packed) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, packed, group=group)
    if dist.get_rank(group) == 0:
        q, k = data["q_hidden_size"], data["kv_hidden_size"]
        widths = (q, k, k, q, k, k)
        columns = zip(*(peer.cpu().split(widths, dim=1) for peer in peers))
        xq, xk, xv, actual_q, actual_k, actual_v = [
            torch.cat(parts, dim=1) for parts in columns
        ]
        # This case has unit weights and identity RoPE. Normalize the unsharded input.
        for actual, x in ((actual_q, xq), (actual_k, xk)):
            expected = x * torch.rsqrt(
                x.float().square().mean(dim=1, keepdim=True) + data["eps"]
            )
            torch.testing.assert_close(actual, expected.to(x.dtype))
        torch.testing.assert_close(actual_v, xv, atol=0, rtol=0)


def _measure_tp_latency(executor, group, device, options):
    import torch.distributed as dist

    # The per-rank profiler has already warmed up both ranks. Keep the start
    # barrier and aggregation outside each synchronized operator call.
    elapsed_us = []
    for _ in range(5):
        dist.barrier(group=group)
        torch.npu.synchronize()
        start = perf_counter_ns()
        executor()
        torch.npu.synchronize()
        elapsed_us.append((perf_counter_ns() - start) / 1000)
    samples = torch.tensor(elapsed_us, device=device, dtype=torch.float32)
    dist.all_reduce(samples, op=dist.ReduceOp.MAX, group=group)
    if dist.get_rank(group) == 0:
        latency = samples.mean().item()
        print(f"TP=2 mean max-rank wall time: {latency:.3f} us", flush=True)
        if options["csv_path"]:
            path = Path(options["csv_path"]).expanduser()
            path = path.with_name(f"{path.stem}.tp2{path.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", newline="") as stream:
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow(
                        ["case", "impl", "samples", "mean_max_rank_wall_us"]
                    )
                writer.writerow(
                    [options["nodeid"], executor.impl, len(elapsed_us), latency]
                )


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
            outputs = _check_outputs(executor, original, data)
            _check_full_tp_output(original, outputs, data, group)
        profile = check_performance(executor, enable_perf=shape in perf_shape)
        if profile is not None:
            _measure_tp_latency(executor, group, original.device, options)


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
