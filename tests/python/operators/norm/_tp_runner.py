"""Run a pytest operator check on two local NPU ranks."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from tests.python.utils.common import use_impls
from tests.python.utils.performance import performance_context


def _run_rank(local_rank, rank, rendezvous, check, args, options):
    import torch.distributed as dist
    import torch_npu

    csv = options["csv_path"]
    if csv:
        path = Path(csv)
        csv = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
    with torch.npu.device(local_rank), use_impls(
        *options["impls"], skip_acc_check=options["skip_acc_check"]
    ), performance_context(
        options["nodeid"],
        {**options["params"], "rank": rank},
        None,
        enable_perf=options["enable_perf"],
        prof_dir=options["prof_dir"],
        csv_path=csv,
    ):
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
            check(*args, tp_group=dist.group.WORLD)
        finally:
            if created:
                dist.destroy_process_group()


def _local_rank(rank, *args):
    _run_rank(rank, rank, *args)


def run_two_ranks(request, tmp_path, monkeypatch, check, *args):
    if "HCCL_NPU_SOCKET_PORT_RANGE" not in os.environ:
        monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    opt = request.config.getoption
    options = {
        k: opt(k) for k in ("enable_perf", "skip_acc_check", "prof_dir", "csv_path")
    }
    options.update(
        impls=(opt("impl"), opt("impl_ref")),
        nodeid=request.node.nodeid,
        params=request.node.callspec.params,
    )
    if "RANK" in os.environ:
        if os.environ.get("WORLD_SIZE") != "2":
            pytest.skip("requires two ranks")
        _run_rank(
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["RANK"]),
            "env://",
            check,
            args,
            options,
        )
    else:
        if opt("npu_device") is not None or torch.npu.device_count() < 2:
            pytest.skip("requires two visible NPUs without --npu-device")
        torch.multiprocessing.spawn(
            _local_rank,
            args=((tmp_path / "tp_store").as_uri(), check, args, options),
            nprocs=2,
            join=True,
        )
