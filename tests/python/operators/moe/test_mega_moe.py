"""Pytest owns the matrix; isolated workers own all HCCL collectives.

Run serially on a host with four or eight visible NPUs and CANN 9.2.0. See README.md
for the independent-reference exception and performance measurement method.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from .mega_moe_quant_worker import QUANT_MODES

ROOT = Path(__file__).resolve().parents[4]
WORKER = Path(__file__).with_name("mega_moe_worker.py")
perf_shape = [(32, 32, 7168, 2048)]


def require_hardware(request, generation, world_size=4):
    import torch
    from sgl_kernel_npu.operators.utils.generation import get_hardware

    workers = getattr(request.config, "workerinput", {}).get("workercount", 0)
    if workers:
        raise pytest.UsageError("MegaMoE tests require serial pytest; remove -n")
    if get_hardware().generation.value != generation:
        pytest.skip(f"requires {generation}")
    device_count = torch.npu.device_count()
    if device_count < 4:
        pytest.fail("MegaMoE integration tests require four visible NPUs")
    if device_count < world_size:
        pytest.skip(f"requires {world_size} visible NPUs in one HCCL group")


def run_workers(arguments, world_size=4):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(WORKER),
        *arguments,
    ]
    env = os.environ.copy()
    env.setdefault("HCCL_BUFFSIZE", "4096")
    env["MOE_EXPERT_TOKEN_NUMS_TYPE"] = "1"
    # Let torchrun manage worker failures; the outer timeout bounds setup too.
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=1200)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        # Killing only torchrun can leave workers holding NPU memory.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@pytest.mark.parametrize("impl", ["auto", "cann_ops_transformer"])
@pytest.mark.parametrize(
    "shape,routing,input_mode,topk",
    [
        pytest.param((1, 32, 2048, 1024), "balanced", "normal", 1, id="single-token"),
        pytest.param((17, 32, 2048, 1024), "balanced", "normal", 4, id="masked-tail"),
        pytest.param((32, 32, 2048, 1024), "skewed", "normal", 4, id="empty-experts"),
        pytest.param((17, 32, 2048, 1024), "balanced", "zero", 4, id="zero-input"),
        # DeepSeek-V3 H/I dimensions, reduced expert count for reference memory.
        pytest.param(
            (32, 32, 7168, 2048), "balanced", "normal", 4, id="deepseek-v3-dims"
        ),
    ],
)
@pytest.mark.extra_checks
def test_bf16_accuracy(request, impl, shape, routing, input_mode, topk):
    require_hardware(request, "arch22")
    tokens, capacity, hidden, intermediate = shape
    perf_override = request.config.getoption("enable_perf")
    enable_perf = shape in perf_shape if perf_override is None else perf_override
    run_workers(
        [
            "--tokens",
            str(tokens),
            "--capacity",
            str(capacity),
            "--hidden",
            str(hidden),
            "--intermediate",
            str(intermediate),
            "--topk",
            str(topk),
            "--routing",
            routing,
            "--input-mode",
            input_mode,
            "--impl",
            impl,
            "--perf-iters",
            "10" if enable_perf else "0",
        ]
    )


@pytest.mark.extra_checks
def test_mxfp4_accuracy_four_rank(request):
    require_hardware(request, "arch35")
    run_workers(
        [
            "--quant-mode",
            "mx_fp8_e4m3_w4",
            "--independent-mxfp4-reference",
            "--num-experts",
            "8",
            "--topk",
            "4",
            "--routing",
            "all-ranks",
        ]
    )


@pytest.mark.extra_checks
def test_bf16_accuracy_eight_rank_group(request):
    require_hardware(request, "arch22", world_size=8)
    run_workers(
        [
            "--tokens",
            "17",
            "--capacity",
            "32",
            "--num-experts",
            "16",
            "--topk",
            "8",
            "--routing",
            "all-ranks",
        ],
        world_size=8,
    )


@pytest.mark.parametrize("impl", ["auto", "cann_ops_transformer"])
@pytest.mark.parametrize(
    "generation,quant_mode",
    [(generation, mode) for generation, modes in QUANT_MODES.items() for mode in modes],
)
@pytest.mark.extra_checks
def test_quantized_public_interface(request, impl, generation, quant_mode):
    require_hardware(request, generation)
    run_workers(
        [
            "--quant-mode",
            quant_mode,
            "--impl",
            impl,
            "--hidden",
            "7168",
            "--intermediate",
            "3072",
            "--num-experts",
            "16",
            "--topk",
            "8",
            "--perf-iters",
            "10" if request.config.getoption("enable_perf") else "0",
        ]
    )
