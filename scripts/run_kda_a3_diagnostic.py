"""Repeat the KDA B8-S8-framework accuracy check with fixed A3 configurations."""

import argparse
import hashlib
import importlib
import importlib.metadata
import os
import re
import subprocess
import tempfile
from pathlib import Path


def npu_smi(*args):
    cmd = ["npu-smi", "info", *args]
    print(f"$ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"npu-smi unavailable: {error}")
        return ""
    print(result.stdout, end="")
    print(result.stderr, end="")
    print(f"npu-smi exit={result.returncode}")
    return result.stdout


def report_environment(torch):
    from triton.backends.ascend.utils import _get_npucompiler_path

    print(f"Device: npu:0, SoC: {torch.npu.get_device_name(0)}")
    for package in ("torch", "torch-npu", "triton", "triton-ascend", "sgl-kernel-npu"):
        print(f"{package}: {importlib.metadata.version(package)}")
    for name in (
        "ASCEND_HOME_PATH",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "TASK_QUEUE_ENABLE",
        "PYTORCH_NPU_ALLOC_CONF",
        "TRITON_ALL_BLOCKS_PARALLEL",
        "TRITON_NPU_COMPILER_PATH",
    ):
        print(f"{name}={os.environ.get(name, '<unset>')}")

    compiler = Path(_get_npucompiler_path()[0]).resolve()
    with compiler.open("rb") as stream:
        print(
            f"BiShengIR: {compiler} sha256={hashlib.file_digest(stream, 'sha256').hexdigest()}"
        )

    npu_smi()
    mapping = npu_smi("-m")
    cards = set(re.findall(r"(?m)^\s*NPU ID\s*:\s*(\d+)", npu_smi("-l")))
    chips = set(re.findall(r"(?m)^[ \t]*(\d+)[ \t]+\S+[ \t]+(\d+)[ \t]+\d+", mapping))
    for card in sorted(cards | {card for card, _ in chips}, key=int):
        npu_smi("-t", "board", "-i", card)
    for card, chip in sorted(chips):
        npu_smi("-t", "board", "-i", card, "-c", chip)
    for component in ("driver", "firmware"):
        path = Path("/usr/local/Ascend") / component / "version.info"
        if path.is_file():
            print(f"{path}:\n{path.read_text().rstrip()}")
        else:
            print(f"{path}: unavailable")


def report_snapshot_difference(torch, kwargs, inputs, reference):
    actual = kwargs["intermediate_states_buffer"].cpu()[inputs["snapshot_indices"]]
    expected = reference.cpu()
    for step in range(inputs["steps"]):
        observed, expected_step = actual[:, step], expected[:, step]
        bad = ~torch.isclose(observed, expected_step, atol=1e-5, rtol=1e-5)
        if not bad.any():
            continue
        error = (observed - expected_step).abs().nan_to_num(nan=float("inf"))
        index = torch.unravel_index(error.masked_fill(~bad, -1).argmax(), error.shape)
        index = tuple(int(dim) for dim in index)
        print(
            f"snapshot step={step} index={index} bad={bad.sum().item()} "
            f"actual={observed[index].item()} expected={expected_step[index].item()}"
        )
        if step + 1 < inputs["steps"]:
            print(f"expected_next={expected[:, step + 1][index].item()}")
        break


def run_config(torch, test, tuner, config):
    tuner.configs = [config]
    failed = 0
    repeats = 20
    print(f"B8-S8-framework repeat={repeats} fixed config={config}")
    for trial in range(1, repeats + 1):
        kwargs, inputs = test.generate_data(
            8, 8, "framework", "preactivated", "omitted", torch.float32
        )
        expected = test._reference(kwargs)
        output = test.kda_target_verify_npu(**kwargs)
        torch.npu.synchronize()
        if trial == 1:
            print(f"Selected config: {tuner.best_config}")
        try:
            test._check_result(output, kwargs, inputs, expected)
        except AssertionError as error:
            failed += 1
            print(f"Trial {trial}/{repeats}: FAIL {error}")
            report_snapshot_difference(torch, kwargs, inputs, expected[1])
        else:
            print(f"Trial {trial}/{repeats}: PASS")
    print(
        f"B8-S8-framework {config.kwargs}: {repeats - failed} passed, {failed} failed"
    )
    return failed


def run_diagnostic(multibuffer):
    import torch
    import torch_npu  # noqa: F401 -- registers torch.npu

    from tests.python.operators.attention.kda import test_kda_target_verify
    from tests.python.utils.common import use_impls

    torch.npu.set_device(0)
    report_environment(torch)
    impl = importlib.import_module(
        "sgl_kernel_npu.implementation.triton.attention.kda.kda_target_verify"
    )
    print(f"KDA source: {impl.__file__}")
    tuner = impl._kda_target_verify_kernel
    (config,) = [
        c
        for c in tuner.configs
        if c.kwargs["BV"] == 64 and c.kwargs["multibuffer"] == multibuffer
    ]
    with use_impls("auto", "torch_native", skip_acc_check=False):
        return int(run_config(torch, test_kda_target_verify, tuner, config) != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("multibuffer", choices=("false", "true"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="kda-a3-triton-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        status = run_diagnostic(args.multibuffer == "true")
    raise SystemExit(status)
