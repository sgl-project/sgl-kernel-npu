"""Repeat the KDA B8-S8-framework accuracy check with fixed A3 configurations."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch


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


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay_compiler(directory, command, source_arg, environment):
    source = directory / "input.ttadapter.mlir"
    compiler = Path(command[0]).resolve()
    print(f"Replay compiler SHA-256: {sha256(compiler)}")
    print(f"Shared input IR SHA-256: {sha256(source)}")
    help_result = subprocess.run(
        [compiler, "--help-hidden"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        timeout=30,
        text=True,
        errors="replace",
        check=False,
    )
    (directory / "compiler-help.txt").write_text(help_result.stdout)
    dump_flags = []
    for flag in ("--mlir-print-ir-after-all", "--bishengir-print-ir-after-all"):
        if flag in help_result.stdout:
            dump_flags.append(flag)
            break
    if not dump_flags:
        dump_flags.append("--mlir-print-ir-after-all")
    if "--mlir-disable-threading" in help_result.stdout:
        dump_flags.append("--mlir-disable-threading")

    for target in ("Ascend910_9392", "Ascend910_9362"):
        for multibuffer in ("True", "False"):
            dest = directory / f"{target}-multibuffer-{multibuffer}"
            dest.mkdir()
            args = list(command)
            args[0] = str(compiler)
            args[source_arg] = str(source)
            output_arg = args.index("-o") + 1
            args[output_arg] = str(dest / "kernel")
            for i, arg in enumerate(args):
                if arg.startswith("--target="):
                    args[i] = f"--target={target}"
                if arg.startswith("--enable-auto-multi-buffer="):
                    args[i] = f"--enable-auto-multi-buffer={multibuffer}"
            dump_args = args + dump_flags
            dump_args[output_arg] = str(dest / "dump-kernel")
            commands = {"standard": args, "dump": dump_args}
            (dest / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            for mode, invocation in commands.items():
                log_path = dest / f"{mode}.log"
                with log_path.open("wb") as log:
                    try:
                        result = subprocess.run(
                            invocation,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=environment,
                            cwd=dest,
                            timeout=120,
                            check=False,
                        )
                        status = result.returncode
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                with log_path.open(errors="replace") as log:
                    dumps = sum("IR Dump After" in line for line in log)
                print(f"{dest.name} {mode}: status={status}, IR dumps={dumps}")
            for artifact in sorted(dest.glob("*.o")):
                print(f"{dest.name}/{artifact.name}: sha256={sha256(artifact)}")
                if readelf := shutil.which("readelf"):
                    with (dest / f"{artifact.name}.elf.txt").open("w") as report:
                        subprocess.run(
                            [readelf, "-SWr", "-x", ".text", artifact],
                            stdout=report,
                            stderr=subprocess.STDOUT,
                            timeout=30,
                            check=False,
                        )


def run_and_replay(multibuffer, directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    real_run = subprocess.run
    captured = None

    def capture(command, *args, **kwargs):
        nonlocal captured
        input_path = None
        if (
            captured is None
            and isinstance(command, (tuple, list))
            and Path(command[0]).name == "bishengir-compile"
        ):
            for index, arg in enumerate(command[1:], start=1):
                if str(arg).endswith(".ttadapter.mlir") and (
                    "_kda_target_verify_kernel" in Path(arg).read_text()
                ):
                    input_path = Path(arg)
                    shutil.copyfile(input_path, directory / "input.ttadapter.mlir")
                    captured = (
                        [str(item) for item in command],
                        index,
                        kwargs.get("env"),
                    )
                    break
        result = real_run(command, *args, **kwargs)
        if input_path is not None:
            runtime = directory / "runtime-compile"
            runtime.mkdir()
            for artifact in sorted(input_path.parent.glob("*.o")):
                shutil.copyfile(artifact, runtime / artifact.name)
                print(f"Runtime {artifact.name}: sha256={sha256(artifact)}")
        return result

    with patch.object(subprocess, "run", capture):
        status = run_diagnostic(multibuffer)
    if captured is None:
        raise RuntimeError("No BiShengIR invocation was captured for the KDA input")
    command, source_arg, environment = captured
    if not all(
        any(arg.startswith(prefix) for arg in command)
        for prefix in ("--target=", "--enable-auto-multi-buffer=")
    ):
        raise RuntimeError(f"Missing target or multibuffer compiler option: {command}")
    (directory / "recorded-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    replay_compiler(directory, command, source_arg, environment)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("multibuffer", choices=("false", "true"))
    parser.add_argument("--ir-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="kda-a3-triton-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        if args.ir_dir is None:
            status = run_diagnostic(args.multibuffer == "true")
        else:
            status = run_and_replay(args.multibuffer == "true", args.ir_dir)
    raise SystemExit(status)
