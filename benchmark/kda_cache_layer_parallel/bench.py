"""Validate and benchmark the two KDA cache movers against their serial baseline.

From the repository root, on an idle NPU:
    python benchmark/kda_cache_layer_parallel/bench.py

No arguments are needed. Results are written to results/kda_cache_layer_parallel/.
TRITON_INTERPRET=1 runs only the small correctness matrix on CPU, never timing.

Batch sweep: upstream PR #831. Timing protocol: three warmups and twenty
alternating-order NPU event samples, as in PR #838. Model dimensions come from
the supplied Kimi3.1 config. TP=16 and draft=4 are fixed TEST choices, not inferred
deployment settings. This allocates one shard on ONE NPU; no model or server is
loaded. This is not a reproduction of #831's unpublished full configuration.
"""

import csv
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WARMUP = 3
REPEAT = 20
# Kimi3.1: 93 total layers, 24 full-attention layers, 69 KDA layers.
KDA_LAYERS = 69
TOTAL_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL = 4
# Fixed, editable test choices, not command-line or distributed-launch options.
TEST_TP = 16
DRAFT_TOKENS = 4
assert TOTAL_HEADS % TEST_TP == 0
LOCAL_HEADS = TOTAL_HEADS // TEST_TP
POOL_SIZE = max(BATCH_SIZES) + 1
MOVE_SHAPE = dict(
    layers=KDA_LAYERS,
    heads=LOCAL_HEADS,
    value_dim=HEAD_DIM,
    key_dim=HEAD_DIM,
    drafts=DRAFT_TOKENS,
)
CONV_SHAPE = dict(
    layers=KDA_LAYERS,
    channels=3 * LOCAL_HEADS * HEAD_DIM,
    window=CONV_KERNEL - 1 + DRAFT_TOKENS - 1,
    drafts=DRAFT_TOKENS,
)
# SGLang's default cache types when config/env do not override them.
MODEL_CASES = (
    ("conv", CONV_SHAPE, torch.bfloat16, False),
    ("move", MOVE_SHAPE, torch.float32, True),
)
BASELINE_COMMIT = "458e72caf6cdfb5b9c06711e6f84d134d15e3f97"
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path("results/kda_cache_layer_parallel")


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_operators():
    if os.environ.get("TRITON_INTERPRET") == "1":
        device = "cpu"
    else:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("An NPU is required; CPU mode cannot benchmark kernels.")
        device = "npu"
    source = (
        REPO
        / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/triton/mamba/mamba_state_update.py"
    )
    # Load this checkout, not a possibly stale wheel from site-packages.
    optimized = load_source("kda_layer_parallel_under_test", source)
    baseline = load_source(
        "kda_serial_baseline", Path(__file__).with_name("baseline.py")
    )
    return device, baseline, optimized


def correctness_cases():
    return itertools.product(
        ("move", "conv"),
        ((1, 1), (4, 0), (4, 1), (32, 1), (4, 8), (4, 128)),
        (False, True),
        (torch.float32, torch.bfloat16),
    )


def generate_data(
    op, batch, shape, device, dtype, transposed=True, mixed=False, pool=None
):
    """Independent destination buffers; unique slots; immutable source and reset data."""
    torch.manual_seed(42)
    pool = pool if pool is not None else 2 * batch + 3
    if pool < batch + 1:
        raise ValueError("pool must leave at least one unselected slot")
    layers, drafts = shape["layers"], shape["drafts"]
    indices = list(reversed(range(1, batch + 1)))
    steps = [((0, -1, 1, drafts - 1)[i % 4] if mixed else 0) for i in range(batch)]
    indices_t = torch.tensor(indices, dtype=torch.int32, device=device)
    steps_t = torch.tensor(steps, dtype=torch.int32, device=device)

    if op == "move":
        h, v, k = shape["heads"], shape["value_dim"], shape["key_dim"]
        # Scratch states only need batch slots; destination pool is kept fixed.
        source = torch.randn(
            layers, max(batch, 1), drafts, h, v, k, dtype=dtype, device=device
        )
        storage_shape = (
            (layers, pool, h, k, v) if transposed else (layers, pool, h, v, k)
        )
        src_indices = torch.arange(batch, dtype=torch.int32, device=device)
    else:
        window, channels = shape["window"], shape["channels"]
        storage_shape = (
            (layers, pool, channels, window)
            if transposed
            else (layers, pool, window, channels)
        )
        source = None

    initial = torch.randn(storage_shape, dtype=dtype, device=device)
    expected = initial.clone()
    buffers = {name: initial.clone() for name in ("old", "new")}

    def view(storage):
        return storage.transpose(-1, -2) if transposed else storage

    # Independent torch reference. Clone rollback slices to avoid overlap.
    reference = view(expected)
    for src, (dst, step) in enumerate(zip(indices, steps)):
        if step < 0:
            continue
        if op == "move":
            reference[:, dst] = source[:, src, step]
        else:
            shift = drafts - 1 - step
            if shift > 0:
                reference[:, dst, shift:] = reference[:, dst, :-shift].clone()

    def execute(module, name):
        destination = view(buffers[name])
        if op == "move":
            result = module.move_intermediate_cache_kda(
                destination, source, indices_t, src_indices, steps_t
            )
        else:
            result = module.conv_state_rollback(destination, indices_t, steps_t, drafts)
        return result

    def reset(name):
        buffers[name].copy_(initial)

    return dict(
        execute=execute,
        reset=reset,
        buffers=buffers,
        expected=expected,
        view=view,
        op=op,
        batch=batch,
        shape=shape,
        pool=pool,
        dtype=str(dtype),
        transposed=transposed,
        pattern="mixed" if mixed else "all_step_zero",
    )


def check_accuracy(data, baseline, optimized):
    for name, module in (("old", baseline), ("new", optimized)):
        data["reset"](name)
        result = data["execute"](module, name)
        destination = data["view"](data["buffers"][name])
        if data["op"] == "move" or data["batch"]:
            assert result.data_ptr() == destination.data_ptr()
            assert result.stride() == destination.stride()
        else:
            assert result is None  # Existing empty-rollback return contract.
        # Bound assertion temporaries while still checking every pool element,
        # including all unselected slots. Whole-cache assert_close can OOM.
        for actual_layer, expected_layer in zip(
            data["buffers"][name], data["expected"]
        ):
            for start in range(0, data["pool"], 8):
                torch.testing.assert_close(
                    actual_layer[start : start + 8],
                    expected_layer[start : start + 8],
                    rtol=0,
                    atol=0,
                )


def check_performance(data, baseline, optimized):
    """NPU event elapsed time per wrapper call, in us; not profiler task duration.

    Reset copies, reference calculations, JIT and warmups are excluded. Event
    intervals can still include device idle time caused by host enqueue gaps.
    Repeated rollbacks always start from the same state, outside the timed range.
    """
    modules = {"old": baseline, "new": optimized}
    for _ in range(WARMUP):
        for name in ("old", "new"):
            data["reset"](name)
            data["execute"](modules[name], name)
    torch.npu.synchronize()
    samples = {"old": [], "new": []}
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for i in range(REPEAT):
        for name in ("old", "new") if i % 2 == 0 else ("new", "old"):
            data["reset"](name)
            torch.npu.synchronize()
            begin.record()
            data["execute"](modules[name], name)
            end.record()
            end.synchronize()
            elapsed_us = begin.elapsed_time(end) * 1000.0
            if elapsed_us <= 0:
                raise RuntimeError(
                    "Nonpositive event duration; cannot report a speedup"
                )
            samples[name].append(elapsed_us)
    return samples


def run_correctness(device, baseline, optimized):
    count = 0
    for op, (layers, batch), transposed, dtype in correctness_cases():
        shape = dict(
            layers=layers,
            drafts=4,
            heads=2,
            value_dim=65,
            key_dim=19,
            window=6,
            channels=32,
        )
        data = generate_data(op, batch, shape, device, dtype, transposed, mixed=True)
        check_accuracy(data, baseline, optimized)
        count += 1
        del data
    print(
        f"Correctness: {count} cases passed for BOTH old and new kernels.", flush=True
    )


def write_environment(device, optimized):
    import torch_npu

    def version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not reported"

    source = Path(optimized.__file__)
    baseline = Path(__file__).with_name("baseline.py")
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        chip=torch.npu.get_device_name(torch.npu.current_device()),
        device=device,
        logical_device_index=torch.npu.current_device(),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        triton_ascend=version("triton-ascend"),
        cann_version=os.environ.get("CANN_VERSION", "not reported"),
        baseline_commit=BASELINE_COMMIT,
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        optimized_source=str(source),
        optimized_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metric="npu_event_elapsed_us_per_wrapper_call",
        statistic="median",
        warmup=WARMUP,
        samples=REPEAT,
        shape_source="user-supplied Kimi3.1 config; fixed TP=16, draft=4 test shard",
        model_type="kimi_k31",
        model_architecture="KimiK31ForConditionalGeneration",
        test_tp=TEST_TP,
        draft_tokens=DRAFT_TOKENS,
        model_layers=KDA_LAYERS,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=CONV_KERNEL,
    )
    (RESULTS / "environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    device, baseline, optimized = load_operators()
    print(f"Device: {device}; optimized code: {optimized.__file__}", flush=True)
    run_correctness(device, baseline, optimized)
    if device == "cpu":
        print("CPU interpreter: accuracy only; no NPU performance measurements.")
        return
    # Check real Kimi dimensions with mixed accepted/rejected/fully accepted
    # steps before timing the all-step-zero (maximum rollback) workload.
    for op, shape, dtype, transposed in MODEL_CASES:
        data = generate_data(op, 8, shape, device, dtype, transposed, mixed=True)
        check_accuracy(data, baseline, optimized)
        del data
    print("Kimi3.1 mixed-step correctness: both operators passed.", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_environment(device, optimized)
    fields = [
        "operator",
        "batch",
        "shape",
        "pool",
        "dtype",
        "layout",
        "steps",
        "accuracy",
        "old_us",
        "new_us",
        "speedup",
        "old_min_us",
        "old_max_us",
        "new_min_us",
        "new_max_us",
    ]
    with (
        (RESULTS / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as summary_file,
        (RESULTS / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as samples_file,
    ):
        summary = csv.DictWriter(summary_file, fieldnames=fields)
        raw = csv.writer(samples_file)
        summary.writeheader()
        raw.writerow(["operator", "batch", "round", "old_us", "new_us", "first"])
        print("operator   batch      old_us      new_us    speedup", flush=True)
        for op, shape, dtype, transposed in MODEL_CASES:
            for batch in BATCH_SIZES:
                print(f"Checking {op}, batch={batch}, shape={shape}", flush=True)
                data = generate_data(
                    op, batch, shape, device, dtype, transposed, pool=POOL_SIZE
                )
                check_accuracy(data, baseline, optimized)
                samples = check_performance(data, baseline, optimized)
                old, new = (statistics.median(samples[name]) for name in ("old", "new"))
                summary.writerow(
                    dict(
                        operator=op,
                        batch=batch,
                        shape=json.dumps(shape, sort_keys=True),
                        pool=POOL_SIZE,
                        dtype=str(dtype),
                        layout="transposed" if transposed else "contiguous",
                        steps=data["pattern"],
                        accuracy="passed",
                        old_us=old,
                        new_us=new,
                        speedup=old / new,
                        old_min_us=min(samples["old"]),
                        old_max_us=max(samples["old"]),
                        new_min_us=min(samples["new"]),
                        new_max_us=max(samples["new"]),
                    )
                )
                for i, (old_sample, new_sample) in enumerate(
                    zip(samples["old"], samples["new"])
                ):
                    raw.writerow(
                        [
                            op,
                            batch,
                            i,
                            old_sample,
                            new_sample,
                            "old" if i % 2 == 0 else "new",
                        ]
                    )
                summary_file.flush()
                samples_file.flush()
                print(
                    f"{op:8s} {batch:6d} {old:11.3f} {new:11.3f} {old / new:9.3f}x",
                    flush=True,
                )
                del data
    print(f"Results: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
