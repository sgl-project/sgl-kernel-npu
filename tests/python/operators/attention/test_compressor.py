"""Public compressor tests; independent state buffers and valid-row comparison."""

import pytest

torch = pytest.importorskip("torch")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# DeepSeek-V4 config defaults: H=4096, index D=128, attention D=512.
# SGLang ascend_dsv4_backend.forward_compress uses TH, r4/coff2 or r128/coff1,
# and cache_mode=2 with explicit state rows. B/S are example serving loads,
# not captured traces. The remaining small widths isolate legacy regressions.
perf_shape = [(2, 8, 4096, 512), (1, 129, 4096, 512)]
# Keep the legacy output tolerances with the shared accuracy checker.
OUTPUT_STANDARD = {
    "torch.float16": AccuracyStandard(atol=0.05, rtol=0),
    "torch.bfloat16": AccuracyStandard(atol=0.05, rtol=0),
}
# Keep the legacy FP32 state tolerance.
STATE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-2, rtol=0)}


def _make_explicit_state_table(starts, capacity, block_size, ratio, coff):
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    history = coff * ratio
    arch35 = get_hardware().generation == Generation.ARCH35
    # A5 C4 uses one ring per SWA page. Other A5 cases use a request ring.
    # Keep arch22's entire history/current interval free of aliases.
    page_size = 128 if arch35 else block_size
    banks_per_request = (
        1
        if arch35 and (ratio, coff) != (4, 2)
        else max(2, (history + max(capacity, 1) + page_size - 1) // page_size)
    )
    dummy_bank = len(starts) * banks_per_request
    table = torch.full(
        (len(starts), history + max(capacity, 1)),
        dummy_bank * block_size,
        dtype=torch.int32,
        device="cpu",
    )
    for request, start in enumerate(starts):
        for column in range(table.shape[1]):
            position = start - history + column
            if position >= 0:
                bank = (
                    request * banks_per_request
                    + (position // page_size) % banks_per_request
                )
                table[request, column] = bank * block_size + position % block_size
    return table, dummy_bank + 1


def generate_data(shape, layout, ratio, coff, cache_mode, start, ring, defaults, dtype):
    b, s, h, d = shape
    w = coff * d
    # Use an explicit CPU generator; conftest selects the NPU default device.
    gen = torch.Generator(device="cpu").manual_seed(20260813)

    def random(size, scale=0.02):
        return torch.randn(size, generator=gen, device="cpu") * scale

    starts = [start] * b
    used = [s] * b
    if not defaults and b > 1:
        used[-1] = max(s - 1, 0)  # Include a partial final request.
    if cache_mode == 1:
        block = max(1, (start + s + ring - 1) // ring)
        table = torch.arange(1, b * block + 1, dtype=torch.int32, device="cpu").reshape(
            b, block
        )
        banks = 1 + b * block  # Bank 0 is a protected dummy page.
    else:
        table, banks = _make_explicit_state_table(starts, s, ring, ratio, coff)
    sr = min(b * s, b * s // ratio + b) if layout == "TH" else (s + ratio - 1) // ratio
    rope_shape = (sr, 64) if layout == "TH" else (b, sr, 64)
    # Distinct RoPE rows expose packed-vs-padded BSH indexing mistakes.
    sine = random(rope_shape, 0.1)
    cosine = 1 + random(rope_shape, 0.1)
    kwargs = dict(
        x=random((b * s, h) if layout == "TH" else (b, s, h)).to(dtype),
        wkv=random((w, h)).to(dtype),
        wgate=random((w, h)).to(dtype),
        state_cache=random((banks, ring, 2 * w), 0.01),
        ape=random((ratio, w), 0.01),
        norm_weight=1 + random((d,)),
        rope_sin=sine,
        rope_cos=cosine,
        state_block_table=table,
        cu_seqlens=(
            torch.arange(b + 1, device="cpu", dtype=torch.int32) * s
            if layout == "TH"
            else None
        ),
        seqused=(
            None if defaults else torch.tensor(used, device="cpu", dtype=torch.int32)
        ),
        start_pos=(
            None if defaults else torch.tensor(starts, device="cpu", dtype=torch.int32)
        ),
        rope_head_dim=64,
        cmp_ratio=ratio,
        coff=coff,
        rotary_mode=2,
        cache_mode=cache_mode,
    )
    device = torch.device("npu", torch.npu.current_device())
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }


def clone_inputs(data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def valid_mask(data, output):
    merged = data["x"].ndim == 2
    b = data["cu_seqlens"].numel() - 1 if merged else data["x"].shape[0]
    starts = (
        data["start_pos"].cpu().tolist() if data["start_pos"] is not None else [0] * b
    )
    if data["seqused"] is not None:
        used = data["seqused"].cpu().tolist()
    elif merged:
        cu = data["cu_seqlens"].cpu().tolist()
        used = [cu[i + 1] - cu[i] for i in range(b)]
    else:
        used = [data["x"].shape[1]] * b
    counts = [
        (p + n) // data["cmp_ratio"] - p // data["cmp_ratio"]
        for p, n in zip(starts, used)
    ]
    mask = torch.zeros(output.shape[:-1], device=output.device, dtype=torch.bool)
    if merged:
        mask[: sum(counts)] = True
    else:
        for request, count in enumerate(counts):
            mask[request, :count] = True
    return mask


class ResetExecutor:
    """Reset mutable state per call; its device copy is included in profiling."""

    def __init__(self, executor, state, initial):
        self.executor = executor
        self.state = state
        self.initial = initial
        self.func = executor.func
        self.impl = executor.impl
        self.impl_ref = executor.impl_ref
        self.accuracy_checked = executor.accuracy_checked

    def __call__(self):
        self.state.copy_(self.initial)
        return self.executor()


@pytest.mark.parametrize(
    "shape,layout,ratio,coff,cache_mode,start,ring,rounds,defaults",
    [
        pytest.param(
            (2, 8, 4096, 512),
            "TH",
            4,
            2,
            2,
            8,
            8,
            1,
            False,
            id="dsv4-attention-c4-serving",
        ),
        pytest.param(
            (1, 129, 4096, 512),
            "TH",
            128,
            1,
            2,
            200,
            256,
            1,
            False,
            id="dsv4-attention-c128-prefill",
        ),
        pytest.param(
            (2, 8, 4096, 128),
            "TH",
            4,
            2,
            2,
            8,
            8,
            1,
            False,
            id="dsv4-indexer-c4-serving",
        ),
        pytest.param(
            (2, 4, 1024, 128),
            "BSH",
            4,
            2,
            1,
            1,
            16,
            1,
            False,
            id="bsh-short-full-load-distinct-rope",
        ),
        pytest.param(
            (2, 9, 1024, 512),
            "BSH",
            4,
            1,
            1,
            13,
            16,
            1,
            False,
            id="bsh-normal-page-crossing",
        ),
        pytest.param(
            (1, 1, 1024, 128),
            "TH",
            4,
            2,
            2,
            0,
            8,
            1,
            False,
            id="edge-one-token-no-complete-group",
        ),
        pytest.param(
            (2, 8, 1024, 512),
            "TH",
            4,
            2,
            2,
            8,
            16,  # Partial acceptance needs the legacy MTP retention margin.
            4,
            False,
            id="ring-four-rounds-writeback",
        ),
        pytest.param(
            (1, 4, 1024, 128),
            "TH",
            2,
            1,
            1,
            0,
            16,
            1,
            True,
            id="default-start-used-length-and-rotary-mode",
        ),
        pytest.param(
            (1, 0, 1024, 128), "TH", 4, 1, 2, 0, 8, 1, False, id="edge-empty-th"
        ),
        pytest.param(
            (1, 32, 1024, 128),
            "TH",
            4,
            2,
            2,
            8,
            16,
            1,
            False,
            id="ring-mtp-keep-inside-chunk",
        ),
        pytest.param(
            (1, 4, 1024, 128),
            "BSH",
            4,
            1,
            2,
            8,
            16,
            1,
            False,
            id="bsh-full-load-nonoverlap-state",
        ),
        pytest.param(
            (0, 4, 1024, 128), "BSH", 4, 1, 1, 0, 16, 1, False, id="edge-empty-batch"
        ),
        pytest.param(
            (1, 32, 1024, 512),
            "TH",
            4,
            2,
            2,
            8,
            8,
            1,
            False,
            id="ring-cross-slice-long-sequence",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compressor(
    shape, layout, ratio, coff, cache_mode, start, ring, rounds, defaults, dtype
):
    from sgl_kernel_npu.operators.attention import compressor

    data = generate_data(
        shape, layout, ratio, coff, cache_mode, start, ring, defaults, dtype
    )
    if defaults:
        data.pop("rotary_mode")
    perf_data = clone_inputs(data)
    executor = make_executor(compressor, kwargs=perf_data)
    if get_skip_acc_check():
        require_test_impl(compressor, executor.impl)
    else:
        require_distinct_impls(compressor, executor.impl, executor.impl_ref)
        actual, expected = clone_inputs(data), clone_inputs(data)
        expected["rotary_mode"] = 2
        # Compare the omitted default against explicit mode 2 on the same backend.
        explicit = clone_inputs(expected) if defaults else None
        actual_ptr = actual["state_cache"].data_ptr()
        for step in range(rounds):
            # A partially accepted verification round can move start backwards.
            position = start + step * max(shape[1] - 1, 1)
            if actual["start_pos"] is not None:
                actual["start_pos"].fill_(position)
                expected["start_pos"].fill_(position)
            if cache_mode == 2:
                table = actual["state_block_table"]
                locations, banks = _make_explicit_state_table(
                    [position] * shape[0], shape[1], ring, ratio, coff
                )
                assert banks == actual["state_cache"].shape[0]
                table.copy_(locations.to(table.device))
                expected["state_block_table"].copy_(table)
            before = clone_inputs(actual)
            out = compressor(**actual, impl=executor.impl)
            ref = compressor(**expected, impl=executor.impl_ref)
            mask = valid_mask(expected, ref)
            assert_accuracy(out[mask], ref[mask], standard=OUTPUT_STANDARD)
            assert_accuracy(
                actual["state_cache"], expected["state_cache"], standard=STATE_STANDARD
            )
            if explicit is not None:
                explicit_out = compressor(**explicit, impl=executor.impl)
                assert_accuracy(out[mask], explicit_out[mask], standard=OUTPUT_STANDARD)
                assert_accuracy(
                    actual["state_cache"],
                    explicit["state_cache"],
                    standard=STATE_STANDARD,
                )
            assert actual["state_cache"].data_ptr() == actual_ptr
            untouched = expected["state_cache"] == before["state_cache"]
            assert torch.equal(
                actual["state_cache"][untouched], before["state_cache"][untouched]
            )
            for name, tensor in before.items():
                if isinstance(tensor, torch.Tensor) and name != "state_cache":
                    assert torch.equal(actual[name], tensor), name
        executor.accuracy_checked = True
    initial = perf_data["state_cache"].clone()
    reset_executor = ResetExecutor(executor, perf_data["state_cache"], initial)
    check_performance(
        reset_executor, enable_perf=shape in perf_shape and dtype == torch.bfloat16
    )


@pytest.mark.parametrize("ring", [8, 16])
def test_compressor_arch35_c4_resume_from_swa_page_tail(ring):
    from sgl_kernel_npu.operators.attention import compressor
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("A5 C4 SWA page-tail retention")
    if get_skip_acc_check():
        pytest.skip("Requires comparing full and resumed prefill outputs")

    prefix_tokens = 128
    total_tokens = 148
    suffix_tokens = total_tokens - prefix_tokens
    data = generate_data(
        (1, total_tokens, 1024, 128), "TH", 4, 2, 2, 0, ring, False, torch.bfloat16
    )
    executor = make_executor(compressor, kwargs=data)
    require_test_impl(compressor, executor.impl)
    full = executor()

    # The matched prefix's page is in the middle of the full prefill. Resume
    # using the state it left behind, with the same suffix inputs and RoPE rows.
    table, _ = _make_explicit_state_table([prefix_tokens], suffix_tokens, ring, 4, 2)
    suffix = dict(
        data,
        x=data["x"][prefix_tokens:],
        state_cache=data["state_cache"].clone(),
        state_block_table=table.to(data["x"].device),
        cu_seqlens=data["cu_seqlens"].new_tensor([0, suffix_tokens]),
        seqused=data["seqused"].new_tensor([suffix_tokens]),
        start_pos=data["start_pos"].new_tensor([prefix_tokens]),
        rope_sin=data["rope_sin"][prefix_tokens // 4 :],
        rope_cos=data["rope_cos"][prefix_tokens // 4 :],
    )
    resumed = compressor(**suffix, impl=executor.impl)
    assert_accuracy(
        resumed[valid_mask(suffix, resumed)],
        full[valid_mask(data, full)][prefix_tokens // 4 :],
        standard=OUTPUT_STANDARD,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("execution", ["nondefault-stream", "graph-replay"])
def test_compressor_async_execution(execution):
    from sgl_kernel_npu.operators.attention import compressor

    data = generate_data(
        (1, 9, 1024, 128), "TH", 4, 2, 2, 13, 16, False, torch.bfloat16
    )
    actual, expected = clone_inputs(data), clone_inputs(data)
    executor = make_executor(compressor, kwargs=actual)
    if get_skip_acc_check():
        require_test_impl(compressor, executor.impl)
    else:
        require_distinct_impls(compressor, executor.impl, executor.impl_ref)
    if executor.impl not in ("auto", "ascendc"):
        pytest.skip("Stream and graph checks target the native NPU implementation")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    if execution == "nondefault-stream":
        with torch.npu.stream(stream):
            actual["x"].mul_(0.75)
            out = executor()
        torch.npu.current_stream().wait_stream(stream)
        expected["x"].mul_(0.75)
    else:
        # Warm up allocations and ACLNN symbols outside capture.
        with torch.npu.stream(stream):
            executor()
            actual["state_cache"].copy_(data["state_cache"])
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            # Each recorded launch starts with the same history so the final
            # replay result remains comparable to one reference invocation.
            for _ in range(32):
                actual["state_cache"].copy_(data["state_cache"])
                out = executor()
        # Match the failing service phase: cleanup before the first replay.
        torch.npu.empty_cache()
        torch.npu.synchronize()
        actual["state_cache"].copy_(data["state_cache"])
        actual["x"].mul_(0.75)
        expected["x"].mul_(0.75)
        torch.npu.synchronize()
        graph.replay()
    torch.npu.synchronize()
    if not get_skip_acc_check():
        ref = compressor(**expected, impl=executor.impl_ref)
        mask = valid_mask(expected, ref)
        assert_accuracy(out[mask], ref[mask], standard=OUTPUT_STANDARD)
        assert_accuracy(
            actual["state_cache"], expected["state_cache"], standard=STATE_STANDARD
        )


@pytest.mark.extra_checks
def test_compressor_arch22_single_page_table():
    from sgl_kernel_npu.operators.attention import compressor
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH22:
        pytest.skip("The mode-1 single-page shorthand is supported on arch22")
    data = generate_data(
        (2, 8, 1024, 128),
        "TH",
        4,
        2,
        1,
        8,
        16,
        False,
        torch.bfloat16,
    )
    actual, expected = clone_inputs(data), clone_inputs(data)
    actual["state_block_table"] = actual["state_block_table"].view(-1)
    executor = make_executor(compressor, kwargs=actual)
    if get_skip_acc_check():
        require_test_impl(compressor, executor.impl)
    else:
        require_distinct_impls(compressor, executor.impl, executor.impl_ref)
    if not get_skip_acc_check():
        out = compressor(**actual, impl=executor.impl)
        ref = compressor(**expected, impl=executor.impl_ref)
        mask = valid_mask(expected, ref)
        assert_accuracy(out[mask], ref[mask], standard=OUTPUT_STANDARD)
        assert_accuracy(
            actual["state_cache"], expected["state_cache"], standard=STATE_STANDARD
        )


# DSV4 model-shaped and streaming regressions retained from PR #939.
VARIANTS = [(4, 2, 128), (4, 2, 512), (128, 1, 512)]
dsv4_perf_shape = [(128, 4096, 4, 2, 512), (128, 4096, 4, 2, 128)]


def parameters(ratio, coff, dim):
    torch.manual_seed(939)
    return (
        torch.randn(coff * dim, 4096, dtype=torch.bfloat16) * 0.02,
        torch.randn(coff * dim, 4096, dtype=torch.bfloat16) * 0.02,
        torch.randn(ratio, coff * dim, dtype=torch.float32) * 0.01,
        torch.ones(dim, dtype=torch.float32),
    )


def state_inputs(batch, ratio, coff, dim, cache_mode, total):
    ring = 16 if cache_mode == 1 else max(16, 2 * coff * ratio)
    pages = (total + ring - 1) // ring if cache_mode == 1 else 1
    state = torch.zeros(1 + batch * pages, ring, 2 * coff * dim, dtype=torch.float32)
    state[0].fill_(0.03125)  # Reserved guard page must not be overwritten.
    table = torch.arange(1, 1 + batch * pages, dtype=torch.int32).reshape(batch, pages)
    return state, table


def step_inputs(params, state, table, starts, counts, variant, mode, layout):
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    ratio, coff, dim = variant
    batch, capacity = len(starts), max(counts)
    wkv, wgate, ape, norm = params
    if layout == "TH":
        x = torch.randn(sum(counts), 4096, dtype=torch.bfloat16) * 0.1
        cu = torch.tensor([0, *counts], dtype=torch.int32).cumsum(0).to(torch.int32)
        out_shape = (min(sum(counts), sum(counts) // ratio + batch), 64)
    else:
        x = torch.randn(batch, capacity, 4096, dtype=torch.bfloat16) * 0.1
        cu = None
        out_shape = (batch, (capacity + ratio - 1) // ratio, 64)
    if mode == 1:
        locations = table
    elif get_hardware().generation == Generation.ARCH35:
        locations = table[:, 0].contiguous()
    else:
        history = coff * ratio
        locations = torch.zeros(batch, history + capacity, dtype=torch.int32)
        for b, start in enumerate(starts):
            for col in range(history + capacity):
                position = start - history + col
                if position >= 0:
                    locations[b, col] = (
                        table[b, 0] * state.shape[1] + position % state.shape[1]
                    )
    kwargs = dict(
        rope_sin=torch.full(out_shape, 0.1, dtype=torch.float32),
        rope_cos=torch.full(out_shape, 0.995, dtype=torch.float32),
        rope_head_dim=64,
        cmp_ratio=ratio,
        coff=coff,
        state_block_table=locations,
        cu_seqlens=cu,
        seqused=torch.tensor(counts, dtype=torch.int32),
        start_pos=torch.tensor(starts, dtype=torch.int32),
        cache_mode=mode,
        rotary_mode=2,
    )
    valid = [(s + n) // ratio - s // ratio for s, n in zip(starts, counts)]

    def valid_rows(output):
        # Native output capacity contains unspecified padding after valid rows.
        if layout == "TH":
            return output[: sum(valid)]
        return torch.cat([output[b, :n] for b, n in enumerate(valid)], dim=0)

    return (x, wkv, wgate, state, ape, norm), kwargs, valid_rows


def assert_state_accuracy(
    state, reference_state, starts, counts, variant, mode, locations
):
    standard = {"torch.float32": AccuracyStandard(atol=0.01, rtol=0.01)}
    if mode != 2 or locations.ndim != 1:
        # A3 explicit locations and paged writes retain their full-state checks.
        assert_accuracy(state, reference_state, standard=standard)
        return

    # A5 cycle backends can stop saving rows already consumed by compression.
    # Only the preceding overlap block and unfinished block are forward-live.
    # Extra rollback retention is backend-specific and is not asserted here.
    ratio, coff, _ = variant
    banks = locations.tolist()
    ring = state.shape[1]
    live = []
    for bank, start, count in zip(banks, starts, counts):
        end = start + count
        boundary = end // ratio * ratio
        keep = max(0, boundary - (coff - 1) * ratio)
        live.extend(bank * ring + position % ring for position in range(keep, end))
    if live:
        indices = torch.tensor(live, dtype=torch.int64, device=state.device)
        assert_accuracy(
            state.reshape(-1, state.shape[-1])[indices],
            reference_state.reshape(-1, reference_state.shape[-1])[indices],
            standard=standard,
        )
    # Every bank not assigned to a request must remain exactly unchanged.
    protected = [bank for bank in range(state.shape[0]) if bank not in banks]
    if protected:
        assert_accuracy(
            state[protected],
            reference_state[protected],
            standard={"torch.float32": AccuracyStandard(atol=0, rtol=0)},
        )


def check_step(
    params, state, reference_state, table, starts, counts, variant, mode, layout
):
    args, kwargs, transform = step_inputs(
        params, state, table, starts, counts, variant, mode, layout
    )
    return check_compressor_accuracy(
        args, kwargs, transform, reference_state, starts, counts, variant, mode
    )


def check_compressor_accuracy(
    args, kwargs, transform, reference_state, starts, counts, variant, mode
):
    from sgl_kernel_npu.operators.attention import compressor

    state = args[3]
    reference_args = (*args[:3], reference_state, *args[4:])
    executor = make_executor(compressor, args=args, kwargs=kwargs)
    if get_skip_acc_check() is True:
        require_test_impl(compressor, executor.impl)
    else:
        require_distinct_impls(compressor, executor.impl, executor.impl_ref)
        expected = compressor(*reference_args, **kwargs, impl=executor.impl_ref)
        actual = executor()
        assert_accuracy(
            transform(actual),
            transform(expected),
            standard={"torch.bfloat16": AccuracyStandard(atol=0.05, rtol=0.05)},
        )
        assert_state_accuracy(
            state,
            reference_state,
            starts,
            counts,
            variant,
            mode,
            kwargs["state_block_table"],
        )
        executor.accuracy_checked = True
    return executor


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "variant", VARIANTS, ids=["c4-indexer", "c4-attention", "c128-attention"]
)
@pytest.mark.parametrize("mode", [1, 2], ids=["paged", "cycle"])
@pytest.mark.parametrize("layout", ["TH"])
@pytest.mark.parametrize("batch", [1, 2])
def test_dsv4_compressor(variant, mode, layout, batch):
    ratio, coff, dim = variant
    counts = [ratio + 1] + [ratio] * (batch - 1)
    params = parameters(*variant)
    state, table = state_inputs(batch, ratio, coff, dim, mode, max(counts))
    executor = check_step(
        params, state, state.clone(), table, [0] * batch, counts, variant, mode, layout
    )
    # Starting at zero makes repeated prefill writes safe for profiling.
    check_performance(
        executor,
        enable_perf=batch == 2 and layout == "TH" and mode == 2,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "variant", VARIANTS, ids=["c4-indexer", "c4-attention", "c128-attention"]
)
@pytest.mark.parametrize("mode", [1, 2], ids=["paged", "cycle"])
def test_compressor_streaming(variant, mode):
    if get_skip_acc_check() is True:
        pytest.skip("Streaming state transitions require accuracy comparison")
    ratio, coff, dim = variant
    steps = [3, 4, 5, 4, 5] if ratio == 4 else [127, 3, 127, 3]
    params = parameters(*variant)
    state, table = state_inputs(2, ratio, coff, dim, mode, sum(steps))
    reference_state = state.clone()
    start = 0
    for count in steps:
        check_step(
            params,
            state,
            reference_state,
            table,
            [start] * 2,
            [count] * 2,
            variant,
            mode,
            "TH",
        )
        start += count
    # Replaying a streaming step mutates its history; do not benchmark it.


def generate_dsv4_data(variant):
    """DeepSeek-V4-Flash C4 attention/indexer prefill: X [128,4096]."""
    ratio, coff, dim = variant
    params = parameters(ratio, coff, dim)
    state, table = state_inputs(1, ratio, coff, dim, 2, 128)
    return step_inputs(params, state, table, [0], [128], variant, 2, "TH")


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param((4, 2, 512), id="deepseek-v4-flash-c4-attention-t128-h4096-d512"),
        pytest.param((4, 2, 128), id="deepseek-v4-flash-c4-indexer-t128-h4096-d128"),
    ],
)
def test_deepseek_v4_flash_compressor(variant):
    args, kwargs, transform = generate_dsv4_data(variant)
    executor = check_compressor_accuracy(
        args, kwargs, transform, args[3].clone(), [0], [128], variant, 2
    )
    check_performance(
        executor,
        enable_perf=(args[0].shape[0], args[0].shape[-1], *variant) in dsv4_perf_shape,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "tokens",
    [16, 18],
    ids=["synthetic-cycle-retention-unaligned", "synthetic-cycle-retention-aligned"],
)
def test_cycle_retention_boundary(tokens):
    # ring=16, window=8 gives write padding 10. With 16 tokens, the
    # retained suffix begins at token 6, inside a four-token group.
    variant = (4, 2, 128)
    params = parameters(*variant)
    state, table = state_inputs(1, *variant, 2, tokens)
    reference_state = state.clone()
    check_step(params, state, reference_state, table, [0], [tokens], variant, 2, "TH")
