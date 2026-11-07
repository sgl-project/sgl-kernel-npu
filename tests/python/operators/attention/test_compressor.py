"""Public compressor tests; independent state buffers and valid-row comparison."""

import pytest

torch = pytest.importorskip("torch")

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
# cache_mode=2, and architecture-specific tables. B/S are example serving loads,
# not captured traces. The remaining small widths isolate legacy regressions.
perf_shape = [(2, 8, 4096, 512), (1, 129, 4096, 512)]
# Keep the legacy output tolerances with the shared accuracy checker.
OUTPUT_STANDARD = {
    "torch.float16": AccuracyStandard(atol=0.05, rtol=0),
    "torch.bfloat16": AccuracyStandard(atol=0.05, rtol=0),
}
# Keep the legacy FP32 state tolerance.
STATE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-2, rtol=0)}


def _make_explicit_state_table(starts, capacity, block_size, history):
    # arch22 EXPLICIT is a physical-location table, not arch35's single-bank
    # ring ABI. The unchanged overlap kernel can save new rows before reading
    # history, so keep the entire history/current interval free of aliases.
    banks_per_request = max(
        2, (history + max(capacity, 1) + block_size - 1) // block_size
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
                    + (position // block_size) % banks_per_request
                )
                table[request, column] = bank * block_size + position % block_size
    return table, dummy_bank + 1


def generate_data(shape, layout, ratio, coff, cache_mode, start, ring, defaults, dtype):
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

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
    bank = cache_mode == 2 and get_hardware().generation == Generation.ARCH35
    if cache_mode == 1:
        block = max(1, (start + s + ring - 1) // ring)
        table = torch.arange(1, b * block + 1, dtype=torch.int32, device="cpu").reshape(
            b, block
        )
        banks = 1 + b * block  # Bank 0 is a protected dummy page.
    elif bank:
        table = torch.arange(b, dtype=torch.int32, device="cpu")
        banks = b + 1  # Extra untouched bank checks write bounds.
    else:
        table, banks = _make_explicit_state_table(starts, s, ring, coff * ratio)
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
            if cache_mode == 2 and actual["state_block_table"].ndim == 2:
                table = actual["state_block_table"]
                locations, banks = _make_explicit_state_table(
                    [position] * shape[0], shape[1], ring, coff * ratio
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
            out = executor()
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
@pytest.mark.parametrize(
    "table_form", ["arch22-single-page", "arch35-column-banks", "arch35-wide-banks"]
)
def test_compressor_legacy_table_forms(table_form):
    from sgl_kernel_npu.operators.attention import compressor
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    mode = 1 if table_form == "arch22-single-page" else 2
    data = generate_data(
        (2, 8, 1024, 128),
        "TH",
        4,
        2,
        mode,
        8,
        16 if mode == 1 else 8,
        False,
        torch.bfloat16,
    )
    actual, expected = clone_inputs(data), clone_inputs(data)
    executor = make_executor(compressor, kwargs=actual)
    if get_skip_acc_check():
        require_test_impl(compressor, executor.impl)
    else:
        require_distinct_impls(compressor, executor.impl, executor.impl_ref)
    target = Generation.ARCH22 if mode == 1 else Generation.ARCH35
    if get_hardware().generation != target:
        pytest.skip("This legacy table form belongs to a different cache ABI")
    table = actual["state_block_table"]
    if table_form == "arch22-single-page":
        actual["state_block_table"] = table.view(-1)
    elif table_form == "arch35-column-banks":
        actual["state_block_table"] = table.view(-1, 1)
    else:
        # Bank IDs are flat entries, not the first column of a 2D table.
        actual["state_block_table"] = torch.cat((table, torch.zeros_like(table))).view(
            2, 2
        )
    if not get_skip_acc_check():
        out = compressor(**actual, impl=executor.impl)
        ref = compressor(**expected, impl=executor.impl_ref)
        mask = valid_mask(expected, ref)
        assert_accuracy(out[mask], ref[mask], standard=OUTPUT_STANDARD)
        assert_accuracy(
            actual["state_cache"], expected["state_cache"], standard=STATE_STANDARD
        )
