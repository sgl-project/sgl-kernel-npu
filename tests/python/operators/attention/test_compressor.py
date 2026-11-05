"""Compressor output, live state and protected-bank accuracy, including streaming."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

VARIANTS = [(4, 2, 128), (4, 2, 512), (128, 1, 512)]
perf_shape = [(128, 4096, 4, 2, 512), (128, 4096, 4, 2, 128)]


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
def test_compressor(variant, mode, layout, batch):
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


def generate_data(variant):
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
    args, kwargs, transform = generate_data(variant)
    executor = check_compressor_accuracy(
        args, kwargs, transform, args[3].clone(), [0], [128], variant, 2
    )
    check_performance(
        executor,
        enable_perf=(args[0].shape[0], args[0].shape[-1], *variant) in perf_shape,
    )
