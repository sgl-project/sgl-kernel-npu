"""Check layer-parallel cache updates, including in-place and stride semantics."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# Production shapes monitored by default; the CLI may override performance.
# Entries are (operator, shape, batch, dtype, transposed).
perf_shape = [
    ("move", (69, 6, 128, 128), 8, torch.float32, True),
    ("conv", (69, 6, 2304), 8, torch.bfloat16, False),
]
PROFILE_WARMUP = 5
PROFILE_ACTIVE = 5


def generate_data(op, shape, batch, dtype, transposed, device, *, with_reference=True):
    torch.manual_seed(42)
    layers, *dims = shape
    pool, drafts = 2 * batch + 3, 4
    storage_shape = (layers, pool, *dims)
    if transposed:
        storage_shape = (*storage_shape[:-2], storage_shape[-1], storage_shape[-2])
    state = torch.randn(storage_shape, dtype=dtype, device=device)
    if transposed:
        state = state.transpose(-1, -2)
    slots = list(reversed(range(1, batch + 1)))
    steps = [(0, -1, 1, drafts - 1)[i % 4] for i in range(batch)]
    indices = torch.tensor(slots, dtype=torch.int32, device=device)
    step_indices = torch.tensor(steps, dtype=torch.int32, device=device)
    expected = state.clone() if with_reference else None
    if op == "move":
        source = torch.randn(
            layers, max(batch, 1), drafts, *dims, dtype=dtype, device=device
        )
        src_indices = torch.arange(batch, dtype=torch.int32, device=device)
        args = (source, indices, src_indices, step_indices)
    else:
        args = (indices, step_indices, drafts)

    if not with_reference:
        return state, args, None

    # Independent reference: skipped requests and unselected pool slots stay intact.
    for src, (dst, step) in enumerate(zip(slots, steps)):
        if step < 0:
            continue
        if op == "move":
            expected[:, dst] = source[:, src, step]
        else:
            shift = drafts - 1 - step
            if shift > 0:
                expected[:, dst, shift:] = expected[:, dst, :-shift].clone()
    return state, args, expected


def make_profile_executor(operator, state, args, accuracy_checked):
    executor = make_executor(
        operator, args=(state, *args), accuracy_checked=accuracy_checked
    )
    pending = None

    def run():
        nonlocal pending
        if pending is None:
            # benchmark_npu calls once before starting the profiler. Prepare
            # every input then, so neither allocation nor reset copies are timed.
            # Use a fresh state for each warmup/sample: rollback is not idempotent.
            pending = iter(
                [
                    make_executor(
                        operator, args=(state.clone(), *args), impl=executor.impl
                    )
                    for _ in range(1 + PROFILE_WARMUP + PROFILE_ACTIVE)
                ]
            )
        return next(pending)()

    run.func = executor.func
    run.impl = executor.impl
    run.impl_ref = executor.impl_ref
    run.accuracy_checked = executor.accuracy_checked
    return run


def check_case(op, shape, batch, dtype, transposed):
    from sgl_kernel_npu.operators.mamba import (
        conv_state_rollback,
        move_intermediate_cache_kda,
    )

    operator = move_intermediate_cache_kda if op == "move" else conv_state_rollback
    impl, impl_ref = get_test_impls()
    skip_accuracy = get_skip_acc_check()
    if skip_accuracy:
        require_test_impl(operator, impl)
    else:
        require_distinct_impls(operator, impl, impl_ref)
    state, args, expected = generate_data(
        op,
        shape,
        batch,
        dtype,
        transposed,
        f"npu:{torch.npu.current_device()}",
        with_reference=not skip_accuracy,
    )
    accuracy_checked = False
    if not skip_accuracy:
        for implementation in (impl_ref, impl):
            # Each implementation receives the same initial values and layout.
            actual = state.clone()
            strides = actual.stride()
            result = operator(actual, *args, impl=implementation)
            if op == "conv" and batch == 0:
                assert result is None
            else:
                assert result.data_ptr() == actual.data_ptr()
                assert result.stride() == strides
            assert actual.stride() == state.stride()
            # Bound assertion temporaries for model shapes; inspect the whole pool.
            for actual_layer, expected_layer in zip(actual, expected):
                for start in range(0, state.shape[1], 8):
                    torch.testing.assert_close(
                        actual_layer[start : start + 8],
                        expected_layer[start : start + 8],
                        atol=0,
                        rtol=0,
                    )
        accuracy_checked = True
    else:
        print(f"Accuracy check skipped: {operator.__name__} [{impl}]")
    # Empty selections launch no device work, even when profiling is forced on.
    if batch:
        executor = make_profile_executor(operator, state, args, accuracy_checked)
        check_performance(
            executor,
            enable_perf=(op, shape, batch, dtype, transposed) in perf_shape,
            warmup=PROFILE_WARMUP,
            active=PROFILE_ACTIVE,
        )


@pytest.mark.parametrize("op", ["move", "conv"])
@pytest.mark.parametrize(
    "layers,batch",
    [
        pytest.param(1, 1, id="single-layer"),
        pytest.param(4, 0, id="empty-batch"),
        pytest.param(4, 1, id="small-batch"),
        pytest.param(32, 1, id="many-layers-small-batch"),
        pytest.param(4, 8, id="mixed-steps"),
        pytest.param(4, 128, id="large-batch"),
    ],
)
@pytest.mark.parametrize("transposed", [False, True], ids=["contiguous", "transposed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_kda_cache_layer_parallel(op, layers, batch, transposed, dtype):
    # Unaligned V/K dimensions exercise partial tiles; batch 8 mixes all steps.
    shape = (layers, 2, 65, 19) if op == "move" else (layers, 6, 32)
    check_case(op, shape, batch, dtype, transposed)


@pytest.mark.parametrize(
    "op,shape,dtype,transposed",
    [
        pytest.param(
            "move", (69, 6, 128, 128), torch.float32, True, id="kimi-k3.1-move-fp32"
        ),
        pytest.param(
            "conv", (69, 6, 2304), torch.bfloat16, False, id="kimi-k3.1-conv-bf16"
        ),
    ],
)
def test_kimi31_cache_shape(op, shape, dtype, transposed):
    # Kimi K3.1: 69 KDA layers, 96 heads, head dim 128, conv kernel 4.
    # Test choices: one TP=16 shard (6 heads), four drafts, batch 8.
    # Conv window=(4-1)+(4-1)=6; channels=3*6*128=2304.
    check_case(op, shape, 8, dtype, transposed)


def test_legacy_entry_points_share_optimized_implementation():
    from sgl_kernel_npu.implementation.triton.mamba import mamba_state_update
    from sgl_kernel_npu.mamba import mamba_state_update_triton

    for name in (
        "move_cache_dynamic_last_kernel_h_block_kda",
        "move_intermediate_cache_kda",
        "_conv_state_rollback_kernel",
        "conv_state_rollback",
    ):
        assert getattr(mamba_state_update_triton, name) is getattr(
            mamba_state_update, name
        )
