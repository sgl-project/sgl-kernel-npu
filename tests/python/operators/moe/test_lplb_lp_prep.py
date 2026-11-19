"""LPLB LP-input builder: in-place buffers, count clamps and index dtypes."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

# Production case: a 256-logical-expert EP group spread over 16 NPU cards,
# with 32 replicated logical experts and 16 GPUs in the IPM row block.
perf_shape = [(224, 32, 64, 16, 256)]


def generate_data(shape, mode, seed=42):
    """Build one consistent LPLB solver state.

    shape is (num_single, num_red_log, num_red_phy, num_gpus, num_logical).
    normal: random counts; every logical expert is either single or replicated,
        matching LPLBSolver's partition of log_single/log_replicated.
    zero_counts: all counts are zero, which exercises the max(total, 1) clamp.
    int32_ids: int32 index vectors instead of int64 (both are supported).
    """
    num_single, num_red_log, num_red_phy, num_gpus, num_logical = shape
    torch.npu.manual_seed(seed)
    nc = num_red_log + num_gpus
    nv = num_red_phy + num_gpus + 2  # ratios + GPUs + M + Big-M column

    A_full = torch.randn(nc, nv, dtype=torch.float32)
    b = torch.empty(nc, dtype=torch.float32)
    t1 = torch.empty(num_single, dtype=torch.float32)
    global_counts = torch.randint(0, 64, (num_logical,), dtype=torch.int32).float()
    if mode == "zero_counts":
        global_counts.zero_()
    index_dtype = torch.int32 if mode == "int32_ids" else torch.int64
    log_single = torch.arange(num_single, dtype=index_dtype)
    log_replicated = torch.arange(
        num_single, num_single + num_red_log, dtype=index_dtype
    )
    B1 = (torch.rand(num_gpus, num_single, dtype=torch.float32) > 0.5).float()
    A_base_row_sum = torch.randn(nc, dtype=torch.float32)
    return (
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((224, 32, 64, 16, 256), id="deepseek-v3-16npu-redundant"),
    ],
)
@pytest.mark.parametrize("mode", ["normal", "zero_counts", "int32_ids"])
def test_lp_prep(shape, mode):
    from sgl_kernel_npu.operators.moe import lp_prep

    args = generate_data(shape, mode)
    # In-place operator: the reference runs on cloned inputs so the Triton call
    # still sees the original state (check_accuracy cannot be used directly).
    executor = make_executor(lp_prep, args=args)
    if not get_skip_acc_check():
        impl, impl_ref = get_test_impls()
        require_distinct_impls(lp_prep, impl, impl_ref)
        expected = lp_prep(*(arg.clone() for arg in args), impl=impl_ref)
        actual = executor()
        assert_accuracy(actual, expected)
        executor.accuracy_checked = True

    # lp_prep only rewrites its output buffers, so repeated calls are
    # idempotent and the same executor can be profiled.
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and mode == "normal"),
    )
