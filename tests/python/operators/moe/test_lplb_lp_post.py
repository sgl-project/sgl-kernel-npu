"""LPLB probability-table builder: scatters, sink rows and window tails."""

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

# 256 logical experts with 2 physical copies on average, matching the DP-attention
# log2phy table of a DeepSeek-V3-class EP group.
perf_shape = [(256, 2, 224, 64)]


def generate_data(shape, mode, seed=42):
    """Build a physical-probability table and its gather map.

    shape is (num_logical, max_copies, num_single, num_red_phy).
    normal: random replica ratios, mixed positive/negative, with -1 padding.
    empty_row: the first logical row maps entirely to -1 (sink slot reads).
    int32_ids: int32 index vectors and table instead of int64.
    """
    num_logical, max_copies, num_single, num_red_phy = shape
    torch.npu.manual_seed(seed)
    phy_prob_size = num_single + num_red_phy + 1
    index_dtype = torch.int32 if mode == "int32_ids" else torch.int64

    log2phy_prob = torch.empty(
        num_logical, max_copies, dtype=torch.float32
    )
    x = torch.randn(num_red_phy, dtype=torch.float32) * 0.3
    t1 = torch.rand(num_single, dtype=torch.float32)
    phy_single = torch.arange(num_single, dtype=index_dtype)
    phy_replicated = torch.arange(
        num_single, num_single + num_red_phy, dtype=index_dtype
    )
    log2phy = torch.randint(
        0, phy_prob_size, (num_logical, max_copies), dtype=index_dtype
    )
    log2phy[torch.rand(num_logical, max_copies) < 0.25] = -1
    if mode == "empty_row":
        log2phy[0] = -1
    return log2phy_prob, x, t1, phy_single, phy_replicated, log2phy


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((256, 2, 224, 64), id="deepseek-v3-2-copies"),
        pytest.param((256, 4, 224, 576), id="four-copies-wide-table"),
        pytest.param((1, 1, 0, 1), id="edge-no-single-expert"),
        pytest.param((37, 3, 17, 9), id="edge-unaligned-table"),
    ],
)
@pytest.mark.parametrize("mode", ["normal", "empty_row", "int32_ids"])
def test_lp_post(shape, mode):
    from sgl_kernel_npu.operators.moe import lp_post

    args = generate_data(shape, mode)
    # In-place operator: the reference writes into a clone of the output buffer.
    reference_args = list(args)
    reference_args[0] = reference_args[0].clone()
    executor = make_executor(lp_post, args=args)
    if not get_skip_acc_check():
        impl, impl_ref = get_test_impls()
        require_distinct_impls(lp_post, impl, impl_ref)
        expected = lp_post(*reference_args, impl=impl_ref)
        actual = executor()
        assert_accuracy(actual, expected)
        assert actual.data_ptr() == args[0].data_ptr(), "lp_post must write in place"
        executor.accuracy_checked = True

    # The kernel overwrites the same buffer from unchanged inputs, so repeated
    # calls are idempotent and the executor can be profiled as-is.
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and mode == "normal"),
    )
