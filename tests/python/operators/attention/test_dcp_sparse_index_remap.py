"""Exact rank-local remap and device timing through the public attention API."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import require_test_impl
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")

# GLM-5.2 Target Verify dimensions with four query rows and 128K context.
perf_shape = [(4, 4, 3, 128, "mixed")]


def generate_data(rows, dcp_size, dcp_rank, page_size, pattern):
    generator = torch.Generator(device="cpu").manual_seed(123)
    indices = torch.randint(
        0, 128_000, (rows, 2048), generator=generator, dtype=torch.int32, device="cpu"
    )
    if pattern == "mixed":
        indices[0, ::3] = -1
        indices[1, :4] = (
            torch.tensor([7, 3, 7, 5], device="cpu") * dcp_size + dcp_rank
        ) * page_size
    elif pattern == "empty_and_full":
        indices[0].fill_(-1)
        indices[1] = (
            torch.arange(2048, 0, -1, device="cpu") * dcp_size + dcp_rank
        ) * page_size
    return indices.npu()


@pytest.mark.parametrize(
    "rows,dcp_size,dcp_rank,page_size,pattern",
    [
        pytest.param(*perf_shape[0], id="glm52-verify-128k-dcp4-topk2048"),
        pytest.param(
            2, 2, 1, 1, "empty_and_full", id="dcp2-unit-pages-all-rejected-and-retained"
        ),
        pytest.param(0, 4, 0, 128, "empty", id="no-query-rows"),
    ],
)
def test_dcp_sparse_index_remap(rows, dcp_size, dcp_rank, page_size, pattern):
    from sgl_kernel_npu.operators.attention import dcp_sparse_index_remap

    indices = generate_data(rows, dcp_size, dcp_rank, page_size, pattern)
    executor = check_accuracy(
        dcp_sparse_index_remap, args=(indices, dcp_size, dcp_rank, page_size)
    )
    check_performance(
        executor,
        enable_perf=(rows, dcp_size, dcp_rank, page_size, pattern) in perf_shape,
    )


@pytest.mark.extra_checks
def test_dcp_sparse_index_remap_rejects_non_power_of_two_page():
    from sgl_kernel_npu.operators.attention import dcp_sparse_index_remap

    require_test_impl(dcp_sparse_index_remap, "ascendc")
    indices = torch.zeros((1, 2048), dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="power-of-two page size"):
        dcp_sparse_index_remap(indices, 4, 0, 3, impl="ascendc")
