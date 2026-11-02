"""One small arch35 smoke case for the bundled Flash MLA operator."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import require_test_impl
from tests.python.utils.performance import check_performance


def generate_test_data(shape, kv_length, dtype):
    """Single-request decode, one page; conftest supplies the NPU device context."""
    torch.npu.manual_seed(42)
    q = torch.randn(shape, dtype=dtype) * 0.1
    cache = torch.randn((1, 128, 1, shape[-1]), dtype=dtype) * 0.1
    table = torch.tensor([[0]], dtype=torch.int32)
    lengths = torch.tensor([kv_length], dtype=torch.int32)
    boundaries = torch.tensor([0, shape[0]], dtype=torch.int32)
    return q, cache, table, lengths, boundaries


@pytest.mark.parametrize("shape", [(1, 64, 576)])
@pytest.mark.parametrize("kv_length", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flash_mla_smoke(shape, kv_length, dtype):
    from sgl_kernel_npu.operators.attention import flash_mla, flash_mla_metadata

    require_test_impl(flash_mla_metadata, "auto")
    q, cache, table, lengths, boundaries = generate_test_data(shape, kv_length, dtype)
    metadata = flash_mla_metadata(
        lengths, shape[1], 1, cu_seqlens_q=boundaries, layout_q="TND"
    )
    assert metadata.dtype == torch.int32 and metadata.ndim == 1
    executor = check_accuracy(
        flash_mla,
        args=(q, cache),
        kwargs=dict(
            block_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=boundaries,
            metadata=metadata,
            softmax_scale=shape[-1] ** -0.5,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out="NTD",
            return_softmax_lse=True,
        ),
    )
    check_performance(executor, enable_perf=False)
