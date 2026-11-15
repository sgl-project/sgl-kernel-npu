"""Small smoke and Kimi K3 production inputs for arch35 Flash MLA."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import require_test_impl
from tests.python.utils.performance import check_performance


def generate_test_data(
    shape,
    cache_shape,
    table_shape,
    kv_length,
    query_length,
    dtype,
    mask_mode,
    use_seqused_q,
):
    """Create paged NPU data; conftest supplies the default device context."""
    torch.npu.manual_seed(42)
    q = torch.randn(shape, dtype=dtype) * 0.1
    cache = torch.randn(cache_shape, dtype=dtype) * 0.1
    batch_size, pages_per_sequence = table_shape
    table = torch.arange(batch_size * pages_per_sequence, dtype=torch.int32).reshape(
        table_shape
    )
    lengths = torch.full((batch_size,), kv_length, dtype=torch.int32)
    boundaries = torch.arange(0, shape[0] + 1, query_length, dtype=torch.int32)
    seqused_q = (
        torch.full((batch_size,), query_length, dtype=torch.int32)
        if use_seqused_q
        else None
    )
    attn_mask = (
        torch.ones((2048, 2048), dtype=torch.int8).triu(diagonal=1)
        if mask_mode == 3
        else None
    )
    return q, cache, table, lengths, boundaries, seqused_q, attn_mask


@pytest.mark.parametrize(
    "shape,cache_shape,table_shape,kv_length,query_length,mask_mode,"
    "use_seqused_q,softmax_scale,return_softmax_lse",
    [
        pytest.param(
            (1, 64, 576),
            (1, 128, 1, 576),
            (1, 1),
            128,
            1,
            0,
            False,
            576**-0.5,
            True,
            id="smoke",
        ),
        # Kimi K3: 16 requests x 8 query tokens, 126 pages/request.
        # The final KV page has only 8 valid tokens; mode 3 is right-aligned causal.
        pytest.param(
            (128, 24, 576),
            (2016, 128, 1, 576),
            (16, 126),
            16008,
            8,
            3,
            True,
            192**-0.5,
            False,
            id="kimi-k3",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flash_mla(
    shape,
    cache_shape,
    table_shape,
    kv_length,
    query_length,
    mask_mode,
    use_seqused_q,
    softmax_scale,
    return_softmax_lse,
    dtype,
):
    from sgl_kernel_npu.operators.attention import flash_mla, flash_mla_metadata

    require_test_impl(flash_mla_metadata, "auto")
    q, cache, table, lengths, boundaries, seqused_q, attn_mask = generate_test_data(
        shape,
        cache_shape,
        table_shape,
        kv_length,
        query_length,
        dtype,
        mask_mode,
        use_seqused_q,
    )
    metadata = flash_mla_metadata(
        lengths,
        shape[1],
        1,
        boundaries,
        seqused_q,
        -1,
        -1,
        shape[-1],
        512,
        mask_mode,
        -1,
        -1,
        "TND",
    )
    # The pinned upstream returns an opaque buffer sized by batch/core counts.
    assert metadata.dtype == torch.int32 and metadata.ndim == 1
    assert metadata.device == q.device
    executor = check_accuracy(
        flash_mla,
        args=(
            q,
            cache,
            table,
            lengths,
            boundaries,
            seqused_q,
            None,
            attn_mask,
            metadata,
            512,
            softmax_scale,
            mask_mode,
            -1,
            -1,
            -1,
            -1,
            "TND",
            "PA_BBND",
            "NTD",
            return_softmax_lse,
        ),
    )
    out, lse = executor()
    assert out.shape == (shape[1], shape[0], 512)
    assert out.dtype == dtype and out.device == q.device
    assert lse.shape == ((shape[1], shape[0]) if return_softmax_lse else (0,))
    check_performance(executor, enable_perf=True)
