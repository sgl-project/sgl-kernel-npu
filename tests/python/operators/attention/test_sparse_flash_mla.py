"""SWA, CSA and HCA coverage for arch22 sparse Flash MLA."""

import pytest

# Import the package before torch_npu initializes the CANN op store. Otherwise
# CANN's built-in SparseFlashMla shadows the bundled vendor and rejects the
# 2048-element metadata with "metadata dim 0 must be 1024".
try:
    import sgl_kernel_npu  # noqa: F401
except ImportError:  # pragma: no cover - the NPU runtime is optional here
    pass

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import require_test_impl
from tests.python.utils.performance import check_performance

HEAD_DIM = 512
NUM_HEADS_Q = 8
NUM_HEADS_KV = 1
ORI_KV_LENGTH = 128
PAGE_SIZE = 128
QUERY_LENGTHS = (16, 8)
INDEX_POOL = 24


def _paged(length):
    """Return the block count for one batch of ``length`` tokens."""
    return (length + PAGE_SIZE - 1) // PAGE_SIZE


def _build_paged_case(
    cmp_ratio,
    cmp_length,
    cmp_topk,
    *,
    query_lengths=QUERY_LENGTHS,
    ori_length=ORI_KV_LENGTH,
    index_pool=INDEX_POOL,
):
    """Create one TND/PA_BBND case; lengths are shared by every batch."""
    batch = len(query_lengths)
    total_q = sum(query_lengths)
    torch.npu.manual_seed(42)
    q = (torch.randn((total_q, NUM_HEADS_Q, HEAD_DIM)) * 0.1).to(torch.bfloat16)
    ori_pages = _paged(ori_length)
    ori_kv = (torch.randn((batch * ori_pages, PAGE_SIZE, 1, HEAD_DIM)) * 0.1).to(
        torch.bfloat16
    )
    ori_table = torch.arange(batch * ori_pages, dtype=torch.int32).reshape(
        batch, ori_pages
    )
    seqused_ori = torch.full((batch,), ori_length, dtype=torch.int32)
    bounds = [0]
    for value in query_lengths:
        bounds.append(bounds[-1] + value)
    cu_seqlens_q = torch.tensor(bounds, dtype=torch.int32)
    sinks = torch.randn((NUM_HEADS_Q,), dtype=torch.float32) * 0.1
    inputs = dict(
        ori_kv=ori_kv,
        ori_block_table=ori_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=seqused_ori,
        sinks=sinks,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if cmp_length else 0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    if not cmp_length:
        return q, inputs, 0
    cmp_pages = _paged(cmp_length)
    cmp_kv = (torch.randn((batch * cmp_pages, PAGE_SIZE, 1, HEAD_DIM)) * 0.1).to(
        torch.bfloat16
    )
    cmp_table = torch.arange(batch * cmp_pages, dtype=torch.int32).reshape(
        batch, cmp_pages
    )
    seqused_cmp = torch.full((batch,), cmp_length, dtype=torch.int32)
    inputs.update(
        cmp_kv=cmp_kv,
        cmp_block_table=cmp_table,
        seqused_cmp_kv=seqused_cmp,
    )
    if cmp_ratio != 1:
        inputs["cmp_residual_kv"] = torch.zeros((batch,), dtype=torch.int32)
    if cmp_topk:
        rows = []
        for _ in range(total_q):
            picked = torch.randperm(index_pool, dtype=torch.int64, device="cpu")
            rows.append(picked[:cmp_topk].sort().values)
        inputs["cmp_sparse_indices"] = (
            torch.stack(rows)
            .reshape(total_q, 1, cmp_topk)
            .to(device=q.device, dtype=torch.int32)
        )
    return q, inputs, 3


def _schedule(inputs, cmp_ratio, cmp_length, cmp_topk, query_lengths, ori_length):
    """Build the metadata that must match the main operator's arguments."""
    from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

    return sparse_flash_mla_metadata(
        NUM_HEADS_Q,
        NUM_HEADS_KV,
        HEAD_DIM,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        seqused_ori_kv=inputs["seqused_ori_kv"],
        seqused_cmp_kv=inputs.get("seqused_cmp_kv"),
        cmp_residual_kv=inputs.get("cmp_residual_kv"),
        max_seqlen_q=max(query_lengths),
        max_seqlen_ori_kv=ori_length,
        max_seqlen_cmp_kv=cmp_length,
        ori_topk=0,
        cmp_topk=cmp_topk,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if cmp_length else 0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=bool(cmp_length),
    )


@pytest.mark.parametrize(
    "cmp_ratio,cmp_length,cmp_topk",
    [
        pytest.param(1, 0, 0, id="swa"),
        pytest.param(2, 64, 8, id="csa-ratio-2"),
        pytest.param(4, 32, 8, id="csa-ratio-4"),
        pytest.param(128, 2, 0, id="hca-ratio-128"),
    ],
)
def test_sparse_flash_mla_paged(cmp_ratio, cmp_length, cmp_topk):
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    require_test_impl(sparse_flash_mla_metadata, "auto")
    q, inputs, _ = _build_paged_case(cmp_ratio, cmp_length, cmp_topk)
    total_q = q.shape[0]
    metadata = _schedule(
        inputs, cmp_ratio, cmp_length, cmp_topk, QUERY_LENGTHS, ORI_KV_LENGTH
    )
    assert metadata.dtype == torch.int32 and metadata.ndim == 1
    assert metadata.device == q.device
    executor = check_accuracy(
        sparse_flash_mla,
        args=(q,),
        kwargs=dict(
            **inputs,
            metadata=metadata,
            softmax_scale=HEAD_DIM**-0.5,
            return_softmax_lse=True,
        ),
    )
    out, lse = executor()
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (NUM_HEADS_KV, total_q, NUM_HEADS_Q // NUM_HEADS_KV)
    assert lse.dtype == torch.float32
    check_performance(executor, enable_perf=True)


# Kimi K3-like decode batch: 16 requests x 8 tokens over a 16008-token
# history with 4x compressed KV. Measures the production latency shape.
DECODE_BATCH = 16
DECODE_QUERY_LENGTH = 8
DECODE_ORI_LENGTH = 16008
DECODE_CMP_LENGTH = 4002
DECODE_CMP_TOPK = 8
DECODE_INDEX_POOL = 3900


def test_sparse_flash_mla_decode():
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    require_test_impl(sparse_flash_mla_metadata, "auto")
    query_lengths = (DECODE_QUERY_LENGTH,) * DECODE_BATCH
    q, inputs, _ = _build_paged_case(
        4,
        DECODE_CMP_LENGTH,
        DECODE_CMP_TOPK,
        query_lengths=query_lengths,
        ori_length=DECODE_ORI_LENGTH,
        index_pool=DECODE_INDEX_POOL,
    )
    total_q = q.shape[0]
    metadata = _schedule(
        inputs,
        4,
        DECODE_CMP_LENGTH,
        DECODE_CMP_TOPK,
        query_lengths,
        DECODE_ORI_LENGTH,
    )
    executor = check_accuracy(
        sparse_flash_mla,
        args=(q,),
        kwargs=dict(
            **inputs,
            metadata=metadata,
            softmax_scale=HEAD_DIM**-0.5,
            return_softmax_lse=True,
        ),
    )
    out, lse = executor()
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (NUM_HEADS_KV, total_q, NUM_HEADS_Q // NUM_HEADS_KV)
    check_performance(executor, enable_perf=True)


def test_sparse_flash_mla_bsnd_swa():
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    require_test_impl(sparse_flash_mla_metadata, "auto")
    torch.npu.manual_seed(42)
    length = 16
    q = (torch.randn((1, length, NUM_HEADS_Q, HEAD_DIM)) * 0.1).to(torch.bfloat16)
    ori_kv = (torch.randn((1, ORI_KV_LENGTH, NUM_HEADS_KV, HEAD_DIM)) * 0.1).to(
        torch.bfloat16
    )
    seqused_q = torch.full((1,), length, dtype=torch.int32)
    seqused_ori = torch.full((1,), ORI_KV_LENGTH, dtype=torch.int32)
    sinks = torch.randn((NUM_HEADS_Q,), dtype=torch.float32) * 0.1
    metadata = sparse_flash_mla_metadata(
        NUM_HEADS_Q,
        NUM_HEADS_KV,
        HEAD_DIM,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori,
        batch_size=1,
        max_seqlen_q=length,
        max_seqlen_ori_kv=ORI_KV_LENGTH,
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    executor = check_accuracy(
        sparse_flash_mla,
        args=(q,),
        kwargs=dict(
            ori_kv=ori_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=HEAD_DIM**-0.5,
            ori_mask_mode=4,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="BSND",
            return_softmax_lse=True,
        ),
    )
    out, lse = executor()
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (1, NUM_HEADS_KV, length, NUM_HEADS_Q // NUM_HEADS_KV)
    check_performance(executor, enable_perf=True)
