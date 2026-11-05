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
PAGE_SIZE = 128

# Production decode batch: 16 requests x 8 tokens (Kimi K3-like shape).
DECODE_BATCH = 16


def _paged(length):
    """Return the block count for one batch of ``length`` tokens."""
    return (length + PAGE_SIZE - 1) // PAGE_SIZE


def _build_paged_case(
    cmp_ratio, cmp_length, cmp_topk, query_lengths, ori_length, index_pool
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
    "query_lengths,ori_length,cmp_ratio,cmp_length,cmp_topk,index_pool",
    [
        pytest.param((16, 8), 128, 1, 0, 0, 24, id="swa"),
        pytest.param((16, 8), 128, 2, 64, 8, 24, id="csa-ratio-2"),
        pytest.param((16, 8), 128, 4, 32, 8, 24, id="csa-ratio-4"),
        pytest.param((16, 8), 128, 128, 2, 0, 24, id="hca-ratio-128"),
        # Production decode shape: 16 requests x 8 tokens over 16008 tokens.
        pytest.param((8,) * DECODE_BATCH, 16008, 4, 4002, 8, 3900, id="decode-csa"),
    ],
)
def test_sparse_flash_mla_paged(
    query_lengths, ori_length, cmp_ratio, cmp_length, cmp_topk, index_pool
):
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    require_test_impl(sparse_flash_mla_metadata, "auto")
    q, inputs, _ = _build_paged_case(
        cmp_ratio, cmp_length, cmp_topk, query_lengths, ori_length, index_pool
    )
    metadata = _schedule(
        inputs, cmp_ratio, cmp_length, cmp_topk, query_lengths, ori_length
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
    ori_kv = (torch.randn((1, 128, NUM_HEADS_KV, HEAD_DIM)) * 0.1).to(torch.bfloat16)
    seqused_q = torch.full((1,), length, dtype=torch.int32)
    seqused_ori = torch.full((1,), 128, dtype=torch.int32)
    sinks = torch.randn((NUM_HEADS_Q,), dtype=torch.float32) * 0.1
    metadata = sparse_flash_mla_metadata(
        NUM_HEADS_Q,
        NUM_HEADS_KV,
        HEAD_DIM,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori,
        batch_size=1,
        max_seqlen_q=length,
        max_seqlen_ori_kv=128,
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
    check_performance(executor, enable_perf=True)
