"""SWA, CSA and HCA coverage for arch22 sparse Flash MLA."""

import pytest

# Import the package before torch_npu initializes the CANN op store. Otherwise
# CANN may select its built-in SparseFlashMla instead of the bundled vendor.
# Matching metadata lengths alone do not identify the selected implementation.
import sgl_kernel_npu  # noqa: F401

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

HEAD_DIM = 512
NUM_HEADS_Q = 8
NUM_HEADS_KV = 1
PAGE_SIZE = 128

# Production decode batch: 16 requests x 8 tokens (Kimi K3-like shape).
DECODE_BATCH = 16


def generate_data(batch, query_len):
    """DeepSeek-V4-Flash Q [T,64,512], shared KV and 128-token pages."""
    torch.manual_seed(12)
    q = torch.randn(batch * query_len, 64, 512, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(batch * 2, 128, 1, 512, dtype=torch.bfloat16)
    cu = torch.arange(batch + 1, dtype=torch.int32) * query_len
    lengths = torch.tensor(([256, 128] * batch)[:batch], dtype=torch.int32)
    common = dict(
        cu_seqlens_q=cu,
        seqused_q=cu[1:] - cu[:-1],
        seqused_ori_kv=lengths,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    return q, kv, common, batch


def check_sparse_flash_mla_accuracy(data, *, target):
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    q, kv, common, batch = data
    impl, impl_ref = get_test_impls()
    skip_accuracy = get_skip_acc_check() is True
    if skip_accuracy:
        require_test_impl(sparse_flash_mla, impl)
    else:
        require_distinct_impls(sparse_flash_mla, impl, impl_ref)
    metadata_kwargs = dict(
        batch_size=batch,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        has_ori_kv=True,
        has_cmp_kv=False,
        **common,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.arange(batch * 2, dtype=torch.int32).reshape(batch, 2),
        sinks=torch.zeros(64, dtype=torch.float32),
        metadata=sparse_flash_mla_metadata(impl=impl, **metadata_kwargs),
        softmax_scale=512**-0.5,
        **common,
    )
    executor = make_executor(sparse_flash_mla, args=(q,), kwargs=kwargs)
    if not skip_accuracy:
        # Opaque CANN tiling and eager descriptors must be built independently.
        reference_kwargs = {
            **kwargs,
            "metadata": sparse_flash_mla_metadata(impl=impl_ref, **metadata_kwargs),
        }
        expected = sparse_flash_mla(q, impl=impl_ref, **reference_kwargs)
        actual = executor()
        assert_accuracy(
            actual,
            expected,
            {"torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03)},
        )
        executor.accuracy_checked = True
    if target == "compute":
        return executor
    metadata = kwargs["metadata"]
    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    from sgl_kernel_npu.operators.attention import get_sparse_flash_mla_metadata_size

    assert metadata.shape == (get_sparse_flash_mla_metadata_size(),)
    assert metadata.is_contiguous()
    # Opaque metadata is checked through paired compute, not byte comparison.
    return make_executor(
        sparse_flash_mla_metadata,
        kwargs=metadata_kwargs,
        accuracy_checked=executor.accuracy_checked,
    )


def generate_dspark_data(batch, query_len, kv_lengths=None):
    """Explicit logical sparse rows for the DSpark multi-token draft ABI."""
    q, kv, common, _ = generate_data(batch, query_len)
    lengths = common["seqused_ori_kv"]
    if kv_lengths is not None:
        lengths.copy_(torch.tensor(kv_lengths, dtype=torch.int32, device=q.device))
    lengths_cpu = lengths.cpu()
    width = ((128 + query_len + 127) // 128) * 128
    indices_cpu = torch.full(
        (batch * query_len, 1, width), -1, dtype=torch.int32, device="cpu"
    )
    counts_cpu = torch.zeros(batch * query_len, 1, dtype=torch.int32, device="cpu")
    for b, kv_len in enumerate(lengths_cpu.tolist()):
        for offset in range(query_len):
            end = kv_len - query_len + offset + 1
            start = max(0, end - 128)
            row = b * query_len + offset
            count = end - start
            indices_cpu[row, 0, :count] = torch.arange(
                start, end, dtype=torch.int32, device="cpu"
            )
            counts_cpu[row, 0] = count
    common.update(
        ori_mask_mode=0,
        ori_win_left=127 + query_len,
        ori_topk_length=counts_cpu.to(q.device),
        cu_seqlens_q_cpu=common["cu_seqlens_q"].cpu(),
        seqused_ori_kv_cpu=lengths_cpu,
    )
    metadata_kwargs = dict(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        batch_size=batch,
        has_ori_kv=True,
        has_cmp_kv=False,
        ori_topk=width,
        **common,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.arange(batch * 2, dtype=torch.int32).reshape(batch, 2),
        ori_sparse_indices=indices_cpu.to(q.device),
        sinks=torch.zeros(64, dtype=torch.float32),
        softmax_scale=512**-0.5,
        **common,
    )
    return q, metadata_kwargs, kwargs


def _require_cann92_arch22():
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH22:
        pytest.skip("CANN 9.2 sparse Flash MLA requires arch22")
    if detect_cann_version() != "9.2.0":
        pytest.skip("Upstream SWA/CSA/HCA cases require the CANN 9.2 ABI")


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

    _require_cann92_arch22()
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


@pytest.mark.parametrize("cmp_ratio", [3, 128], ids=["csa-ratio-3", "csa-ratio-128"])
def test_sparse_flash_mla_metadata_rejects_unsupported_csa_ratio(cmp_ratio):
    from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

    _require_cann92_arch22()
    require_test_impl(sparse_flash_mla_metadata, "auto")
    query_lengths = (16, 8)
    ori_length, cmp_topk = 128, 8
    cmp_length = ori_length // cmp_ratio
    _, inputs, _ = _build_paged_case(
        cmp_ratio, cmp_length, cmp_topk, query_lengths, ori_length, 24
    )
    # The mainline pin accepts CSA ratios 1, 2 and 4; other ratios are invalid.
    with pytest.raises(RuntimeError, match=r"(?s)cmp_ratio.*must be 1, 2 or 4"):
        _schedule(inputs, cmp_ratio, cmp_length, cmp_topk, query_lengths, ori_length)


def test_sparse_flash_mla_bsnd_swa():
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    _require_cann92_arch22()
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


perf_shape = [(1, 1), (1, 128)]


@pytest.mark.parametrize(
    "batch,query_len",
    [
        pytest.param(1, 1, id="deepseek-v4-flash-decode-b1-q1-n64-d512-page128"),
        pytest.param(1, 128, id="deepseek-v4-flash-prefill-b1-q128-n64-d512-page128"),
        pytest.param(
            2, 2, id="synthetic-ragged-extend", marks=pytest.mark.extra_checks
        ),
    ],
)
def test_sparse_flash_mla(batch, query_len):
    data = generate_data(batch, query_len)
    executor = check_sparse_flash_mla_accuracy(data, target="compute")
    check_performance(executor, enable_perf=(batch, query_len) in perf_shape)


@pytest.mark.parametrize("batch", [1, 10], ids=["draft-b1", "draft-b10"])
def test_dspark_sparse_flash_mla_graph_replay(batch):
    from sgl_kernel_npu.operators.attention import (
        get_sparse_flash_mla_metadata_size,
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    from tests.python.utils.accuracy import assert_accuracy
    from tests.python.utils.common import (
        get_skip_acc_check,
        get_test_impls,
        require_distinct_impls,
    )
    from tests.python.utils.standards import AccuracyStandard

    impl, impl_ref = get_test_impls()
    require_distinct_impls(sparse_flash_mla, impl, impl_ref)
    query, metadata_kwargs, kwargs = generate_dspark_data(batch, 6)
    # Allocation must not launch metadata work on temporary synthetic inputs.
    metadata = torch.zeros(
        get_sparse_flash_mla_metadata_size(),
        dtype=torch.int32,
        device=query.device,
    )
    metadata.copy_(sparse_flash_mla_metadata(impl=impl, **metadata_kwargs))
    kwargs["metadata"] = metadata
    sparse_flash_mla(query, impl=impl, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = sparse_flash_mla(query, impl=impl, **kwargs)
    pointer = metadata.data_ptr()
    for step in (1, 2):
        lengths = [
            256 - step * 16 if b % 2 == 0 else 128 - step * 8 for b in range(batch)
        ]
        query_new, metadata_new, kwargs_new = generate_dspark_data(batch, 6, lengths)
        query.copy_(query_new * (1.0 / (step + 1)))
        for key in ("seqused_ori_kv", "ori_topk_length", "ori_sparse_indices"):
            kwargs[key].copy_(kwargs_new[key])
        metadata_kwargs["seqused_ori_kv_cpu"].copy_(metadata_new["seqused_ori_kv_cpu"])
        metadata.copy_(sparse_flash_mla_metadata(impl=impl, **metadata_kwargs))
        graph.replay()
        torch.npu.synchronize()
        assert metadata.data_ptr() == pointer
        if get_skip_acc_check() is not True:
            expected = sparse_flash_mla(
                query,
                impl=impl_ref,
                **{
                    **kwargs,
                    "metadata": sparse_flash_mla_metadata(
                        impl=impl_ref, **metadata_kwargs
                    ),
                },
            )
            assert_accuracy(
                actual,
                expected,
                {"torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03)},
            )
