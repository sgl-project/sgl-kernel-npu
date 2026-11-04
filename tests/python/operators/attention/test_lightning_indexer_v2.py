"""A3 source/consumer coverage with distinct analytic scores, not production traces."""

import pytest

torch = pytest.importorskip("torch")

from sgl_kernel_npu.operators.attention import (
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
    sparse_lightning_indexer,
)

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def canonical(operator, args, kwargs, *, impl="auto"):
    """Compare sets and values together; upstream does not promise slot ordering."""
    out = list(operator(*args, **kwargs, impl=impl))
    indices, order = out[0].sort(dim=-1)
    out[0] = indices
    if out[1].numel():
        out[1] = out[1].gather(-1, order)
    if len(out) == 4 and out[2].numel():
        out[2] = out[2].sort(dim=-1).values
    return tuple(out)


def generate_test_data(batch, nq, nk, heads, dtype):
    # Positive, distinct scores prevent ambiguous TopK membership at ties.
    q = torch.ones(batch, nq, heads, 128, dtype=dtype)
    k = torch.zeros(batch, nk, 1, 128, dtype=dtype)
    # Two exactly representable channels keep all scores distinct even in bf16.
    position = torch.arange(nk, dtype=torch.int32)
    k[:, :, 0, 0] = (position // 128).to(dtype)
    k[:, :, 0, 1] = (position % 128).to(dtype) / 128
    w = torch.full((batch, nq, heads), 1 / heads, dtype=torch.float32)
    return q, k, w


def check_operator(operator, args, kwargs, *, enable_perf=False):
    require_test_impl(operator, "auto")
    # Accuracy normalization is outside the public operator performance region.
    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(operator, impl, impl_ref)
        assert_accuracy(
            canonical(operator, args, kwargs, impl=impl),
            canonical(operator, args, kwargs, impl=impl_ref),
        )
    executor = make_executor(
        operator, args=args, kwargs=kwargs, accuracy_checked=checked
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_mode", [0, 3])
@pytest.mark.parametrize("return_value", [0, 1])
@pytest.mark.parametrize("candidate_topk_blocks", [None, -1, 64, 128])
def test_lightning_indexer_v2(dtype, mask_mode, return_value, candidate_topk_blocks):
    args = (*generate_test_data(2, 4, 513, 32, dtype), 16)
    kwargs = dict(
        mask_mode=mask_mode,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
    )
    check_operator(lightning_indexer_v2, args, kwargs)


@pytest.mark.parametrize("topk", [1, 2048, 3072])
def test_dense_topk_boundaries(topk):
    args = (*generate_test_data(1, 1, 16, 1, torch.float16), topk)
    check_operator(lightning_indexer_v2, args, dict(return_value=1))


@pytest.mark.parametrize("block_size", [2, 8, 64])
@pytest.mark.parametrize("return_value", [0, 1])
@pytest.mark.parametrize("candidate_kind", ["source", "dirty", "empty"])
def test_sparse_lightning_indexer(block_size, return_value, candidate_kind):
    require_test_impl(lightning_indexer_v2, "auto")
    q, k, w = generate_test_data(1, 4, 513, 64, torch.bfloat16)
    _, _, candidates, length = lightning_indexer_v2(
        q, k, w, 16, candidate_topk_blocks=64, candidate_block_size=block_size
    )
    if candidate_kind != "source":
        candidates = torch.full((1, 4, 1, 64), -1, dtype=torch.int32)
        if candidate_kind == "dirty":
            candidates[..., :4] = torch.tensor([0, 0, -1, 999999], dtype=torch.int32)
    kwargs = dict(
        candidate_block_size=block_size,
        return_value=return_value,
        candidate_block_length=length,
        mask_mode=3,
    )
    check_operator(sparse_lightning_indexer, (q, k, w, 16, candidates), kwargs)


def test_metadata_combination():
    require_test_impl(lightning_indexer_v2_metadata, "auto")
    q, k, w = generate_test_data(1, 1, 128, 32, torch.float16)
    metadata = lightning_indexer_v2_metadata(
        32,
        1,
        128,
        16,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=128,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=0,
    )
    assert metadata.dtype == torch.int32 and metadata.shape == (2048,)
    check_operator(lightning_indexer_v2, (q, k, w, 16), dict(metadata=metadata))


@pytest.mark.parametrize("operator", [lightning_indexer_v2, sparse_lightning_indexer])
def test_representative_performance(operator):
    # Independently selected representative dimensions; no production provenance claimed.
    q, k, w = generate_test_data(1, 4, 8192, 64, torch.bfloat16)
    args = (q, k, w, 2048)
    kwargs = {}
    if operator is sparse_lightning_indexer:
        candidates = (
            torch.arange(64, dtype=torch.int32).expand(1, 4, 1, 64).contiguous()
        )
        args += (candidates,)
    check_operator(operator, args, kwargs, enable_perf=True)
