"""A3 source/consumer coverage with distinct analytic scores, not production traces."""

from functools import partial

import pytest

torch = pytest.importorskip("torch")

from sgl_kernel_npu.operators.indexer import (
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
    sparse_lightning_indexer,
)
from sgl_kernel_npu.operators.utils.registry import register_impl, register_operator

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor, require_test_impl
from tests.python.utils.performance import check_performance


@register_operator("test.lightning_indexer.canonical")
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


for _impl in ("auto", "cann_ops_transformers_cann920", "torch_native"):
    register_impl("test.lightning_indexer.canonical", _impl)(
        partial(canonical, impl=_impl)
    )


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
    checked = check_accuracy(canonical, args=(operator, args, kwargs))
    executor = make_executor(
        operator, args=args, kwargs=kwargs, accuracy_checked=checked.accuracy_checked
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


@pytest.mark.parametrize(
    "nq",
    [16, 1],
    ids=["DeepSeek-V4.1-Flash-chunked-prefill", "DeepSeek-V4.1-Flash-decode"],
)
@pytest.mark.parametrize(
    "operator",
    [lightning_indexer_v2, sparse_lightning_indexer],
    ids=["source", "consumer"],
)
def test_model_scenario(operator, nq):
    # Layer 20 and subsequent ratio-1 layers use a two-level indexer:
    # https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/517ef62/config.json
    # 32 query heads, D=128, token TopK=512, 2048 candidate blocks of 8.
    # Generated BF16 activations exercise a 32K context; no trace is claimed.
    require_test_impl(lightning_indexer_v2, "auto")
    torch.manual_seed(42)
    q = torch.randn(1, nq, 32, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 32768, 1, 128, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype) * 128**0.5
    w = torch.randn(1, nq, 32, dtype=torch.float32) / (128 * 32) ** 0.5
    args = (q, k, w, 512)
    kwargs = dict(mask_mode=3, return_value=1)
    if operator is lightning_indexer_v2:
        kwargs.update(candidate_topk_blocks=2048, candidate_block_size=8)
    else:
        # Candidate indices are produced by the source, as in the model.
        candidates = lightning_indexer_v2(
            *args,
            mask_mode=3,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
        )[2]
        args += (candidates,)
        kwargs["candidate_block_size"] = 8
    check_operator(operator, args, kwargs)


@pytest.mark.parametrize("model", ["DeepSeek-V4.1-Flash"])
def test_model_metadata(model):
    # Same model configuration as test_model_scenario; A3 ignores metadata.
    require_test_impl(lightning_indexer_v2_metadata, "auto")
    args = (*generate_test_data(1, 1, 32768, 32, torch.bfloat16), 512)
    metadata = lightning_indexer_v2_metadata(
        32,
        1,
        128,
        512,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=32768,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=1,
    )
    assert metadata.dtype == torch.int32 and metadata.shape == (2048,)
    check_operator(lightning_indexer_v2, args, dict(mask_mode=3, metadata=metadata))
