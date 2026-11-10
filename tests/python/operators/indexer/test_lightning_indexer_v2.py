"""Basic DeepSeek-V4.1-Flash source/consumer and metadata coverage."""

import pytest
import torch
from sgl_kernel_npu.operators.indexer import (
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
    sparse_lightning_indexer,
)

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def generate_test_data(nq):
    # Model dimensions only; activations are generated rather than captured:
    # https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/517ef62/config.json
    # Ratio-1 source/consumer layers: H=32, D=128, TopK=512, 2048 blocks of 8.
    torch.manual_seed(42)
    q = torch.randn(1, nq, 32, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 32768, 1, 128, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype) * 128**0.5
    w = torch.randn(1, nq, 32, dtype=torch.float32) / (128 * 32) ** 0.5
    return q, k, w, 512


def canonical(outputs):
    # Upstream does not promise TopK slot order; keep values paired with indices.
    out = list(outputs)
    out[0], order = out[0].sort(dim=-1)
    if out[1].numel():
        out[1] = out[1].gather(-1, order)
    if len(out) == 4 and out[2].numel():
        out[2] = out[2].sort(dim=-1).values
    return tuple(out)


def check_operator(operator, args, kwargs, *, enable_perf=False):
    executor = make_executor(operator, args=args, kwargs=kwargs)
    if get_skip_acc_check():
        require_test_impl(operator, executor.impl)
    else:
        require_distinct_impls(operator, executor.impl, executor.impl_ref)
        actual = canonical(operator(*args, **kwargs, impl=executor.impl))
        reference = canonical(operator(*args, **kwargs, impl=executor.impl_ref))
        assert_accuracy(actual, reference)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize(
    "nq",
    [
        pytest.param(16, id="DeepSeek-V4.1-Flash-chunked-prefill"),
        pytest.param(1, id="DeepSeek-V4.1-Flash-decode"),
    ],
)
@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(lightning_indexer_v2, id="source"),
        pytest.param(sparse_lightning_indexer, id="consumer"),
    ],
)
def test_lightning_indexer(operator, nq):
    args = generate_test_data(nq)
    kwargs = dict(mask_mode=3, return_value=1)
    executor = make_executor(operator)
    require_test_impl(operator, executor.impl)
    if operator is lightning_indexer_v2:
        kwargs.update(candidate_topk_blocks=2048, candidate_block_size=8)
    else:
        # Reuse the source's candidates, as in subsequent model layers.
        candidates = lightning_indexer_v2(
            *args,
            mask_mode=3,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            impl=executor.impl,
        )[2]
        args += (candidates,)
        kwargs["candidate_block_size"] = 8
    check_operator(operator, args, kwargs, enable_perf=nq == 1)


@pytest.mark.parametrize("nq", [pytest.param(1, id="DeepSeek-V4.1-Flash-metadata")])
def test_lightning_indexer_metadata(nq):
    args = generate_test_data(nq)
    executor = make_executor(lightning_indexer_v2)
    require_test_impl(lightning_indexer_v2_metadata, executor.impl)
    # There is no independent backend for opaque scheduling data. Check its
    # contract, then compare the combined main call against its Torch reference.
    metadata = lightning_indexer_v2_metadata(
        32,
        1,
        128,
        512,
        batch_size=1,
        max_seqlen_q=nq,
        max_seqlen_k=32768,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=1,
        impl=executor.impl,
    )
    assert metadata.dtype == torch.int32 and metadata.shape == (2048,)
    check_operator(lightning_indexer_v2, args, dict(mask_mode=3, metadata=metadata))
