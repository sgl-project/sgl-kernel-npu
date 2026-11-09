from functools import partial

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# W: synthetic one-token-per-request PA workload, using GLM-5.3-Flash's
# 32 index heads, 128 channels and pool=4; not a captured production tensor.
perf_shape = [(16, 512, 32)]


def generate_test_data(layout, rows, n_pools, heads, dtype, topk=16):
    """Positive, separated scores make pool ranking independent of roundoff."""
    pool_size = 4
    q = torch.zeros(rows, 1, heads, 128, dtype=dtype)
    q[..., 0] = 1
    weights = torch.ones(rows, 1, heads, dtype=dtype)
    keys = torch.zeros(rows, n_pools, 1, 128, dtype=dtype)
    keys[..., 0] = torch.arange(1, n_pools + 1, dtype=dtype).view(1, n_pools, 1)
    inputs = dict(
        query=q,
        pool_key=keys,
        weights=weights,
        pool_tail_k=[3] * rows,
        topk=topk,
        pool_size=pool_size,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        quant_mode=-1,
        return_value=False,
    )
    if layout == "TND":
        inputs.update(
            query=q.squeeze(1),
            weights=weights.squeeze(1),
            layout_q="TND",
            layout_k="PA_BBND",
            actual_seq_q=list(range(1, rows + 1)),
            actual_seq_k=[n_pools] * rows,
            block_table=torch.arange(rows, dtype=torch.int32).view(rows, 1),
        )
    elif layout != "BSND":
        raise ValueError(layout)
    # Independent literal ranking: scores increase strictly with pool id.
    selected = torch.arange(n_pools - topk // pool_size, n_pools, dtype=torch.int32)
    tokens = (selected[:, None] * pool_size + torch.arange(pool_size)).flatten()
    tail = torch.arange(n_pools * pool_size, n_pools * pool_size + 3, dtype=torch.int32)
    expected = torch.cat((tokens, tail)).to(torch.int32).expand(rows, -1)
    if layout == "BSND":
        expected = expected.unsqueeze(1)
    return inputs, expected


# Six representative tuples replace layout x dtype x shape expansion.
@pytest.mark.parametrize(
    "layout,dtype,rows,n_pools,heads,topk",
    [
        # Case 01: Typical TND BF16: 16 requests, 512 pools, 32 heads, topk2048; profile.
        pytest.param(
            "TND", torch.bfloat16, 16, 512, 32, 2048, id="W-glm53-flash-tnd-topk2048"
        ),
        # Case 02: Typical BSND BF16 with the same full-pool selection.
        pytest.param(
            "BSND", torch.bfloat16, 16, 512, 32, 2048, id="W-glm53-flash-bsnd-topk2048"
        ),
        # Case 03: Minimum TND FP16: one request/head and a ranked subset.
        pytest.param("TND", torch.float16, 1, 16, 1, 16, id="edge-tnd-fp16-minimum"),
        # Case 04: Minimum BSND FP16 with a ranked subset.
        pytest.param("BSND", torch.float16, 1, 16, 1, 16, id="edge-bsnd-fp16-minimum"),
        # Case 05: TND BF16: odd request count and a ranked subset.
        pytest.param(
            "TND", torch.bfloat16, 3, 32, 4, 16, id="edge-odd-requests-ranked-subset"
        ),
        # Case 06: BSND BF16: multiple requests/heads and a ranked subset.
        pytest.param(
            "BSND", torch.bfloat16, 16, 64, 32, 16, id="edge-multi-head-ranked-subset"
        ),
    ],
)
def test_pool_key_indexer(layout, dtype, rows, n_pools, heads, topk):
    from sgl_kernel_npu.operators.indexer import pool_key_indexer

    impl, _ = get_test_impls()
    require_test_impl(pool_key_indexer, impl)
    inputs, expected = generate_test_data(layout, rows, n_pools, heads, dtype, topk)

    # Special accuracy case: the independent ranked fixture allows unordered
    # pool results, so compare the selected token set and ordered tail directly.
    # There is no separate registered reference backend for check_accuracy.
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        actual = pool_key_indexer(**inputs, impl=impl)
        # Pool order is not assumed. Check the selected token set and the
        # separate ordered tail, rather than accepting arbitrary in-range ids.
        assert actual.shape == expected.shape
        assert actual.dtype == torch.int32
        topk = inputs["topk"]
        assert_accuracy(actual[..., :topk].sort().values, expected[..., :topk])
        assert_accuracy(actual[..., topk:], expected[..., topk:])

    executor = partial(pool_key_indexer, **inputs, impl=impl)
    executor.impl = impl
    executor.impl_ref = "independent_ranked_fixture"
    executor.accuracy_checked = accuracy_checked

    # Performance: profile the public entry without sorting/comparison.
    check_performance(
        executor,
        enable_perf=(rows, n_pools, heads) in perf_shape
        and layout == "TND"
        and dtype == torch.bfloat16,
    )
