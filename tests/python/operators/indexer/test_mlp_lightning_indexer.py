from itertools import accumulate

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Physical query rows, maximum KV length, query heads, sparse_count. These are
# synthetic LongCat2-shaped workloads, not end-to-end model measurements.
perf_shape = [(352, 640, 8, 128), (16, 4096, 8, 2048)]


def generate_data(case, dtype):
    """Build independent requests with shuffled PA pages on the fixture device."""
    torch.manual_seed(42)
    q_lens, k_lens = case["q_lens"], case["k_lens"]
    rows = sum(max(length, 1) for length in q_lens)
    heads, head_dim, page_size = 8, 128, 16
    page_counts = [max((length + page_size - 1) // page_size, 1) for length in k_lens]
    page_ids = torch.randperm(sum(page_counts), dtype=torch.int64)
    block_table = torch.full((len(q_lens), max(page_counts)), -1, dtype=torch.int32)
    cursor = 0
    for batch, count in enumerate(page_counts):
        block_table[batch, :count] = page_ids[cursor : cursor + count].to(torch.int32)
        cursor += count

    # Binary fractions keep the fp32 score arithmetic exact for both fp16 and
    # bf16 inputs, while exercising signed weights, ReLU and all head channels.
    query = torch.randint(-4, 5, (rows, heads, head_dim)).to(dtype) / 4
    key = torch.randint(-4, 5, (sum(page_counts), page_size, 1, head_dim)).to(dtype) / 4
    weights = torch.randint(-4, 5, (rows, heads)).float() / 8
    if case.get("ties"):
        query.zero_()

    inputs = dict(
        query=query,
        key=key,
        weights=weights,
        cur_seq_lengths_query=torch.tensor([0, *accumulate(q_lens)], dtype=torch.int64),
        cur_seq_lengths_key=torch.tensor([0, *accumulate(k_lens)], dtype=torch.int64),
        block_table=block_table,
        init_tensor=None,
        local_tensor=None,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=case.get("sparse_count", 8),
        kv_block_len=case.get("kv_block_len", 1),
        q_block_len=1,
        init_num=case.get("init_num", 0),
        local_num=case.get("local_num", 0),
        sparse_mode=case.get("sparse_mode", 3),
        return_value=case.get("return_value", True),
    )
    if case.get("tensor_windows"):
        inputs["init_tensor"] = torch.tensor([0, 4], dtype=torch.int32)
        inputs["local_tensor"] = torch.tensor([4, 8], dtype=torch.int32)
    if case.get("layout_key") == "TND":
        inputs["key"] = torch.cat(
            [
                key[block_table[batch, :count].long()].reshape(-1, 1, head_dim)[
                    : max(length, 1)
                ]
                for batch, (count, length) in enumerate(zip(page_counts, k_lens))
            ],
            dim=0,
        )
        inputs["layout_key"] = "TND"
        inputs["block_table"] = None
    return inputs


def _assert_selection(actual, reference, inputs):
    """Compare score-optimal unique selections, allowing different tied indices."""
    from sgl_kernel_npu.implementation.torch_native.indexer.mlp_lightning_indexer import (
        _compute_scores,
    )

    shape = (inputs["query"].shape[0], 1, inputs["sparse_count"])
    for result in (actual, reference):
        assert isinstance(result, tuple) and len(result) == 2
        for tensor, dtype in zip(result, (torch.int32, inputs["query"].dtype)):
            assert tensor.shape == shape
            assert tensor.dtype == dtype
            assert tensor.device == inputs["query"].device

    score_inputs = {
        name: value
        for name, value in inputs.items()
        if name not in ("sparse_count", "return_value")
    }
    scores = _compute_scores(**score_inputs)
    expected_count = scores.isfinite().sum(-1).clamp_max(inputs["sparse_count"])
    selected_scores = []
    for indices, values in (actual, reference):
        indices = indices[:, 0].long()
        assert bool(((indices >= -1) & (indices < scores.shape[1])).all())
        valid = indices >= 0
        assert_accuracy(valid.sum(-1), expected_count)
        sorted_indices = indices.sort(dim=-1).values
        repeated = (sorted_indices[:, 1:] == sorted_indices[:, :-1]) & (
            sorted_indices[:, 1:] >= 0
        )
        assert not bool(repeated.any()), "Top-k contains a repeated valid block"
        chosen = scores.gather(1, indices.clamp_min(0)).masked_fill(
            ~valid, float("-inf")
        )
        assert_accuracy(chosen.isfinite(), valid)
        selected_scores.append(chosen.sort(dim=-1, descending=True).values)
        if inputs["return_value"]:
            expected_values = chosen.clamp_max(65504).to(values.dtype).unsqueeze(1)
            assert_accuracy(values, expected_values)

    # Exact oracle scores distinguish true ties from nearby non-tied candidates;
    # comparing only rounded bf16 output values could conceal ranking mistakes.
    assert_accuracy(
        selected_scores[0],
        selected_scores[1],
        {"torch.float32": AccuracyStandard(atol=0, rtol=0)},
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "case",
    [
        # Retain the original PR's four request/length combinations.
        pytest.param(
            dict(q_lens=[1, 2], k_lens=[16, 24], sparse_mode=0), id="legacy-mode0"
        ),
        pytest.param(dict(q_lens=[2, 3], k_lens=[16, 32]), id="legacy-causal"),
        pytest.param(
            dict(
                q_lens=[1, 2, 1, 3, 2, 4, 1, 2],
                k_lens=[16, 32, 48, 64, 80, 96, 112, 128],
                sparse_count=16,
            ),
            id="legacy-batch8",
        ),
        pytest.param(
            dict(
                q_lens=[32, 48, 16, 64, 24, 40, 56, 72],
                k_lens=[128, 256, 192, 384, 320, 448, 512, 640],
                sparse_count=128,
                init_num=16,
                local_num=64,
            ),
            id="W-longcat2-prefill-sink-local",
        ),
        pytest.param(
            dict(
                q_lens=[1] * 16,
                k_lens=[4096] * 16,
                sparse_count=2048,
                init_num=16,
                local_num=64,
                return_value=False,
            ),
            id="W-longcat2-decode",
        ),
        pytest.param(dict(q_lens=[1], k_lens=[1]), id="edge-minimum-padding"),
        pytest.param(
            dict(q_lens=[1, 0, 2], k_lens=[16, 16, 32], sparse_count=32),
            id="edge-zero-query-physical-slot",
        ),
        pytest.param(
            dict(q_lens=[2, 3], k_lens=[32, 64], sparse_count=16, ties=True),
            id="edge-equal-scores",
        ),
        pytest.param(
            dict(
                q_lens=[2, 3],
                k_lens=[32, 64],
                sparse_count=8,
                init_num=16,
                local_num=64,
            ),
            id="edge-overlapping-windows",
        ),
        pytest.param(
            dict(q_lens=[1, 2], k_lens=[32, 64], tensor_windows=True),
            id="edge-per-request-windows",
        ),
        pytest.param(
            dict(q_lens=[1, 2], k_lens=[16, 24], layout_key="TND"),
            id="edge-contiguous-key",
        ),
        *[
            pytest.param(
                dict(
                    q_lens=[2, 3],
                    k_lens=[32, 64],
                    kv_block_len=block,
                    init_num=16,
                    local_num=16,
                ),
                id=f"edge-kv-block{block}",
            )
            for block in (2, 4, 8, 16)
        ],
    ],
)
def test_mlp_lightning_indexer(case, dtype):
    from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer

    impl, impl_ref = get_test_impls()
    require_test_impl(mlp_lightning_indexer, impl)
    inputs = generate_data(case, dtype)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(mlp_lightning_indexer, impl, impl_ref)
        reference = mlp_lightning_indexer(**inputs, impl=impl_ref)
        actual = mlp_lightning_indexer(**inputs, impl=impl)
        # Custom comparison is necessary for tied top-k choices and the
        # intentionally undefined values when return_value=False.
        _assert_selection(actual, reference, inputs)

    executor = make_executor(
        mlp_lightning_indexer,
        kwargs=inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    shape = (
        inputs["query"].shape[0],
        max(case["k_lens"]),
        inputs["query"].shape[1],
        inputs["sparse_count"],
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and dtype == torch.bfloat16
    )
