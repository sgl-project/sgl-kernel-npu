"""LPLB physical-expert dispatch: inverse CDF, zero-row fallback and tails."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# DeepSeek-V3-class routing: 256 logical experts, up to 2 physical copies.
# 4096 x 8 is a prefill-sized routed-token batch; 1 x 8 is a decode step.
perf_shape = [(256, 2, 4096, 8)]


def generate_data(shape, mode, seed=42):
    """Build the probability table, physical map and routed ids.

    shape is (num_logical, max_copies, num_tokens, topk).
    normal: random probabilities with -1 padding in the map.
    zero_rows: row 0 has zero probability and no valid copy at all, so it must
        fall back to the clamped last slot; row 1 (when it exists) has zero
        probability but valid copies, so it samples them uniformly. Every token
        routes to those rows.
    int64_ids: int64 routed ids and map instead of int32.
    non_contiguous: routed ids are a strided column slice of a fused layout.
    """
    num_logical, max_copies, num_tokens, topk = shape
    torch.npu.manual_seed(seed)
    prob = torch.rand(num_logical, max_copies, dtype=torch.float32)
    log2phy_map = torch.arange(
        num_logical * max_copies, dtype=torch.int32
    ).view(num_logical, max_copies)
    log2phy_map[prob < 0.25] = -1
    ids = torch.randint(0, num_logical, (num_tokens, topk), dtype=torch.int32)
    if mode == "zero_rows":
        prob[0] = 0.0
        log2phy_map[0] = -1
        ids[:, 0] = 0
        if num_logical > 1:
            prob[1] = 0.0
        if topk > 1:
            ids[:, 1] = min(1, num_logical - 1)
    if mode == "non_contiguous":
        ids = torch.stack([ids, ids], dim=-1).flatten(-2)[:, ::2]
        if ids.numel() and topk > 1:
            assert not ids.is_contiguous()
    if mode == "int64_ids":
        ids = ids.to(torch.int64)
        log2phy_map = log2phy_map.to(torch.int64)
    random_vals = torch.rand(ids.numel(), dtype=torch.float32)
    return ids, prob, log2phy_map, random_vals


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((256, 2, 4096, 8), id="deepseek-v3-prefill"),
        pytest.param((256, 2, 1, 8), id="deepseek-v3-decode"),
        pytest.param((13, 3, 37, 2), id="edge-non-power-of-two-table"),
        pytest.param((8, 4, 0, 8), id="edge-empty-batch"),
        pytest.param((1, 1, 5, 1), id="edge-minimum"),
    ],
)
@pytest.mark.parametrize("mode", ["normal", "zero_rows", "int64_ids", "non_contiguous"])
def test_dispatch_probability(shape, mode):
    from sgl_kernel_npu.operators.moe import dispatch_probability

    args = generate_data(shape, mode)
    executor = check_accuracy(dispatch_probability, args=args)
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and mode == "normal"),
    )
