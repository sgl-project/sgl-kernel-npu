"""Four independent 2560-wide Gemma norm groups in BF16 x[R,10240].

Independent CPU FP64 reference; atol=rtol=.005, unchanged from the old tests.
R4096/R8193 have disclosed historical failures: keep them visible, not xfailed.
"""

import pytest
import torch
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

from tests.python.operators.qsa_test_utils import (
    ARCH22_ONLY,
    assert_unchanged,
    check_graph_updates,
)
from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = ARCH22_ONLY

STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.005, rtol=0.005)}
perf_shape = [1, 32, 4096]


def generate_data(rows):
    generator = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn(rows, 10240, generator=generator, device="cpu").to(torch.bfloat16)
    weight = (torch.randn(10240, generator=generator, device="cpu") * 0.1).to(
        torch.bfloat16
    )
    return x.to("npu"), weight.to("npu"), 2560, 1e-6


@pytest.mark.parametrize("rows", [0, 1, 32, 33, 4096, 8193])
def test_grouped_gemma_rmsnorm_accuracy(rows):
    args = generate_data(rows)
    saved = [x.clone() for x in args[:2]]
    executor = check_accuracy(grouped_gemma_rmsnorm, args=args, standard=STANDARD)
    assert executor().is_contiguous()
    assert_unchanged(args, saved)
    check_performance(executor, enable_perf=rows in perf_shape)


def test_grouped_gemma_rmsnorm_graph_updates():
    args = generate_data(8)
    check_graph_updates(
        grouped_gemma_rmsnorm,
        args,
        [lambda: args[0].mul_(-0.5), lambda: args[1].add_(0.125)],
        standard=STANDARD,
    )


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), True])
def test_grouped_gemma_rmsnorm_rejects_epsilon(eps):
    x, weight, group, _ = generate_data(0)
    with pytest.raises(ValueError):
        grouped_gemma_rmsnorm(x, weight, group, eps)
