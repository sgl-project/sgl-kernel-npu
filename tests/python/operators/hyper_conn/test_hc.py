"""HC4/H2560/L320 BF16 mix and combine, with FP32 production intermediates.

Independent CPU FP64 reference, BF16 output; atol=.005, rtol=.01.
Inputs and learned weights must remain unchanged; graph reads updated weights.
"""

import pytest
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_combine, hc_mix

from tests.python.operators.qsa_test_utils import (
    ARCH22_ONLY,
    assert_unchanged,
    check_graph_updates,
)
from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = ARCH22_ONLY

STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.005, rtol=0.01)}
OPERATORS = {"mix": hc_mix, "combine": hc_combine}
perf_shape = [1, 32, 4096]


def _cpu_inputs(op, rows):
    generator = torch.Generator(device="cpu").manual_seed(73)

    def rand(shape, scale=1.0):
        return (torch.randn(shape, generator=generator, device="cpu") * scale).to(
            torch.bfloat16
        )

    x = rand((rows, 10240))
    if op == "mix":
        return x, rand((320, 10240), 0.02), rand((10240, 320), 0.02), 4, 2560
    return rand((rows, 2560)), x, rand((rows, 10240)), rand((4, 10240), 0.02), 4, 2560


def generate_data(op, rows):
    return tuple(
        x.to("npu") if isinstance(x, torch.Tensor) else x for x in _cpu_inputs(op, rows)
    )


@pytest.mark.parametrize("op", ["mix", "combine"])
@pytest.mark.parametrize("rows", [0, 1, 32, 33, 4096, 8193])
def test_hc_accuracy(op, rows):
    args = generate_data(op, rows)
    saved = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    executor = check_accuracy(OPERATORS[op], args=args, standard=STANDARD)
    assert executor().is_contiguous()
    assert_unchanged(args, saved)
    check_performance(executor, enable_perf=rows in perf_shape)


@pytest.mark.parametrize("op", ["mix", "combine"])
@pytest.mark.parametrize("rows", [33, 8193])
def test_hc_graph_input_and_weight_updates(op, rows):
    args = generate_data(op, rows)
    updates = [
        lambda x=x: x.mul_(-0.5).add_(0.03125)
        for x in args
        if isinstance(x, torch.Tensor)
    ]
    check_graph_updates(OPERATORS[op], args, updates, standard=STANDARD)


@pytest.mark.parametrize("op", ["mix", "combine"])
@pytest.mark.parametrize("bad", ["dtype", "shape", "layout", "hc"])
def test_hc_rejects_metadata(op, bad):
    args = list(generate_data(op, 1))
    if bad == "dtype":
        args[0] = args[0].float()
    elif bad == "shape":
        args[0] = args[0][:, :-1].contiguous()
    elif bad == "layout":
        args[0] = torch.stack((args[0], args[0]), -1)[..., 0]
    else:
        args[-2] = 5
    with pytest.raises(ValueError):
        OPERATORS[op](*args)


def test_hc_combine_cancellation():
    args = list(generate_data("combine", 1))
    block, _, normed, weight = args[:4]
    gate = 2 * torch.sigmoid(normed.cpu().double() @ weight.cpu().double().T / 4)
    args[1] = (
        (-(block.cpu().double()[:, None, :] * gate[:, :, None]))
        .flatten(1)
        .to(device=block.device, dtype=block.dtype)
    )
    check_accuracy(hc_combine, args=args, standard=STANDARD)
