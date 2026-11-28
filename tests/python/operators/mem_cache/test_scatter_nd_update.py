"""Check a DeepSeek paged indexer cache update and in-place aliasing."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")


def generate_data(page_shape, slots, dtype):
    destination = torch.zeros(page_shape, dtype=torch.uint8, device="npu").view(dtype)
    # The C4 indexer has 32 compressed slots per 128-token model page.
    destination = destination.flatten(0, 1)
    indices = torch.tensor(
        slots, dtype=torch.int64, device=destination.device
    ).unsqueeze(-1)
    generator = torch.Generator(device="cpu").manual_seed(42)
    updates = torch.randint(
        -8, 9, (len(slots), *destination.shape[1:]), device="cpu", generator=generator
    )
    updates = updates.to(dtype).view(torch.uint8).npu().view(dtype)
    return dict(input=destination, indices=indices, updates=updates)


@pytest.mark.parametrize(
    "page_shape, slots",
    [
        pytest.param(
            (4, 32, 1, 128), (65, 63, 96), id="deepseek-v4-flash-c4-indexer-k-cache"
        )
    ],
)
def test_scatter_nd_update_(page_shape, slots):
    from sgl_kernel_npu.operators.mem_cache import scatter_nd_update_
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    dtype = (
        torch.float8_e4m3fn
        if get_hardware().generation is Generation.ARCH35
        else torch.int8
    )
    kwargs = generate_data(page_shape, slots, dtype)
    executor = check_accuracy(
        scatter_nd_update_,
        kwargs=kwargs,
        mutable_kwargs=("input",),
        standard={"torch.float8_e4m3fn": AccuracyStandard(atol=0, rtol=0)},
    )
    if executor.accuracy_checked:
        assert executor().data_ptr() == kwargs["input"].data_ptr()
    check_performance(executor)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_scatter_nd_update_ascendc_negative_indices(dtype):
    from sgl_kernel_npu.operators.mem_cache import scatter_nd_update_

    require_test_impl(scatter_nd_update_, "ascendc")
    cache = torch.full((4, 33), 11, dtype=dtype, device="npu")
    indices = torch.tensor([[2], [-1], [0]], dtype=torch.int64, device=cache.device)
    updates = torch.arange(99, dtype=torch.float32).reshape(3, 33).to(dtype)
    expected = cache.cpu()
    expected[2] = updates[0]
    expected[0] = updates[2]

    result = scatter_nd_update_(
        cache, indices, updates.to(cache.device), impl="ascendc"
    )

    assert result is cache
    assert torch.equal(cache.cpu(), expected)
