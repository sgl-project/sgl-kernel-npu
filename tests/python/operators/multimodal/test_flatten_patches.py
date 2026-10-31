from functools import partial
from math import prod

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# Representative image/video shapes.
perf_shape = [(1, 1, 3, 448, 448), (1, 5, 3, 224, 224)]


def generate_data(shape, dtype, noncontiguous):
    # Distinct neighbouring values expose channel, frame and patch-order errors.
    patches = (torch.arange(prod(shape)) % 127).to(dtype)
    patches = patches.reshape(shape)
    if noncontiguous:
        patches = patches.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not patches.is_contiguous()
    return patches


def reference_flatten_patches(patches, patch_size, merge_size, temporal_patch_size):
    # Evaluate the independent 10-D layout reference on CPU.
    patches = patches.cpu()
    remainder = patches.shape[1] % temporal_patch_size
    if remainder:
        patches = torch.cat(
            [
                patches,
                patches[:, -1:].repeat(1, temporal_patch_size - remainder, 1, 1, 1),
            ],
            dim=1,
        )
    batch, frames, channels, height, width = patches.shape
    grid_t = frames // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = patches.reshape(
        batch,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    ).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return (
        patches.reshape(
            batch,
            grid_t * grid_h * grid_w,
            channels * temporal_patch_size * patch_size * patch_size,
        ),
        grid_t,
        grid_h,
        grid_w,
    )


@pytest.mark.parametrize(
    "shape,patch_size,merge_size,temporal_patch_size,dtype,noncontiguous",
    [
        pytest.param((1, 1, 1, 1, 1), 1, 1, 1, torch.float32, False, id="minimum"),
        pytest.param(
            (2, 1, 3, 8, 12), 2, 2, 2, torch.bfloat16, False, id="image-repeat-frame"
        ),
        pytest.param(
            (2, 5, 3, 8, 12), 2, 2, 3, torch.float32, False, id="video-pad-last-frame"
        ),
        pytest.param(
            (1, 4, 2, 6, 10), 2, 1, 2, torch.float16, False, id="video-no-merge"
        ),
        pytest.param(
            (1, 3, 3, 8, 12), 2, 2, 2, torch.float32, True, id="strided-spatial-input"
        ),
        pytest.param(
            (1, 1, 3, 448, 448),
            14,
            2,
            2,
            torch.bfloat16,
            False,
            id="glm-image-geometry",
        ),
        pytest.param(
            (1, 5, 3, 224, 224),
            14,
            2,
            2,
            torch.bfloat16,
            False,
            id="glm-video-geometry",
        ),
    ],
)
def test_flatten_patches(
    shape, patch_size, merge_size, temporal_patch_size, dtype, noncontiguous
):
    from sgl_kernel_npu.operators.multimodal import flatten_patches

    patches = generate_data(shape, dtype, noncontiguous)
    args = (patches, patch_size, merge_size, temporal_patch_size)
    impl, _ = get_test_impls()
    require_test_impl(flatten_patches, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        original = patches.clone()
        expected, *grid = reference_flatten_patches(*args)
        actual = flatten_patches(*args, impl=impl)
        # Layout-only: require bitwise equality, including dtype and grid.
        assert torch.equal(actual[0], expected.to(patches.device))
        assert actual[0].dtype == patches.dtype
        assert actual[1:] == tuple(grid)
        assert_accuracy(patches, original)

    # auto/native alias one layout implementation; the CPU layout reference is independent.
    executor = partial(flatten_patches, *args, impl=impl)
    executor.impl = impl
    executor.impl_ref = "cpu_layout_reference"
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize(
    "shape,patch_size,merge_size,temporal_patch_size",
    [
        pytest.param((1, 3, 8, 8), 2, 2, 2, id="rank"),
        pytest.param((1, 0, 3, 8, 8), 2, 2, 2, id="empty-video"),
        pytest.param((1, 1, 3, 8, 8), 0, 2, 2, id="zero-patch"),
        pytest.param((1, 1, 3, 8, 8), 2, 0, 2, id="zero-merge"),
        pytest.param((1, 1, 3, 8, 8), 2, 2, 0, id="zero-temporal"),
        pytest.param((1, 1, 3, 7, 8), 2, 2, 2, id="unaligned-height"),
        pytest.param((1, 1, 3, 8, 6), 2, 2, 2, id="unaligned-merge-width"),
    ],
)
def test_flatten_patches_rejects_invalid_inputs(
    shape, patch_size, merge_size, temporal_patch_size
):
    from sgl_kernel_npu.operators.multimodal import flatten_patches

    impl, _ = get_test_impls()
    require_test_impl(flatten_patches, impl)
    patches = torch.zeros(shape)
    with pytest.raises(ValueError):
        flatten_patches(patches, patch_size, merge_size, temporal_patch_size, impl=impl)
