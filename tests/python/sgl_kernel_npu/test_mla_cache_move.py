import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.implementation.triton.mem_cache.move_mla_kv_cache import (
    _mla_move_indices_kernel,
)


@pytest.mark.parametrize("large_slots", [False, True])
def test_nz_indices_preserve_large_physical_rows(large_slots):
    # Even int32 logical slots can expand past int32 physical NZ row limits.
    locations = [2**31 - 129, 2**31 - 128, 2**31 - 1]
    if large_slots:
        locations = [value + 512 for value in locations]
    source = torch.tensor(locations, dtype=torch.int64, device="npu")
    target = source.roll(1)
    outputs = [
        torch.empty(n, dtype=torch.int64, device="npu") for n in (96, 96, 12, 12)
    ]
    _mla_move_indices_kernel[(1,)](
        source,
        target,
        *outputs,
        3,
        SRC_STRIDE=1,
        TGT_STRIDE=1,
        K_TILES=32,
        R_TILES=4,
        PAGE=128,
        PAGE_SHIFT=7,
        LARGE_SLOTS=large_slots,
        BLOCK=256,
    )
    for actual, slots, tiles in zip(
        outputs, [locations, locations[-1:] + locations[:-1]] * 2, [32, 32, 4, 4]
    ):
        expected = []
        for location in slots:
            page, slot = divmod(location, 128)
            expected.extend(
                page * tiles * 128 + tile * 128 + slot for tile in range(tiles)
            )
        torch.testing.assert_close(actual.cpu(), torch.tensor(expected), rtol=0, atol=0)
