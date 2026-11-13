"""Check the public matmul dispatch without launching either NPU kernel."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("generation", ["ARCH22", "ARCH35"])
@pytest.mark.parametrize(
    "shape, is_prefill, arch22_uses_ascendc",
    [
        ((2, 16, 512, 128), None, False),
        ((2, 128, 512, 128), None, True),
        ((65536, 1, 1, 1), None, True),
        ((1, 1, 1, 65536), None, True),
        ((2, 16, 512, 128), True, False),
        ((2, 128, 512, 128), True, False),
        ((2, 16, 512, 128), False, True),
    ],
)
def test_batch_matmul_transpose_auto_dispatch(
    registry_for_generation,
    monkeypatch,
    generation,
    shape,
    is_prefill,
    arch22_uses_ascendc,
):
    import torch
    import torch_npu

    modules = registry_for_generation(generation)
    matmul = modules.import_module("operators.matmul").batch_matmul_transpose
    m, batch, k, n = shape
    a = torch.zeros((m, batch, k * 2), dtype=torch.bfloat16)[..., ::2]
    b = torch.zeros((batch, k, n), dtype=a.dtype)
    vendor_output = a.new_ones((m, batch, n))
    vendor = Mock(return_value=vendor_output)
    ascendc = Mock(side_effect=lambda a, b, out: out.fill_(2))
    monkeypatch.setattr(torch_npu, "npu_transpose_batchmatmul", vendor)
    monkeypatch.setattr(
        torch.ops, "npu", SimpleNamespace(batch_matmul_transpose=ascendc)
    )

    output = matmul(a, b, is_prefill=is_prefill)
    if generation == "ARCH22" and arch22_uses_ascendc:
        vendor.assert_not_called()
        ascendc.assert_called_once()
        assert ascendc.call_args.args[0].is_contiguous()
        assert ascendc.call_args.args[2] is output
        torch.testing.assert_close(output, torch.full_like(vendor_output, 2))
    else:
        ascendc.assert_not_called()
        vendor.assert_called_once_with(
            a, b, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
        )
        assert output is vendor_output
