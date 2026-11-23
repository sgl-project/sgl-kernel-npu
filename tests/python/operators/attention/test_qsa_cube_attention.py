"""Short A3 prefill with shuffled 128-token pages, 512 groups, and a partial tail."""
from importlib import import_module
from unittest.mock import patch

import pytest
import torch
from sgl_kernel_npu.operators.attention.qsa_cube_attention import (
    is_qsa_cube_attention_supported,
    qsa_cube_attention,
)


@pytest.mark.parametrize(
    "arch,soc,expected", [(2201, 220, False), (2201, 253, True), (3510, 260, False)]
)
def test_short_cube_target(arch, soc, expected):
    from sgl_kernel_npu.operators.utils.generation import Hardware, generation_from_arch

    module = import_module(is_qsa_cube_attention_supported.__module__)
    hardware = Hardware(arch, generation_from_arch(arch))
    with (
        patch.object(module, "get_hardware", return_value=hardware),
        patch.object(module, "get_soc_version", return_value=soc),
    ):
        assert is_qsa_cube_attention_supported.__wrapped__() is expected


@pytest.mark.skipif(not is_qsa_cube_attention_supported(), reason="A3 QSA Cube")
@pytest.mark.parametrize("compact", [True, False])
def test_qsa_cube_attention(compact):
    rng = torch.Generator().manual_seed(75)
    rows, prefix = 7, 65529
    length = prefix + rows
    q = (torch.randn(rows, 3, 256, generator=rng, device="cpu") * 0.2).bfloat16()
    k = (torch.randn(512, 128, 1, 256, generator=rng, device="cpu") * 0.2).bfloat16()
    v = torch.randn(k.shape, generator=rng, device="cpu").bfloat16()
    pages = torch.randperm(512, generator=rng, device="cpu").int().unsqueeze(0)
    packed = torch.full((rows, 514 if compact else 2051), -1, dtype=torch.int32, device="cpu")
    expected = []
    for row in range(rows):
        visible = prefix + row + 1
        groups = torch.randperm(visible // 4, generator=rng, device="cpu")[:512]
        tail = torch.arange(visible // 4 * 4, visible, device="cpu")
        tokens = torch.cat(((groups[:, None] * 4 + torch.arange(4, device="cpu")).flatten(), tail))
        if compact:
            packed[row, :512] = groups
            packed[row, -2:] = torch.tensor([visible - 1, length], device="cpu")
        else:
            packed[row, :tokens.numel()] = tokens
        chosen = pages[0, tokens // 128].long()
        keys = k[chosen, tokens % 128, 0].double()
        values = v[chosen, tokens % 128, 0].double()
        expected.append(((q[row].double() @ keys.T / 16).softmax(-1) @ values).bfloat16())
    requests = torch.zeros(rows, dtype=torch.int32, device="npu")
    output = qsa_cube_attention(q.npu(), k.npu(), v.npu(), packed.npu(), pages.npu(), requests)
    torch.testing.assert_close(output.cpu(), torch.stack(expected), atol=0.004, rtol=0.02)
