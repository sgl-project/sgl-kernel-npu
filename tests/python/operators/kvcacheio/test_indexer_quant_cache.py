"""On-device production/reference checks for kvcacheio operators."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_indexer_cache(impl_pair):
    from sgl_kernel_npu.operators.kvcacheio import indexer_quant_cache

    x = torch.randn(8, 128, device="npu", dtype=torch.bfloat16)
    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], device="npu", dtype=torch.int32)
    outputs = []
    for selected in impl_pair:
        keys = torch.zeros(1, 128, 1, 128, device="npu", dtype=torch.float8_e4m3fn)
        scales = torch.zeros(1, 128, 1, 1, device="npu", dtype=torch.float32)
        indexer_quant_cache(keys, scales, x, slots, quant_mode="fp8", impl=selected)
        outputs.append((keys, scales))
    torch.testing.assert_close(
        outputs[0][0].float(), outputs[1][0].float(), atol=16, rtol=0.06
    )
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=1e-5, rtol=0.03)
