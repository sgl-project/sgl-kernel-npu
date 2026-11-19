"""Per-token-head FP8 PI: scales must participate in scoring and selection."""

import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.standards import AccuracyStandard


@pytest.fixture(autouse=True)
def require_arch35():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("FP8 PI/cache requires Ascend A5 (arch35)")


@pytest.mark.parametrize("layout", ["BSND", "TND", "PA_BBND"])
def test_pool_key_indexer_fp8(layout):
    from sgl_kernel_npu.operators.indexer import pool_key_indexer

    torch.manual_seed(19)
    impl, _ = get_test_impls()
    q = torch.randn(2, 2, 2, 128).to(torch.float8_e4m3fn)
    k = torch.randn(2, 4, 1, 128).to(torch.float8_e4m3fn)
    weights = torch.rand(2, 2, 2).bfloat16()
    qs = torch.rand(2, 2, 2) + 0.1
    ks = torch.tensor([0.1, 0.25, 2.0, 4.0, 4.0, 2.0, 0.25, 0.1]).reshape(2, 4, 1)
    dequant_q = q.float() * qs.unsqueeze(-1)
    dequant_k = k.float() * ks.unsqueeze(-1)
    scores = torch.einsum("bqhd,bkd->bqhk", dequant_q, dequant_k[:, :, 0])
    scores = (scores * (128**-0.5)).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    values, pools = scores.topk(2, dim=-1)
    expected = torch.full((2, 2, 11), -1, dtype=torch.int32, device=q.device)
    expected[:, :, :8] = (
        pools.unsqueeze(-1) * 4 + torch.arange(4, device=q.device)
    ).reshape(2, 2, 8)
    kwargs = dict(layout_q="BSND", layout_k="BSND")
    if layout != "BSND":
        q = q.reshape(4, 2, 128)
        qs = qs.reshape(4, 2)
        weights = weights.reshape(4, 2)
        kwargs = dict(
            layout_q="TND", layout_k=layout, actual_seq_q=torch.tensor([2, 4])
        )
        if layout == "TND":
            k = k.reshape(8, 1, 128)
            ks = ks.reshape(8, 1)
            kwargs["actual_seq_k"] = torch.tensor([4, 8])
        else:
            pages = torch.zeros(
                2, 16, 1, 128, dtype=torch.float8_e4m3fn, device=k.device
            )
            scales = torch.ones(2, 16, 1, device=k.device)
            pages[:, :4] = k
            scales[:, :4] = ks
            k, ks = pages, scales
            kwargs["block_table"] = torch.tensor(
                [[0], [1]], dtype=torch.int32, device=k.device
            )
            kwargs["actual_seq_k"] = torch.tensor([4, 4])
        expected = expected.reshape(4, 11)
        values = values.reshape(4, 2)
    indices, actual_values = pool_key_indexer(
        q,
        k,
        weights,
        torch.tensor([0, 0]),
        topk=8,
        pool_size=4,
        mask_mode=0,
        quant_mode=0,
        return_value=True,
        q_descale=qs,
        k_descale=ks,
        impl=impl,
        **kwargs,
    )
    assert_accuracy(indices, expected)
    assert_accuracy(
        actual_values, values, {"torch.float32": AccuracyStandard(atol=5e-2, rtol=5e-2)}
    )
