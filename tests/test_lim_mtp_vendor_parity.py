"""Random BF16 Q4 full-visible LIM parity against the vendor causal indexer."""

import json
import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("sgl_kernel_npu")

from test_lim_mtp_full_visible import CAPACITY, ROUTES, TOPK, _lim_case  # noqa: E402


@pytest.mark.parametrize("length", [16385, 64000])
def test_q4_random_vendor_causal_selection_and_attention(length):
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    args = _lim_case(length)
    gen = torch.Generator().manual_seed(20261028 + length)
    q = torch.randn((ROUTES, 32, 128), generator=gen).to(torch.bfloat16)
    keys = torch.randn((args[4].shape[0] * 128, 128), generator=gen).to(torch.bfloat16)
    weights = torch.rand((ROUTES, 32), generator=gen).to(torch.bfloat16)
    args[2].copy_(q)
    args[0].copy_(weights)
    args[4].copy_(keys.view(-1, 128, 1, 128))
    torch.ops.npu.fused_lightning_indexer_manage(*args, full_visible=True)
    vendor = torch_npu.npu_lightning_indexer(
        query=args[2],
        key=args[4],
        weights=args[0],
        actual_seq_lengths_query=args[6],
        actual_seq_lengths_key=args[7],
        block_table=args[5],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    vendor = vendor[0] if isinstance(vendor, (tuple, list)) else vendor
    torch.npu.synchronize()
    actual = args[13].cpu()[:, 0].long()
    reference = vendor.cpu().view(ROUTES, TOPK).long()
    slots = args[14].cpu()[:, 0].long()
    count = int(args[18].item())
    assert 0 < count <= CAPACITY
    miss_sources = args[16].cpu()[0, :count].long()
    miss_slots = args[17].cpu()[0, :count].long()
    assert miss_slots.unique().numel() == count
    resident = torch.full((CAPACITY,), -1, dtype=torch.long)
    resident[miss_slots] = miss_sources
    payload = torch.randn((length, 576), generator=gen)
    attn_q = torch.randn((ROUTES, 4, 576), generator=gen)
    for route in range(ROUTES):
        visible = length - ROUTES + route + 1
        a, b = actual[route], reference[route]
        assert torch.all((a >= 0) & (a < visible))
        assert torch.all((b >= 0) & (b < visible))
        torch.testing.assert_close(resident[slots[route]], a, rtol=0, atol=0)
        # The FP32 score oracle describes the observed TopK boundary and ties;
        # it is diagnostic because the native BF16 reduction can round differently.
        scores = torch.relu(q[route].float() @ keys[:visible].float().T)
        scores = (scores * weights[route].float()[:, None]).sum(0)
        ranked = scores.topk(TOPK + 1).values
        cutoff, margin = float(ranked[TOPK - 1]), float(ranked[TOPK - 1] - ranked[TOPK])
        set_a, set_b = set(a.tolist()), set(b.tolist())
        selected_a, selected_b = payload[a], payload[b]
        scores_a = attn_q[route] @ selected_a.T / math.sqrt(576)
        scores_b = attn_q[route] @ selected_b.T / math.sqrt(576)
        out_a = torch.softmax(scores_a, dim=-1) @ selected_a[:, :512]
        out_b = torch.softmax(scores_b, dim=-1) @ selected_b[:, :512]
        diff = out_a - out_b
        print(
            "LIM_VENDOR_METRICS "
            + json.dumps(
                {
                    "length": length,
                    "route": route,
                    "intersection": len(set_a & set_b),
                    "topk": TOPK,
                    "fp32_boundary_margin": margin,
                    "fp32_boundary_ties": int((scores == cutoff).sum()),
                    "lim_below_fp32_cutoff": int((scores[a] < cutoff).sum()),
                    "vendor_below_fp32_cutoff": int((scores[b] < cutoff).sum()),
                    "lim_min_score_minus_cutoff": float(scores[a].min() - cutoff),
                    "vendor_min_score_minus_cutoff": float(scores[b].min() - cutoff),
                    "attention_max_abs": float(diff.abs().max()),
                    "attention_rms": float(diff.square().mean().sqrt()),
                }
            )
        )
        torch.testing.assert_close(out_a, out_b, rtol=0.01, atol=0.002)
