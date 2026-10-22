"""Native full-visible LIM plans consumed directly by combined Copy-SFA."""

import pytest
import torch
from test_copy_sfa_combined import _CopySfaCase


@pytest.mark.parametrize("graph_replay", [False, True])
def test_lim_copy_sfa_partial_page_padding_and_request_reuse(graph_replay):
    case = _CopySfaCase(2, [1, 2, 0], torch.bfloat16)
    try:
        batch, heads, width = case.batch, 32, case.max_context_len
        device = case.device
        query = torch.zeros(batch, heads, 128, dtype=case.dtype, device=device)
        query[..., 0] = 1
        key = torch.zeros(
            2 * width // 128, 128, 1, 128, dtype=case.dtype, device=device
        )
        table = torch.stack(
            [
                torch.arange(width // 128, dtype=torch.int32, device=device)
                + req * (width // 128)
                if req < 2
                else torch.zeros(width // 128, dtype=torch.int32, device=device)
                for req in case.batch_req_ids
            ]
        )
        lengths = torch.tensor([2049, 0, 4097], dtype=torch.int32, device=device)
        states = torch.tensor([-2, -3, -2], dtype=torch.int32, device=device)
        pool = torch.full((3, width), -1, dtype=torch.int32, device=device)
        ids = torch.full_like(case.topk_src, -1)
        slots = torch.full_like(ids, -1)
        topk_counts = torch.zeros_like(case.topk_miss_counts)
        miss_src = torch.full((batch, 32768), -1, dtype=torch.int32, device=device)
        miss_dst = torch.full_like(miss_src, -1)
        miss_counts = torch.zeros_like(topk_counts)

        def enqueue():
            torch.ops.npu.fused_lightning_indexer_manage(
                torch.full((batch, heads), 1 / heads, dtype=case.dtype, device=device),
                torch.ones(batch, heads, dtype=torch.float32, device=device),
                query,
                torch.ones(key.shape[:3], dtype=torch.float32, device=device),
                key,
                table,
                case.actual_q,
                lengths,
                lengths,
                torch.full_like(lengths, 2048),
                states,
                case.req_ids,
                pool,
                ids,
                slots,
                topk_counts,
                miss_src,
                miss_dst,
                miss_counts,
            )
            valid = lengths > 0
            case.topk_src.copy_(torch.where(valid[:, None, None], ids, -1))
            case.topk_dst.copy_(torch.where(valid[:, None, None], slots, -1))
            case.topk_miss_counts.copy_(torch.where(valid, topk_counts, 0))
            case.miss_src.copy_(torch.where(valid[:, None], miss_src[:, :2048], -1))
            case.miss_dst.copy_(torch.where(valid[:, None], miss_dst[:, :2048], -1))
            case.miss_counts.copy_(torch.where(valid, miss_counts, 0))
            case._copy_sfa()

        def set_lengths(a, b, reset=False):
            lengths.copy_(torch.tensor([b, 0, a], dtype=torch.int32, device=device))
            key.view(2, width, 128)[..., 0] = -8
            for req, length in ((0, a), (1, b)):
                key.view(2, width, 128)[req, length - 2048 : length, 0] = 8
            states.copy_(
                torch.tensor(
                    [-2 if reset else -1, -3, -2 if reset else -1],
                    dtype=torch.int32,
                    device=device,
                )
            )

        set_lengths(4097, 2049, reset=True)
        enqueue()
        torch.npu.synchronize()
        graph = None
        if graph_replay:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                enqueue()
        for a, b, reset in (
            (4097, 2049, True),
            (4098, 2050, False),
            (4225, 2177, False),
            (3001, 2501, True),
        ):
            set_lengths(a, b, reset)
            graph.replay() if graph is not None else enqueue()
            torch.npu.synchronize()
            expected = {0: list(range(a - 2048, a)), 1: list(range(b - 2048, b))}
            for row, req in ((0, 1), (2, 0)):
                assert set(case.topk_src[row, 0].cpu().tolist()) == set(expected[req])
                physical = case.hbm.cpu().view(-1, 128, 576)
                for src, slot in zip(
                    case.topk_src[row, 0].cpu().tolist(),
                    case.topk_dst[row, 0].cpu().tolist(),
                ):
                    page = req * 16 + ((slot // 128) * 7) % 16
                    torch.testing.assert_close(
                        physical[page, slot % 128],
                        case.logical_kv[req, src],
                        atol=0,
                        rtol=0,
                    )
            case.assert_attention(expected)
            case.assert_invalid_outputs()
            assert torch.all(case.hbm[32:] == 19)
    finally:
        case.close()
