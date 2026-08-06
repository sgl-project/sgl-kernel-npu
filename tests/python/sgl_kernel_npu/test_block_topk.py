import unittest

import torch
from sgl_kernel_npu.moe.block_topk import block_topk_cann_hybrid


def _block_topk_reference(
    router_logits,
    correction_bias,
    block_size,
    expert_capacity,
    top_k,
):
    num_tokens, num_experts = router_logits.shape
    base_scores = torch.sigmoid(router_logits.float())
    routing_scores = base_scores + correction_bias.float()
    tie_break = -torch.arange(num_experts, dtype=torch.float32) * 3e-7
    result_ids = []
    result_weights = []

    for start in range(0, num_tokens, block_size):
        block_scores = routing_scores[start : start + block_size]
        capacity_ids = (
            (block_scores.max(dim=0).values + tie_break).topk(expert_capacity).indices
        )
        token_scores = block_scores[:, capacity_ids] + tie_break[capacity_ids]
        local_ids = token_scores.topk(top_k, dim=-1).indices
        selected_ids = capacity_ids[local_ids]
        selected_scores = base_scores[start : start + block_size].gather(
            1, selected_ids
        )
        result_ids.append(selected_ids)
        result_weights.append(
            selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        )

    return torch.cat(result_weights), torch.cat(result_ids).to(torch.int32)


class TestBlockTopK(unittest.TestCase):
    def test_matches_reference(self):
        torch.manual_seed(2)
        num_tokens = 63
        num_experts = 256
        block_size = 32
        expert_capacity = 48
        top_k = 8
        logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)
        bias = torch.randn(num_experts, dtype=torch.float32) * 0.1

        expected_weights, expected_ids = _block_topk_reference(
            logits,
            bias,
            block_size,
            expert_capacity,
            top_k,
        )
        weights, ids = block_topk_cann_hybrid(
            logits.to("npu"),
            bias.to("npu"),
            block_size,
            expert_capacity,
            top_k,
        )

        torch.testing.assert_close(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.float().cpu(),
            expected_weights,
            rtol=5e-3,
            atol=5e-4,
        )

    def test_zero_score_fallback(self):
        logits = torch.full(
            (32, 16),
            float("-inf"),
            dtype=torch.bfloat16,
            device="npu",
        )
        bias = torch.zeros(16, dtype=torch.float32, device="npu")

        weights, ids = block_topk_cann_hybrid(
            logits,
            bias,
            block_size=32,
            expert_capacity=8,
            top_k=4,
        )

        torch.testing.assert_close(
            weights.float().cpu(),
            torch.full((32, 4), 0.25),
        )
        self.assertEqual(ids.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
