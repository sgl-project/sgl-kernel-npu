import numpy as np
import torch
from sgl_kernel_npu.moe.fused_remap_deepep import fused_remap_deepep


def _reference_fused_remap_deepep(
    topk_ids,
    topk_weights,
    num_fused_shared_experts,
    num_physical_routed_experts,
    ep_rank,
    ep_size,
    routed_scaling_factor,
):
    N, K = topk_ids.shape
    num_local_routed = num_physical_routed_experts // ep_size
    num_local_experts = num_local_routed + num_fused_shared_experts

    out_ids = torch.zeros(N, K + num_fused_shared_experts, dtype=topk_ids.dtype)
    out_weights = torch.zeros(N, K + num_fused_shared_experts, dtype=topk_weights.dtype)

    shared_weight = 1.0 if not routed_scaling_factor else 1.0 / routed_scaling_factor

    for row in range(N):
        for col in range(K + num_fused_shared_experts):
            if col < K:
                routed_id = topk_ids[row, col].item()
                out_ids[row, col] = routed_id + routed_id // num_local_routed
                out_weights[row, col] = topk_weights[row, col].item()
            else:
                shared_idx = col - K
                out_ids[row, col] = (
                    ep_rank * num_local_experts + num_local_routed + shared_idx
                )
                out_weights[row, col] = shared_weight

    return out_ids, out_weights


def test_fused_remap_deepep_basic():
    N, K = 4, 6
    num_fused_shared = 1
    num_physical_routed = 256
    ep_rank = 0
    ep_size = 16
    routed_scaling_factor = 2.5

    topk_ids = torch.randint(0, 16, (N, K), dtype=torch.int32).npu()
    topk_weights = torch.rand(N, K, dtype=torch.float32).npu()

    out_ids, out_weights = fused_remap_deepep(
        topk_ids,
        topk_weights,
        num_fused_shared_experts=num_fused_shared,
        num_physical_routed_experts=num_physical_routed,
        ep_rank=ep_rank,
        ep_size=ep_size,
        routed_scaling_factor=routed_scaling_factor,
    )

    ref_ids, ref_weights = _reference_fused_remap_deepep(
        topk_ids.cpu(),
        topk_weights.cpu(),
        num_fused_shared,
        num_physical_routed,
        ep_rank,
        ep_size,
        routed_scaling_factor,
    )

    np.testing.assert_array_equal(out_ids.cpu().numpy(), ref_ids.numpy())
    np.testing.assert_allclose(
        out_weights.cpu().numpy(),
        ref_weights.numpy(),
        rtol=1e-6,
    )


def test_fused_remap_deepep_multi_shared():
    N, K = 8, 6
    num_fused_shared = 3
    num_physical_routed = 256
    ep_rank = 3
    ep_size = 16
    routed_scaling_factor = 0.0

    topk_ids = torch.randint(0, 16, (N, K), dtype=torch.int32).npu()
    topk_weights = torch.rand(N, K, dtype=torch.float32).npu()

    out_ids, out_weights = fused_remap_deepep(
        topk_ids,
        topk_weights,
        num_fused_shared_experts=num_fused_shared,
        num_physical_routed_experts=num_physical_routed,
        ep_rank=ep_rank,
        ep_size=ep_size,
        routed_scaling_factor=routed_scaling_factor,
    )

    ref_ids, ref_weights = _reference_fused_remap_deepep(
        topk_ids.cpu(),
        topk_weights.cpu(),
        num_fused_shared,
        num_physical_routed,
        ep_rank,
        ep_size,
        routed_scaling_factor,
    )

    np.testing.assert_array_equal(out_ids.cpu().numpy(), ref_ids.numpy())
    np.testing.assert_allclose(
        out_weights.cpu().numpy(),
        ref_weights.numpy(),
        rtol=1e-6,
    )


def test_fused_remap_deepep_empty():
    topk_ids = torch.zeros(0, 6, dtype=torch.int32).npu()
    topk_weights = torch.zeros(0, 6, dtype=torch.float32).npu()

    out_ids, out_weights = fused_remap_deepep(
        topk_ids,
        topk_weights,
        num_fused_shared_experts=1,
        num_physical_routed_experts=256,
        ep_rank=0,
        ep_size=16,
        routed_scaling_factor=2.5,
    )

    assert out_ids.shape[0] == 0
    assert out_weights.shape[0] == 0


if __name__ == "__main__":
    test_fused_remap_deepep_basic()
    test_fused_remap_deepep_multi_shared()
    test_fused_remap_deepep_empty()
    print("All tests passed!")
