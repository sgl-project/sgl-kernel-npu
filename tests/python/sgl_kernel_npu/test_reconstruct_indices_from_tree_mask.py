import logging

import torch
from sgl_kernel_npu.speculative import (
    reconstruct_indices_from_tree_mask_torch_wrapper,
    reconstruct_indices_from_tree_mask_triton,
)

logger = logging.getLogger(__name__)


def test_alignment():
    logger.info(
        "=== reconstruct_indices_from_tree_mask_triton vs reconstruct_indices_from_tree_mask_torch_wrapper ==="
    )

    test_bs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    D = 4

    for bs in test_bs_list:
        logger.info(f"\nTest case: batch_size={bs}, draft_token_num={D}")

        tree_mask = torch.rand(bs, D, D, device="npu") < 0.3
        i_indices = torch.arange(D, device="npu").reshape(1, D, 1).expand(bs, D, D)
        t_indices = torch.arange(D, device="npu").reshape(1, 1, D).expand(bs, D, D)
        tree_mask = tree_mask & (i_indices < t_indices)

        verified_seq_len = torch.randint(
            1, 1000, (bs,), dtype=torch.int64, device="npu"
        )

        ri_torch = torch.empty(bs * D, dtype=torch.int64, device="npu")
        pos_torch = torch.empty(bs * D, dtype=torch.int64, device="npu")
        ntk_torch = torch.empty(bs * D, dtype=torch.int64, device="npu")
        nsb_torch = torch.empty(bs * D, dtype=torch.int64, device="npu")

        ri_triton = torch.empty_like(ri_torch)
        pos_triton = torch.empty_like(pos_torch)
        ntk_triton = torch.empty_like(ntk_torch)
        nsb_triton = torch.empty_like(nsb_torch)

        reconstruct_indices_from_tree_mask_torch_wrapper(
            tree_mask,
            verified_seq_len,
            pos_torch,
            ri_torch,
            ntk_torch,
            nsb_torch,
            bs,
            D,
        )
        reconstruct_indices_from_tree_mask_triton(
            tree_mask,
            verified_seq_len,
            pos_triton,
            ri_triton,
            ntk_triton,
            nsb_triton,
            bs,
            D,
        )

        assert torch.all(ri_torch == ri_triton), f"retrive_index diff"
        assert torch.all(pos_torch == pos_triton), f"positions diff"
        assert torch.all(ntk_torch == ntk_triton), f"retrive_next_token diff"
        assert torch.all(nsb_torch == nsb_triton), f"retrive_next_sibling diff"

        logger.info("all close")

    logger.info("\n=== Success ===")
