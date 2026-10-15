"""A5 smoke test for the quantized shared-KV attention sparse-index path."""

import math
import unittest

import sgl_kernel_npu  # noqa: F401 - registers the custom operators
import torch
import torch_npu


class TestKvQuantSparseAttnSharedkv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        if "950" not in torch_npu.npu.get_device_name(0).lower():
            raise unittest.SkipTest("the quantized kernel requires Ascend 950")
        torch_npu.npu.set_device(0)

    def test_tnd_ori_sparse_indices(self):
        device = torch.device("npu:0")
        # Every packed KV byte is zero, including the BF16 RoPE component and
        # FP8 NoPE component. The attention result must therefore be zero.
        packed_kv = (
            torch.zeros((1, 16, 1, 640), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(device)
        )
        q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device=device)
        cu_q_cpu = torch.tensor([0, 1], dtype=torch.int32)
        seq_kv_cpu = torch.tensor([16], dtype=torch.int32)
        cu_q = cu_q_cpu.to(device)
        seq_kv = seq_kv_cpu.to(device)
        sparse_indices = torch.full((1, 1, 512), -1, dtype=torch.int32)
        sparse_indices[0, 0, :4] = torch.arange(4, dtype=torch.int32)

        metadata = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            layout_q="TND",
            layout_kv="PA_ND",
            cu_seqlens_q=cu_q_cpu,
            seqused_kv=seq_kv_cpu,
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_kv=16,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=15,
            ori_win_right=0,
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        output, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(
            q,
            ori_kv=packed_kv,
            ori_sparse_indices=sparse_indices.to(device),
            ori_block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
            cu_seqlens_q=cu_q,
            seqused_q=torch.tensor([1], dtype=torch.int32, device=device),
            seqused_kv=seq_kv,
            sinks=torch.full((64,), -1.0e4, dtype=torch.float32, device=device),
            metadata=metadata,
            softmax_scale=1.0 / math.sqrt(512),
            cmp_ratio=1,
            ori_win_left=15,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        torch_npu.npu.synchronize()
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output.cpu().float(), torch.zeros_like(q.cpu().float()))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.numel(), 0)


if __name__ == "__main__":
    unittest.main()
