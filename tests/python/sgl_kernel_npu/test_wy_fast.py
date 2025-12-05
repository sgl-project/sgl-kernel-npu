import math
from typing import Optional, Tuple

import torch
from sgl_kernel_npu.fla.wy_fast import recompute_w_u_fwd_npu
from torch_npu.testing.testcase import TestCase, run_tests

device = "npu"


class TestRecomputeWUFwd(TestCase):
    def recompute_w_u_fwd_ref(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: torch.Tensor,
        A: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        is_varlen = cu_seqlens is not None
        B = k.shape[0]
        max_seqlen = k.shape[1]
        Hg = k.shape[2]
        K = k.shape[3]
        H = v.shape[2]
        V = v.shape[3]
        BT = A.shape[-1]
        device = k.device
        dtype = k.dtype

        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, B * max_seqlen + 1, step=max_seqlen, dtype=torch.int32, device=device
            )
            is_varlen = False

        num_seq = len(cu_seqlens) - 1
        assert len(cu_seqlens) == num_seq + 1

        group_size = H // Hg

        w = k.new_empty(B, max_seqlen, H, K)
        u = v.new_empty(B, max_seqlen, H, V)

        for seq_idx in range(num_seq):
            bos = cu_seqlens[seq_idx].item()
            eos = cu_seqlens[seq_idx + 1].item()
            seq_len = eos - bos
            if seq_len == 0:
                continue

            NT = math.ceil(seq_len / BT)

            if is_varlen:
                beta_b = beta[0, bos:eos]
                g_cumsum_b = g_cumsum[0, bos:eos]
                A_b = A[0, bos:eos]
                k_b = k[0, bos:eos]
                v_b = v[0, bos:eos]
                b_idx = 0
                time_base = bos
            else:
                beta_b = beta[seq_idx, :seq_len]
                g_cumsum_b = g_cumsum[seq_idx, :seq_len]
                A_b = A[seq_idx, :seq_len]
                k_b = k[seq_idx, :seq_len]
                v_b = v[seq_idx, :seq_len]
                b_idx = seq_idx
                time_base = 0

            for i_t in range(NT):
                start = i_t * BT
                end = min(start + BT, seq_len)
                cur_bt = end - start

                pad_beta = torch.zeros(BT, H, dtype=beta_b.dtype, device=device)
                pad_beta[:cur_bt] = beta_b[start:end]
                pad_beta_f32 = pad_beta.to(torch.float32)

                pad_g_cumsum = torch.zeros(BT, H, dtype=g_cumsum_b.dtype, device=device)
                pad_g_cumsum[:cur_bt] = g_cumsum_b[start:end]
                pad_g_f32 = torch.exp(pad_g_cumsum.to(torch.float32))

                pad_A = torch.zeros(BT, H, BT, dtype=A_b.dtype, device=device)
                pad_A[:cur_bt] = A_b[start:end]
                pad_A_f32 = pad_A.to(torch.float32)

                pad_v = torch.zeros(BT, H, V, dtype=v_b.dtype, device=device)
                pad_v[:cur_bt] = v_b[start:end]
                pad_v_f32 = pad_v.to(torch.float32)

                pad_k = torch.zeros(BT, Hg, K, dtype=k_b.dtype, device=device)
                pad_k[:cur_bt] = k_b[start:end]
                pad_k_f32 = pad_k.to(torch.float32)

                for h in range(H):
                    i_g = h // group_size

                    beta_p = pad_beta_f32[:, h]
                    g_p = pad_g_f32[:, h]
                    A_p = pad_A_f32[:, h, :]
                    v_p = pad_v_f32[:, h, :]
                    k_p = pad_k_f32[:, i_g, :]

                    vb_p = v_p * beta_p.unsqueeze(1)
                    kb_p = k_p * beta_p.unsqueeze(1) * g_p.unsqueeze(1)

                    u_p = A_p @ vb_p
                    w_p = A_p @ kb_p

                    store_time = time_base + start
                    t0, t1 = store_time, store_time + cur_bt

                    u[b_idx, t0:t1, h, :] = u_p[:cur_bt].to(v.dtype)
                    w[b_idx, t0:t1, h, :] = w_p[:cur_bt].to(dtype)

        return w, u

    def test_recompute_w_u_fwd(self):
        # test data config
        B = 1
        H = 8
        Hg = 8
        K = 128
        V = 128
        BT = 64

        dtype = torch.bfloat16

        # generate data
        seq_lens = [10, 25, 40]
        T_total = sum(seq_lens)

        # cu_seqlens = [0, len1, len1+len2, len1+len2+len3]
        cu = [0]
        for l in seq_lens:
            cu.append(cu[-1] + l)
        cu_seqlens = torch.tensor(cu, dtype=torch.long, device=device)
        T = T_total

        k = torch.randn(B, T, Hg, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        beta = torch.randn(B, T, H, dtype=dtype, device=device)
        g_cumsum = torch.randn(B, T, H, dtype=dtype, device=device)
        A = torch.randn(B, T, H, BT, dtype=dtype, device=device).contiguous()

        w_npu, u_npu = recompute_w_u_fwd_npu(
            k=k,
            v=v,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            cu_seqlens=cu_seqlens,
        )

        w_ref, u_ref = self.recompute_w_u_fwd_ref(
            k=k,
            v=v,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            cu_seqlens=cu_seqlens,
        )

        self.assertTrue(
            torch.allclose(w_npu.cpu(), w_ref.cpu(), atol=0.001, rtol=0.001),
            f"w mismatch. max diff: {torch.max(torch.abs(w_npu.cpu() - w_ref.cpu()))}",
        )

        self.assertTrue(
            torch.allclose(u_npu.cpu(), u_ref.cpu(), atol=0.001, rtol=0.001),
            f"u mismatch. max diff: {torch.max(torch.abs(u_npu.cpu() - u_ref.cpu()))}",
        )


if __name__ == "__main__":
    run_tests()
