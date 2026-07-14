import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FUSED_SIGMOID_GATING_RECURRENT = (
    ROOT
    / "python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent.py"
)


class TestGDNVerifyDecodeParity(unittest.TestCase):
    def test_decode_h0_state_uses_dv_major_layout(self):
        source = FUSED_SIGMOID_GATING_RECURRENT.read_text(encoding="utf-8")
        compact = "".join(source.split())

        self.assertNotIn("+o_k[:,None]*V+o_v[None,:]", compact)
        self.assertEqual(compact.count("+o_v[None,:]*K+o_k[:,None]"), 2)

    def test_verify_matches_serial_decode_with_dk128_dv64(self):
        import torch

        try:
            import torch_npu  # noqa: F401
        except ImportError:
            self.skipTest("requires torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("requires NPU")

        import sgl_kernel_npu  # noqa: F401
        from sgl_kernel_npu.fla.fused_gdn_gating import (
            fused_gdn_gating_kernel_without_sigmoid,
        )
        from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update_npu,
        )

        torch.manual_seed(3)
        device = "npu"
        dtype = torch.bfloat16
        batch_size, mtp = 4, 4
        num_heads, num_value_heads = 4, 8
        dk, dv = 128, 64

        cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        num_accepted_tokens = torch.tensor(
            [1, 2, 4, 3], dtype=torch.int32, device=device
        )
        actual_seq_lengths = torch.full(
            (batch_size,), mtp, dtype=torch.int32, device=device
        )
        ssm_state_indices = torch.arange(
            batch_size * mtp, dtype=torch.int32, device=device
        ).view(batch_size, mtp)

        q = torch.randn(
            batch_size, mtp, num_heads, dk, dtype=dtype, device=device
        ) * 0.1
        k = torch.randn(
            batch_size, mtp, num_heads, dk, dtype=dtype, device=device
        ) * 0.1
        v = torch.randn(
            batch_size, mtp, num_value_heads, dv, dtype=dtype, device=device
        ) * 0.1
        a = torch.randn(
            batch_size, mtp, num_value_heads, dtype=dtype, device=device
        ) * 0.1
        b = torch.randn(
            batch_size, mtp, num_value_heads, dtype=dtype, device=device
        ) * 0.1
        a_log = torch.full((num_value_heads,), -2.0, dtype=torch.float32, device=device)
        dt_bias = torch.zeros(num_value_heads, dtype=torch.float32, device=device)
        recurrent_state = (
            torch.randn(
                batch_size, num_value_heads, dv, dk, dtype=dtype, device=device
            )
            * 0.01
        )
        intermediate_seed = (
            torch.randn(
                batch_size, mtp, num_value_heads, dv, dk, dtype=dtype, device=device
            )
            * 0.01
        )
        intermediate_seed[:, 0] = recurrent_state[cache_indices.long()]

        mix_qkv = torch.cat(
            [
                q.reshape(batch_size, mtp, num_heads * dk),
                k.reshape(batch_size, mtp, num_heads * dk),
                v.reshape(batch_size, mtp, num_value_heads * dv),
            ],
            dim=-1,
        ).contiguous()
        g, beta = fused_gdn_gating_kernel_without_sigmoid(
            a_log,
            a.reshape(-1, num_value_heads),
            b.reshape(-1, num_value_heads),
            dt_bias,
        )

        intermediate_verify = intermediate_seed.clone()
        out_verify = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state.clone(),
            beta=beta.view(batch_size, mtp, num_value_heads).to(dtype),
            scale=dk**-0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=intermediate_verify.view(
                -1, num_value_heads, dv, dk
            ),
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g.view(batch_size, mtp, num_value_heads).float(),
        )

        out_decode = torch.empty_like(out_verify)
        intermediate_decode = torch.empty_like(intermediate_seed)
        q_decode = q.reshape(batch_size * mtp, num_heads, dk).unsqueeze(0)
        k_decode = k.reshape(batch_size * mtp, num_heads, dk).unsqueeze(0)
        v_decode = v.reshape(batch_size * mtp, num_value_heads, dv).unsqueeze(0)
        a_decode = a.reshape(batch_size * mtp, num_value_heads)
        b_decode = b.reshape(batch_size * mtp, num_value_heads)
        cu_one_token = torch.tensor([0, 1], dtype=torch.int32, device=device)
        state_index = torch.zeros(1, dtype=torch.int64, device=device)

        for request_idx in range(batch_size):
            accepted = int(num_accepted_tokens[request_idx].item())
            state = torch.empty(
                1, num_value_heads, dv, dk, dtype=dtype, device=device
            )
            if accepted == 1:
                state[0] = recurrent_state[cache_indices[request_idx].long()]
            else:
                state[0] = intermediate_seed[request_idx, accepted - 1]

            for step_idx in range(mtp):
                pos = request_idx * mtp + step_idx
                out = fused_sigmoid_gating_delta_rule_update_npu(
                    A_log=a_log,
                    a=a_decode[pos : pos + 1],
                    dt_bias=dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=q_decode[:, pos : pos + 1],
                    k=k_decode[:, pos : pos + 1],
                    v=v_decode[:, pos : pos + 1],
                    b=b_decode[pos : pos + 1],
                    initial_state_source=state,
                    initial_state_indices=state_index,
                    scale=dk**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_one_token,
                )
                out_decode[request_idx, step_idx] = out[0, 0]
                intermediate_decode[request_idx, step_idx] = state[0]

        eps = 2**-8
        self.assertTrue(torch.isfinite(out_verify).all())
        self.assertTrue(torch.isfinite(out_decode).all())
        self.assertTrue(torch.isfinite(intermediate_verify).all())
        self.assertTrue(torch.isfinite(intermediate_decode).all())
        torch.testing.assert_close(out_verify, out_decode, atol=eps, rtol=eps)
        torch.testing.assert_close(
            intermediate_verify.view(batch_size, mtp, num_value_heads, dv, dk),
            intermediate_decode,
            atol=eps,
            rtol=eps,
        )


if __name__ == "__main__":
    unittest.main()
