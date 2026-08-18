import pytest
import torch
import torch.nn.functional as F

import sgl_kernel_npu  # noqa: F401


HC_MULT = 4
HIDDEN_SIZE = 3584
MIX_SIZE = HC_MULT * (HC_MULT + 2)


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.contiguous().view(torch.int32)
    return (bits & (~((1 << 13) - 1))).view(torch.float32)


def _reference_pre(residual, fn, scale, base, sinkhorn_iters=20):
    residual_fp32 = residual.float()
    residual_flat = residual_fp32.flatten(1)
    mixes = F.linear(_to_hf32(residual_flat), _to_hf32(fn))
    mixes *= torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    pre, post, comb = mixes.split([HC_MULT, HC_MULT, HC_MULT**2], dim=-1)
    pre = torch.sigmoid(pre * scale[0] + base[:HC_MULT]) + 1e-6
    post = 2.0 * torch.sigmoid(
        post * scale[1] + base[HC_MULT : 2 * HC_MULT]
    )
    comb = (
        comb * scale[2] + base[2 * HC_MULT :]
    ).unflatten(-1, (HC_MULT, HC_MULT))
    comb = comb.softmax(dim=-1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    layer_input = (pre.unsqueeze(-1) * residual_fp32).sum(dim=1)
    return layer_input.to(torch.bfloat16), post, comb


@pytest.mark.parametrize("num_tokens", [1, 8, 129])
def test_hc_pre_post_telechat4(num_tokens):
    torch.manual_seed(20260807 + num_tokens)
    residual = torch.randn(
        num_tokens, HC_MULT, HIDDEN_SIZE, dtype=torch.bfloat16
    )
    fn = torch.randn(MIX_SIZE, HC_MULT * HIDDEN_SIZE) / (
        HC_MULT * HIDDEN_SIZE
    ) ** 0.5
    scale = torch.tensor([0.8, 1.1, 0.7], dtype=torch.float32)
    base = torch.randn(MIX_SIZE, dtype=torch.float32) * 0.1

    expected_pre = _reference_pre(residual, fn, scale, base)
    actual_pre = torch.ops.npu.hc_pre(
        residual.npu(),
        fn.npu(),
        scale.npu(),
        base.npu(),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )

    torch.testing.assert_close(
        actual_pre[0].cpu(), expected_pre[0], atol=0.03125, rtol=5e-3
    )
    torch.testing.assert_close(
        actual_pre[1].cpu(), expected_pre[1], atol=5e-4, rtol=2e-3
    )
    torch.testing.assert_close(
        actual_pre[2].cpu(), expected_pre[2], atol=5e-4, rtol=2e-3
    )

    x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    post = actual_pre[1].cpu().to(torch.bfloat16)
    comb = actual_pre[2].cpu().to(torch.bfloat16)
    expected_post = (
        post.float().unsqueeze(-1) * x.float().unsqueeze(1)
        + (comb.float().unsqueeze(-1) * residual.float().unsqueeze(2)).sum(
            dim=1
        )
    ).to(torch.bfloat16)
    actual_post = torch.ops.npu.hc_post(
        x.npu(), residual.npu(), post.npu(), comb.npu()
    )
    torch.testing.assert_close(
        actual_post.cpu(), expected_post, atol=0.03125, rtol=5e-3
    )


def test_hc_meta_accepts_3584_and_rejects_unknown_hidden_size():
    def make_inputs(hidden_size):
        return (
            torch.empty(
                2, HC_MULT, hidden_size, dtype=torch.bfloat16, device="meta"
            ),
            torch.empty(MIX_SIZE, HC_MULT * hidden_size, device="meta"),
            torch.empty(3, device="meta"),
            torch.empty(MIX_SIZE, device="meta"),
        )

    layer_input, post, comb = torch.ops.npu.hc_pre(
        *make_inputs(HIDDEN_SIZE),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )
    assert layer_input.shape == (2, HIDDEN_SIZE)
    assert post.shape == (2, HC_MULT)
    assert comb.shape == (2, HC_MULT, HC_MULT)

    with pytest.raises(RuntimeError, match="3584, 4096 or 7168"):
        torch.ops.npu.hc_pre(
            *make_inputs(2048),
            hc_mult=HC_MULT,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
