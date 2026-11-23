"""Both outputs of the fused four-stream HC prefill kernel use BF16 residuals."""
import pytest
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_prefill_combine_norm
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware


@pytest.mark.skipif(get_hardware().generation is not Generation.ARCH22, reason="HC prefill requires ARCH22")
@pytest.mark.parametrize("rows,shared", [(512, True), (513, False)])
def test_hc_prefill_combine_norm(rows, shared):
    rng = torch.Generator().manual_seed(31)
    def bf16(*shape):
        return (0.2 * torch.randn(*shape, generator=rng, device="cpu")).bfloat16()

    residual = bf16(rows, 10240)
    block_output = bf16(rows, 2560)
    logits = torch.randn(rows, 4, generator=rng, device="cpu")
    weights = bf16(1 if shared else 4, 2560)
    out, normalized = hc_prefill_combine_norm(
        residual.npu(), block_output.npu(), logits.npu(), weights.npu(), 1e-6, 4
    )
    gate = 2 * (logits.double() / 4).sigmoid()
    combined = (
        residual.double().view(rows, 4, 2560) + gate[..., None] * block_output.double()[:, None]
    ).bfloat16()
    x = combined.double()
    ref = (x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + weights.double())).bfloat16()
    torch.testing.assert_close(out.cpu(), combined.flatten(1), atol=0.005, rtol=0.01)
    torch.testing.assert_close(normalized.cpu(), ref.flatten(1), atol=0.005, rtol=0.01)
