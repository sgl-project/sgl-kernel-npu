import pytest
import torch
from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused


def argmax_softmax_prob_golden(logits: torch.Tensor):
    """Reference: argmax id and the softmax probability of that id, in fp32."""
    ref = logits.float()
    argmax = ref.argmax(dim=-1)
    prob = torch.softmax(ref, dim=-1).gather(1, argmax.unsqueeze(1)).squeeze(1)
    return argmax, prob


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(32, 4096), (128, 157184), (1024, 32000)])
def test_argmax_softmax_prob(shape, dtype):
    torch.manual_seed(0)
    B, V = shape
    logits = torch.randn(B, V, dtype=dtype, device="npu")

    ref_argmax, ref_prob = argmax_softmax_prob_golden(logits)
    argmax, prob = argmax_softmax_prob_fused(logits)

    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


def test_small_vocab_does_not_overflow_the_tile():
    """A vocab below the default tile must clamp BLOCK_V, not allocate past it."""
    logits = torch.randn(8, 17, dtype=torch.float32, device="npu")

    ref_argmax, ref_prob = argmax_softmax_prob_golden(logits)
    argmax, prob = argmax_softmax_prob_fused(logits)

    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


def test_row_stride_is_honoured():
    """A vocab-truncated view is passed without a copy, so a row stride wider
    than the vocab must still address rows correctly."""
    padded = torch.randn(16, 4096, dtype=torch.bfloat16, device="npu")
    view = padded[:, :3000]
    assert view.stride(0) == 4096 and view.stride(1) == 1

    ref_argmax, ref_prob = argmax_softmax_prob_golden(view)
    argmax, prob = argmax_softmax_prob_fused(view)

    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


def test_ties_keep_the_lower_index():
    """Matches torch.argmax: on an exact tie the earlier index wins, including
    across the kernel's chunk boundary."""
    logits = torch.full((4, 40000), -5.0, dtype=torch.float32, device="npu")
    logits[:, 100] = 7.0
    logits[:, 30000] = 7.0

    argmax, _ = argmax_softmax_prob_fused(logits)

    assert torch.equal(argmax, torch.full((4,), 100, dtype=torch.int64, device="npu"))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
