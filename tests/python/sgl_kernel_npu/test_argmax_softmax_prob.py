import contextlib

import pytest
import sgl_kernel_npu.sample.argmax_softmax_prob as mod
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


@pytest.mark.parametrize("bad", [0, -1, 3, 100, 1000])
def test_rejects_a_block_v_that_is_not_a_positive_power_of_two(bad):
    """Without the check the failure is an opaque Triton compilation error
    rather than a message naming the argument."""
    logits = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError, match="positive power of two"):
        argmax_softmax_prob_fused(logits, block_v=bad)


@contextlib.contextmanager
def _record_block_v():
    """Capture the BLOCK_V actually handed to the kernel."""
    seen = []
    real = mod._argmax_prob_kernel

    class _Recorder:
        def __getitem__(self, grid):
            launch = real[grid]

            def call(*args, **kwargs):
                seen.append(kwargs["BLOCK_V"])
                return launch(*args, **kwargs)

            return call

    mod._argmax_prob_kernel = _Recorder()
    try:
        yield seen
    finally:
        mod._argmax_prob_kernel = real


def test_record_block_v_observes_the_launch():
    """Control for _record_block_v: it must see the default width, not nothing."""
    logits = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
    with _record_block_v() as seen:
        argmax_softmax_prob_fused(logits)
    assert seen == [4096], seen


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("V", [17, 4096, 157184])
@pytest.mark.parametrize("block_v", [None, 64, 65536])
def test_the_tile_reaching_the_kernel_is_a_power_of_two(block_v, V, dtype):
    """Rejecting a bad block_v is only half of it: the width the kernel sees is
    the caller's value clamped by the dtype budget and by the vocab, and it is
    that value tl.arange has to accept."""
    logits = torch.randn(2, V, dtype=dtype, device="npu")
    with _record_block_v() as seen:
        argmax_softmax_prob_fused(logits, block_v=block_v)
    (used,) = seen
    assert used > 0 and used & (used - 1) == 0, f"tile is not a power of two: {used}"


@pytest.mark.parametrize("good", [64, 1024, 8192])
def test_accepts_a_power_of_two_block_v(good):
    """A valid override must still produce the reference result, including when
    it is wider than the vocab and gets clamped."""
    logits = torch.randn(8, 4096, dtype=torch.bfloat16, device="npu")
    ref_argmax, ref_prob = argmax_softmax_prob_golden(logits)
    argmax, prob = argmax_softmax_prob_fused(logits, block_v=good)
    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_oversized_power_of_two_block_v_is_capped_by_the_dtype_budget(dtype):
    """A legal power of two can still exceed the unified buffer in the input
    dtype: fp32 block_v=16384 is 64 KiB against a 32 KiB tile, and fails to
    compile. The vocab is wide enough that next_power_of_2(V) does not clamp it
    first, so only the dtype budget can."""
    # 16384 is exactly the bf16 budget, so that arm would be capped by nothing;
    # scale the override with the dtype so both arms actually exceed it.
    over = {torch.float32: 16384, torch.bfloat16: 32768}[dtype]
    logits = torch.randn(4, 32000, dtype=dtype, device="npu")
    ref_argmax, ref_prob = argmax_softmax_prob_golden(logits)
    argmax, prob = argmax_softmax_prob_fused(logits, block_v=over)
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
