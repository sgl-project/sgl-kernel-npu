"""NPU precision checks for fused filtering and packed-key restoration."""

import pytest
import torch
from sgl_kernel_npu.sample import probability


@pytest.fixture
def npu_device():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")
    return "npu"


@pytest.mark.parametrize("division", ["native", "rn"])
@pytest.mark.parametrize("vocab", [4, 97, 2051])
@pytest.mark.parametrize("mode", ["k", "p"])
@pytest.mark.parametrize("scalar", [False, True])
def test_fused_renorm_matches_shared_sort(npu_device, division, vocab, mode, scalar):
    from sgl_kernel_npu.sample.probability_triton import filter_and_renorm

    generator = torch.Generator().manual_seed(17)
    probs = torch.softmax(torch.randn(5, vocab, generator=generator), dim=-1)
    # Include ties, zero mass, one-hot and a non-block-aligned vocabulary.
    probs[0].fill_(1.0 / vocab)
    probs[1].zero_()
    probs[2].zero_()
    probs[2, -1] = 1.0
    probs = probs.to(npu_device)
    original = probs.clone()
    values, indices = probs.sort(dim=-1, descending=True)
    if mode == "p":
        thresholds = torch.tensor([0.0, 0.25, 0.5, 0.95, 1.0], device=npu_device)
        cumulative = values.cumsum(dim=-1)
    else:
        thresholds = torch.tensor(
            [1, 2, 3, vocab - 1, vocab], device=npu_device, dtype=torch.long
        )
        cumulative = None
    if scalar:
        # Expanded scalar parameters have stride 0, unlike per-row thresholds.
        thresholds = thresholds[3].expand(5)

    filtered = values.clone()
    if mode == "p":
        filtered.masked_fill_(cumulative - filtered > thresholds[:, None], 0.0)
    else:
        positions = torch.arange(vocab, device=npu_device)
        filtered.masked_fill_(positions >= thresholds[:, None], 0.0)
    filtered.div_(filtered.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    expected = torch.zeros_like(probs).scatter_(-1, indices, filtered)

    actual = filter_and_renorm(
        probs, values, indices, thresholds, cumulative, division=division
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(probs, original, rtol=0, atol=0)


@pytest.mark.parametrize("division", ["native", "rn"])
def test_public_top_k_then_top_p_dispatch(npu_device, division, monkeypatch):
    from sgl_kernel_npu.sample import probability_triton

    # Unique positive values avoid separate-sort tie ambiguity in this test.
    probs = torch.arange(1, 98, device=npu_device, dtype=torch.float32)[None, :]
    probs = (probs / probs.sum()).repeat(3, 1)
    top_ks = torch.tensor([1, 31, 97], device=npu_device)
    top_ps = torch.tensor([0.0, 0.95, 1.0], device=npu_device)
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_TRITON", "0")
    expected = probability.top_p_renorm_prob(
        probability.top_k_renorm_prob(probs, top_ks), top_ps
    )

    calls = []
    original = probability_triton.filter_and_renorm

    def record_call(*args, **kwargs):
        calls.append(kwargs["division"])
        return original(*args, **kwargs)

    monkeypatch.setattr(probability_triton, "filter_and_renorm", record_call)
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_TRITON", "1")
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_TRITON_DIV", division)
    actual = probability.top_p_renorm_prob(
        probability.top_k_renorm_prob(probs, top_ks), top_ps
    )
    assert calls == [division, division]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
