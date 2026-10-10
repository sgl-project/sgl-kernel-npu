import pytest
import torch
from sgl_kernel_npu.sample.probability import (
    _encode_keep_keys,
    _renorm_from_sorted_probs,
    top_k_renorm_prob,
    top_p_renorm_prob,
)


def test_top_k_top_p_renorm_matches_sequential_reference():
    torch.manual_seed(7)
    probs = torch.softmax(torch.randn(4, 97), dim=-1)
    top_ks = torch.tensor([1, 7, 31, 97])
    top_ps = torch.tensor([0.3, 0.75, 0.95, 1.0])

    actual = top_p_renorm_prob(top_k_renorm_prob(probs, top_ks), top_ps)

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    positions = torch.arange(probs.shape[-1]).view(1, -1)
    sorted_probs[positions >= top_ks.view(-1, 1)] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    top_k_probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    sorted_probs, sorted_indices = top_k_probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[cumulative - sorted_probs > top_ps.view(-1, 1)] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_renorm_accepts_scalar_thresholds():
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])

    top_k_actual = top_k_renorm_prob(probs, 2)
    top_k_expected = torch.tensor(
        [[4.0 / 7.0, 3.0 / 7.0, 0.0, 0.0], [0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0]]
    )
    torch.testing.assert_close(top_k_actual, top_k_expected)

    top_p_actual = top_p_renorm_prob(probs, 0.6)
    torch.testing.assert_close(top_p_actual, top_k_expected)


@pytest.fixture(params=["cpu", "npu"])
def probability_device(request):
    if request.param == "npu" and (
        not hasattr(torch, "npu") or not torch.npu.is_available()
    ):
        pytest.skip("NPU is not available")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["edges", "random"])
@pytest.mark.parametrize(
    "filter_kind,threshold",
    [("p", p) for p in [0.0, 0.25, 0.5, 0.95, 1.0]] + [("k", k) for k in [1, 2, 4]],
)
def test_renorm_preserves_sorted_selection(
    probability_device, dtype, case, filter_kind, threshold
):
    if case == "edges":
        probs = torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.5, 0.0, 0.5],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=probability_device,
        )
    else:
        generator = torch.Generator().manual_seed(7)
        probs = torch.softmax(torch.randn(4, 97, generator=generator), dim=-1)
        probs = probs.to(device=probability_device, dtype=dtype)

    original_probs = probs.clone()
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    if filter_kind == "p":
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_probs.masked_fill_(cumulative - sorted_probs > threshold, 0.0)
    else:
        positions = torch.arange(probs.shape[-1], device=probability_device)
        sorted_probs.masked_fill_(positions >= threshold, 0.0)

    # Share the selected permutation so ties cannot differ between two sorts.
    reference_probs = sorted_probs.clone()
    reference_probs.div_(reference_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    expected = torch.zeros_like(probs).scatter_(-1, sorted_indices, reference_probs)
    actual = _renorm_from_sorted_probs(probs, sorted_probs, sorted_indices)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(probs, original_probs, rtol=0, atol=0)


def test_keep_keys_reject_inexact_vocab_size():
    # Expanded views exercise the limit without allocating a full vocabulary.
    sorted_probs = torch.tensor([[1.0]]).expand(1, 2**23 + 1)
    sorted_indices = torch.tensor([[0]]).expand_as(sorted_probs)
    with pytest.raises(ValueError, match="vocab_size <="):
        _encode_keep_keys(sorted_probs, sorted_indices)
