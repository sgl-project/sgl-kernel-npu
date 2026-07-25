import torch

from sgl_kernel_npu.sample.probability import top_k_top_p_renorm_probs


def test_top_k_top_p_renorm_matches_sequential_reference():
    torch.manual_seed(7)
    probs = torch.softmax(torch.randn(4, 97), dim=-1)
    top_ks = torch.tensor([1, 7, 31, 97])
    top_ps = torch.tensor([0.3, 0.75, 0.95, 1.0])

    actual = top_k_top_p_renorm_probs(
        probs, top_ks, top_ps, True, True
    )

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    positions = torch.arange(probs.shape[-1]).view(1, -1)
    sorted_probs[positions >= top_ks.view(-1, 1)] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    top_k_probs = torch.zeros_like(probs).scatter(
        -1, sorted_indices, sorted_probs
    )
    sorted_probs, sorted_indices = top_k_probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[cumulative - sorted_probs > top_ps.view(-1, 1)] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(probs).scatter(
        -1, sorted_indices, sorted_probs
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
