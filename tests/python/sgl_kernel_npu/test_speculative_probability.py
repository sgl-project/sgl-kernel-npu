import torch
from sgl_kernel_npu.sample.probability import top_k_renorm_prob, top_p_renorm_prob


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
