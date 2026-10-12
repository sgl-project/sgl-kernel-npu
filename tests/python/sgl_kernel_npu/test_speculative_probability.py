from unittest.mock import patch

import pytest
import torch
import torch_npu
from sgl_kernel_npu.sample import probability
from sgl_kernel_npu.sample.probability import top_k_renorm_prob, top_p_renorm_prob

npu_only = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
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


def _reference_renorm(probs, sorted_probs, sorted_indices):
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    return torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["dense", "sparse", "zeros", "nan", "strided"])
def test_sorted_probability_scatter_matches_dense_reference(device, dtype, kind):
    if device == "npu" and not torch_npu.npu.is_available():
        pytest.skip("NPU is required")
    torch.manual_seed(47)
    probs = torch.rand((4, 513), dtype=dtype, device=device)
    if kind == "strided":
        probs = probs.T.contiguous().T
    values, indices = probs.sort(dim=-1, descending=True)
    if kind == "sparse":
        values[:, 17:] = 0
    elif kind == "zeros":
        values.zero_()
    elif kind == "nan":
        values[0, 0] = float("nan")
    expected = _reference_renorm(probs, values.clone(), indices)
    actual = probability._renorm_from_sorted_probs(probs, values.clone(), indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(0, 97), (4, 0)])
@npu_only
def test_empty_sorted_probability_scatter(shape):
    probs = torch.empty(shape, device="npu")
    values = torch.empty_like(probs)
    indices = torch.empty(shape, device="npu", dtype=torch.int64)
    actual = probability._renorm_from_sorted_probs(probs, values, indices)
    assert actual.shape == shape and actual.numel() == 0


@pytest.mark.parametrize("k", [1, 20, 64, 513])
@npu_only
def test_npu_heterogeneous_filters_match_original(k):
    torch.manual_seed(81)
    probs = torch.softmax(torch.randn((4, 513), device="npu"), dim=-1)
    ks = torch.tensor([1, min(7, k), min(31, k), k], device="npu")
    ps = torch.tensor([0.0, 0.5, 0.95, 1.0], device="npu")
    with patch.object(probability, "_renorm_from_sorted_probs", _reference_renorm):
        expected = top_p_renorm_prob(top_k_renorm_prob(probs, ks), ps)
        expected_p = top_p_renorm_prob(probs, ps)
    actual = top_p_renorm_prob(top_k_renorm_prob(probs, ks), ps)
    actual_p = top_p_renorm_prob(probs, ps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_p, expected_p, rtol=0, atol=0)


@npu_only
def test_npu_probability_graph_replays_changed_values():
    probs = torch.rand((2, 513), device="npu")
    values, indices = probs.sort(dim=-1, descending=True)
    probability._renorm_from_sorted_probs(probs, values.clone(), indices)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = probability._renorm_from_sorted_probs(probs, values, indices)
    for nonzeros in [0, 1, 20, 64, 513]:
        next_values = torch.rand_like(probs)
        next_values[:, nonzeros:] = 0
        values.copy_(next_values)
        indices.copy_(indices.roll(1, dims=1))
        graph.replay()
        torch.npu.synchronize()
        expected = _reference_renorm(probs, next_values, indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("grid_limit", [2, 9])
@npu_only
def test_npu_probability_scatter_chunks_launch_grid(grid_limit):
    from sgl_kernel_npu.sample import probability_scatter

    probs = torch.rand((9, 1025), device="npu")
    values, indices = probs.sort(dim=-1, descending=True)
    values[:, 3:] = 0
    with patch.object(probability_scatter, "_MAX_GRID_PROGRAMS", grid_limit):
        actual = probability._renorm_from_sorted_probs(probs, values.clone(), indices)
    expected = _reference_renorm(probs, values, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@npu_only
@pytest.mark.parametrize("last_column_live", [False, True])
def test_npu_large_vocabulary_bound_preserves_holes(last_column_live):
    torch.manual_seed(67)
    probs = torch.softmax(torch.randn((40, 248320), device="npu"), dim=-1)
    values, indices = probs.sort(dim=-1, descending=True)
    values[:, 64:] = 0
    values[:, 1::3] = 0
    if last_column_live:
        values[:, -1] = values[:, 0]
    values[0] = 0
    expected = _reference_renorm(probs, values.clone(), indices)
    actual = probability._renorm_from_sorted_probs(probs, values.clone(), indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
