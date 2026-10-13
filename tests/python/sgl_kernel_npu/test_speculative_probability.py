"""NPU correctness and graph tests for the fixed fused RN implementation."""

import importlib

import pytest
import torch


@pytest.fixture
def probability():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    return importlib.import_module("sgl_kernel_npu.sample.probability")


def _reference(probs, threshold, top_p):
    values, indices = probs.sort(dim=-1, descending=True)
    if top_p:
        values.masked_fill_(values.cumsum(-1) - values > threshold[:, None], 0.0)
    else:
        positions = torch.arange(probs.shape[1], device=probs.device)
        values.masked_fill_(positions >= threshold[:, None], 0.0)
    values.div_(values.sum(-1, keepdim=True).clamp_min_(1e-20))
    return torch.zeros_like(probs).scatter_(-1, indices, values)


@pytest.mark.parametrize(
    "rows,vocab",
    [
        (7, 4),
        (7, 97),
        (7, 2051),
        (7, 154880),
        (7, 248320),
        (216, 154880),
        (216, 248320),
    ],
)
@pytest.mark.parametrize("top_p", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_fused_renorm_shared_sort(probability, rows, vocab, top_p, scalar):
    gen = torch.Generator().manual_seed(17)
    probs = torch.softmax(torch.randn(rows, vocab, generator=gen), -1).npu()
    probs[0].fill_(1.0 / vocab)
    probs[1].zero_()
    probs[2].zero_()
    probs[2, -1] = 1.0
    probs[6] *= 1e-25
    original = probs.clone()
    values, indices = probs.sort(dim=-1, descending=True)
    if top_p:
        threshold = torch.tensor([0, 0.25, 0.5, 0.95, 1, 1e-8, 0.95], device="npu")
        cumulative = values.cumsum(-1)
    else:
        threshold = torch.tensor([1, 2, 3, vocab - 1, vocab, 1, vocab], device="npu")
        cumulative = None
    threshold = threshold.repeat((rows + 6) // 7)[:rows]
    if scalar:
        threshold = threshold[3].expand(rows)
    filtered = values.clone()
    if top_p:
        filtered.masked_fill_(cumulative - filtered > threshold[:, None], 0.0)
    else:
        positions = torch.arange(vocab, device="npu")
        filtered.masked_fill_(positions >= threshold[:, None], 0.0)
    filtered.div_(filtered.sum(-1, keepdim=True).clamp_min_(1e-20))
    # Reuse the same sort permutation to avoid tie-order ambiguity.
    expected = torch.zeros_like(probs).scatter_(-1, indices, filtered)
    actual = probability.filter_and_renorm(
        probs, values, indices, threshold, cumulative
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(probs, original, rtol=0, atol=0)


@pytest.mark.parametrize("scalar_kind", ["python", "tensor", "rows"])
def test_public_top_k_then_top_p(probability, scalar_kind):
    probs = torch.arange(1, 195, dtype=torch.float32, device="npu").reshape(2, 97)
    probs /= probs.sum(-1, keepdim=True)
    if scalar_kind == "python":
        ks, ps = 31, 0.95
    elif scalar_kind == "tensor":
        ks, ps = torch.tensor(31), torch.tensor(0.95)
    else:
        ks, ps = torch.tensor([1, 97]), torch.tensor([0, 1.0])
    k_rows = torch.as_tensor(ks, device="npu").expand(2)
    p_rows = torch.as_tensor(ps, device="npu").expand(2)
    expected = _reference(_reference(probs, k_rows, False), p_rows, True)
    actual = probability.top_p_renorm_prob(probability.top_k_renorm_prob(probs, ks), ps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("top_p", [False, True])
def test_strided_input_thresholds_and_clamping(probability, top_p):
    probs = torch.arange(1, 195, device="npu", dtype=torch.float32).reshape(97, 2).t()
    probs = probs / probs.sum(-1, keepdim=True)
    probs = probs.t().contiguous().t()
    assert not probs.is_contiguous()
    raw = torch.tensor([-1, 999, 2 if top_p else 200, 999], device="npu")
    threshold = raw[::2]
    assert threshold.stride(0) == 2
    fn = probability.top_p_renorm_prob if top_p else probability.top_k_renorm_prob
    clamped = threshold.clamp(0, 1) if top_p else threshold.clamp(1, 97)
    original = probs.clone()
    actual = fn(probs, threshold)
    torch.testing.assert_close(
        actual, _reference(probs, clamped, top_p), rtol=0, atol=0
    )
    torch.testing.assert_close(probs, original, rtol=0, atol=0)


def test_validation_and_empty_batch(probability):
    p = torch.ones(2, 4, device="npu")
    for fn in (probability.top_p_renorm_prob, probability.top_k_renorm_prob):
        assert fn(p[:0], 1).shape == (0, 4)
        with pytest.raises(ValueError, match="shape"):
            fn(p[0], 1)
        with pytest.raises(ValueError, match="float32"):
            fn(p.bfloat16(), 1)
        with pytest.raises(ValueError, match="NPU"):
            fn(p.cpu(), 1)
        with pytest.raises(ValueError, match="vocab_size"):
            fn(p[:, :0], 1)
        with pytest.raises(ValueError, match="vocab_size"):
            fn(p[:1, :1].expand(1, 2**23 + 1), 1)
        with pytest.raises(ValueError, match="scalar or"):
            fn(p, torch.ones(3))
        with pytest.raises(ValueError, match="scalar or"):
            fn(p, torch.ones(2, 1))


@pytest.mark.parametrize("rows,vocab", [(4, 2051), (216, 154880), (216, 248320)])
def test_graph_replay_with_changed_inputs(probability, rows, vocab):
    probs = torch.softmax(torch.randn(rows, vocab, device="npu"), -1)
    ps = torch.tensor([0, 0.5, 0.95, 1], device="npu").repeat((rows + 3) // 4)[:rows]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            probability.top_p_renorm_prob(probs, ps)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        actual = probability.top_p_renorm_prob(probs, ps)
    torch.npu.synchronize()
    for _ in range(2):
        probs.copy_(torch.softmax(torch.randn_like(probs), -1))
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, _reference(probs, ps, True), rtol=0, atol=0)
