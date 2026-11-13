import pytest

torch = pytest.importorskip("torch")

from sgl_kernel_npu.operators.sample import top_k_renorm_prob, top_p_renorm_prob

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_skip_acc_check, get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard


def _reference(probs, threshold, top_p):
    _, impl_ref = get_test_impls()
    op = top_p_renorm_prob if top_p else top_k_renorm_prob
    return op(probs, threshold, impl=impl_ref)


perf_shape = [(216, 154880), (216, 248320)]


def generate_data(rows, vocab, top_p, scalar):
    torch.manual_seed(17)
    probs = torch.softmax(torch.randn(rows, vocab), -1)
    probs[0].fill_(1.0 / vocab)
    probs[1].zero_()
    probs[2].zero_()
    probs[2, -1] = 1.0
    probs[6] *= 1e-25
    threshold = torch.tensor(
        [0, 0.25, 0.5, 0.95, 1, 1e-8, 0.95]
        if top_p
        else [1, 2, 3, vocab - 1, vocab, 1, vocab]
    )
    threshold = threshold.repeat((rows + 6) // 7)[:rows]
    return probs, threshold[3].expand(rows) if scalar else threshold


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
def test_renorm(rows, vocab, top_p, scalar):
    from sgl_kernel_npu.operators.sample import top_k_renorm_prob, top_p_renorm_prob

    # Generic sampler: original vocabulary sizes cover ragged tails and large model vocabularies.
    op = top_p_renorm_prob if top_p else top_k_renorm_prob
    args = generate_data(rows, vocab, top_p, scalar)
    original = args[0].clone()
    executor = check_accuracy(
        op, args=args, standard={"torch.float32": AccuracyStandard(0, 0)}
    )
    if not get_skip_acc_check():
        assert_accuracy(
            args[0], original, standard={"torch.float32": AccuracyStandard(0, 0)}
        )
    check_performance(executor, enable_perf=(rows, vocab) in perf_shape)


def generate_chain_data(scalar_kind):
    probs = torch.arange(1, 195, dtype=torch.float32).reshape(2, 97)
    probs /= probs.sum(-1, keepdim=True)
    if scalar_kind == "python":
        ks, ps = 31, 0.95
    elif scalar_kind == "tensor":
        ks, ps = torch.tensor(31, device="cpu"), torch.tensor(0.95, device="cpu")
    else:
        ks, ps = torch.tensor([1, 97], device="cpu"), torch.tensor(
            [0, 1.0], device="cpu"
        )
    return probs, ks, ps


@pytest.mark.parametrize("scalar_kind", ["python", "tensor", "rows"])
def test_public_top_k_then_top_p(scalar_kind):
    probs, ks, ps = generate_chain_data(scalar_kind)
    exact = {"torch.float32": AccuracyStandard(0, 0)}
    first = check_accuracy(top_k_renorm_prob, args=(probs, ks), standard=exact)
    second = check_accuracy(top_p_renorm_prob, args=(first(), ps), standard=exact)
    check_performance(first, enable_perf=False)
    check_performance(second, enable_perf=False)


def generate_strided_data(top_p):
    probs = torch.arange(1, 195, dtype=torch.float32).reshape(97, 2).t()
    probs = probs / probs.sum(-1, keepdim=True)
    probs = probs.t().contiguous().t()
    raw = torch.tensor([-1, 999, 2 if top_p else 200, 999])
    return probs, raw[::2]


@pytest.mark.parametrize("top_p", [False, True])
def test_strided_input_thresholds_and_clamping(top_p):
    probs, threshold = generate_strided_data(top_p)
    assert not probs.is_contiguous()
    assert threshold.stride(0) == 2
    fn = top_p_renorm_prob if top_p else top_k_renorm_prob
    original = probs.clone()
    exact = {"torch.float32": AccuracyStandard(0, 0)}
    executor = check_accuracy(fn, args=(probs, threshold), standard=exact)
    if not get_skip_acc_check():
        clamped = threshold.clamp(0, 1) if top_p else threshold.clamp(1, 97)
        assert_accuracy(executor(), _reference(probs, clamped, top_p), standard=exact)
        assert_accuracy(probs, original, standard=exact)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
def test_validation_and_empty_batch():
    impl, _ = get_test_impls()
    p = torch.ones(2, 4, device="npu")
    for fn in (top_p_renorm_prob, top_k_renorm_prob):
        assert fn(p[:0], 1, impl=impl).shape == (0, 4)
        with pytest.raises(ValueError, match="shape"):
            fn(p[0], 1, impl=impl)
        with pytest.raises(ValueError, match="float32"):
            fn(p.bfloat16(), 1, impl=impl)
        with pytest.raises(ValueError, match="NPU"):
            fn(p.cpu(), 1, impl=impl)
        with pytest.raises(ValueError, match="vocab_size"):
            fn(p[:, :0], 1, impl=impl)
        with pytest.raises(ValueError, match="vocab_size"):
            fn(p[:1, :1].expand(1, 2**23 + 1), 1, impl=impl)
        with pytest.raises(ValueError, match="scalar or"):
            fn(p, torch.ones(3), impl=impl)
        with pytest.raises(ValueError, match="scalar or"):
            fn(p, torch.ones(2, 1), impl=impl)


@pytest.mark.parametrize("rows,vocab", [(4, 2051), (216, 154880), (216, 248320)])
@pytest.mark.extra_checks
def test_graph_replay_with_changed_inputs(rows, vocab):
    impl, _ = get_test_impls()
    probs = torch.softmax(torch.randn(rows, vocab, device="npu"), -1)
    ps = torch.tensor([0, 0.5, 0.95, 1], device="npu").repeat((rows + 3) // 4)[:rows]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            top_p_renorm_prob(probs, ps, impl=impl)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        actual = top_p_renorm_prob(probs, ps, impl=impl)
    torch.npu.synchronize()
    for _ in range(2):
        probs.copy_(torch.softmax(torch.randn_like(probs), -1))
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, _reference(probs, ps, True), rtol=0, atol=0)
