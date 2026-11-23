import pytest

torch = pytest.importorskip("torch")

from sgl_kernel_npu.operators.sample import top_k_top_p_renorm_prob

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_skip_acc_check, get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Shapes cover the minimum vocabulary, ragged tails, a page-tail size, the
# SGLang EAGLE verify call (5 requests x 4 draft tokens), large vocabularies,
# and the perf shapes used by the existing renorm tests (216 rows).
perf_shape = [(216, 154880), (216, 248320)]


def generate_data(rows, vocab, scalar):
    torch.manual_seed(17)
    probs = torch.softmax(torch.randn(rows, vocab), -1)
    probs[0].fill_(1.0 / vocab)
    probs[1].zero_()
    probs[1, -1] = 1.0
    ks = torch.tensor([20, 5, 100, 1, 64, 3, 7], dtype=torch.int64)
    ps = torch.tensor([1.0, 0.95, 0.5, 0.0, 0.8, 0.99, 0.95])
    ks = ks.repeat((rows + 6) // 7)[:rows].clamp(max=vocab)
    ps = ps.repeat((rows + 6) // 7)[:rows]
    if scalar:
        return probs, 16, 0.95, 16
    return probs, ks, ps, int(ks.max())


@pytest.mark.parametrize(
    "rows,vocab",
    [
        (7, 4),
        (7, 97),
        (7, 2051),
        (20, 248320),
        (7, 154880),
        (7, 248320),
        (216, 248320),
    ],
    ids=[
        "vocab-minimum",
        "vocab-ragged-tail",
        "vocab-page-tail",
        "sglang-eagle-verify",
        "vocab-large",
        "vocab-max",
        "perf-batch-216",
    ],
)
@pytest.mark.parametrize(
    "scalar", [False, True], ids=["row-thresholds", "scalar-thresholds"]
)
def test_joint_renorm(rows, vocab, scalar):
    probs, ks, ps, max_keep = generate_data(rows, vocab, scalar)
    original = probs.clone()
    executor = check_accuracy(
        top_k_top_p_renorm_prob,
        args=(probs, ks, ps),
        kwargs={"max_keep": max_keep},
    )
    if not get_skip_acc_check():
        assert_accuracy(
            probs, original, standard={"torch.float32": AccuracyStandard(0, 0)}
        )
    check_performance(executor, enable_perf=(rows, vocab) in perf_shape)


def test_scalar_tensor_thresholds():
    probs, _, _, _ = generate_data(7, 2051, False)
    ks = torch.tensor(16)
    ps = torch.tensor(0.95)
    check_accuracy(
        top_k_top_p_renorm_prob, args=(probs, ks, ps), kwargs={"max_keep": 16}
    )


@pytest.mark.parametrize(
    "rows,vocab",
    [(7, 97), (7, 2051)],
    ids=["vocab-ragged-tail", "vocab-page-tail"],
)
def test_max_keep_none_matches_full_path(rows, vocab):
    probs, ks, ps, _ = generate_data(rows, vocab, False)
    executor = check_accuracy(
        top_k_top_p_renorm_prob, args=(probs, ks, ps), kwargs={"max_keep": None}
    )
    bounded = top_k_top_p_renorm_prob(probs, ks, ps, int(ks.max()), impl="triton")
    assert_accuracy(executor(), bounded)


def test_max_keep_truncates_rows_above_bound():
    torch.manual_seed(3)
    probs = torch.softmax(torch.randn(2, 97, device="npu"), -1)
    ks = torch.tensor([20, 5], device="npu", dtype=torch.int32)
    ps = torch.tensor([1.0, 1.0], device="npu")
    out = top_k_top_p_renorm_prob(probs, ks, ps, 10, impl="triton")
    assert int((out[0] > 0).sum()) == 10
    assert int((out[1] > 0).sum()) == 5


@pytest.mark.extra_checks
def test_validation_and_empty_batch():
    impl, _ = get_test_impls()
    p = torch.ones(2, 4, device="npu")
    assert top_k_top_p_renorm_prob(p[:0], 1, 0.9, impl=impl).shape == (0, 4)
    with pytest.raises(ValueError, match="shape"):
        top_k_top_p_renorm_prob(p[0], 1, 0.9, impl=impl)
    with pytest.raises(ValueError, match="float32"):
        top_k_top_p_renorm_prob(p.bfloat16(), 1, 0.9, impl=impl)
    with pytest.raises(ValueError, match="NPU"):
        top_k_top_p_renorm_prob(p.cpu(), 1, 0.9, impl=impl)
    with pytest.raises(ValueError, match="vocab_size"):
        top_k_top_p_renorm_prob(p[:, :0], 1, 0.9, impl=impl)
    with pytest.raises(ValueError, match="vocab_size"):
        top_k_top_p_renorm_prob(
            p[:1, :1].expand(1, 2**23 + 1), 1, 0.9, impl=impl
        )
    with pytest.raises(ValueError, match="scalar or"):
        top_k_top_p_renorm_prob(p, torch.ones(3), 0.9, impl=impl)
    with pytest.raises(ValueError, match="scalar or"):
        top_k_top_p_renorm_prob(p, 1, torch.ones(3), impl=impl)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_keep"):
            top_k_top_p_renorm_prob(p, 1, 0.9, bad, impl=impl)


@pytest.mark.parametrize(
    "rows,vocab",
    [(4, 2051), (216, 154880)],
    ids=["vocab-page-tail", "perf-batch-216"],
)
@pytest.mark.extra_checks
def test_graph_replay_with_changed_inputs(rows, vocab):
    impl, impl_ref = get_test_impls()
    probs = torch.softmax(torch.randn(rows, vocab, device="npu"), -1)
    ks = torch.full((rows,), 20, device="npu", dtype=torch.int32)
    ps = torch.tensor([0.0, 0.5, 0.95, 1.0], device="npu").repeat(
        (rows + 3) // 4
    )[:rows]
    max_keep = 20
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            top_k_top_p_renorm_prob(probs, ks, ps, max_keep, impl=impl)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        actual = top_k_top_p_renorm_prob(probs, ks, ps, max_keep, impl=impl)
    torch.npu.synchronize()
    for _ in range(2):
        probs.copy_(torch.softmax(torch.randn_like(probs), -1))
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert_accuracy(
            actual,
            top_k_top_p_renorm_prob(probs, ks, ps, max_keep, impl=impl_ref),
        )