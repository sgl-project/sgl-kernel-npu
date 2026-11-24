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
    actual = top_k_top_p_renorm_prob(probs, ks, ps, 10, impl="triton")
    expected = top_k_top_p_renorm_prob(probs, ks, ps, 10, impl="torch_native")
    assert int((actual[0] > 0).sum()) == 10
    assert int((actual[1] > 0).sum()) == 5
    assert_accuracy(actual, expected)


def test_tied_rows_keep_equal_mass():
    probs = torch.full((3, 97), 1.0 / 97, device="npu")
    ks = torch.tensor([10, 20, 5], device="npu", dtype=torch.int32)
    ps = torch.tensor([1.0, 0.95, 0.0], device="npu")
    actual = top_k_top_p_renorm_prob(probs, ks, ps, 20, impl="triton")
    expected = top_k_top_p_renorm_prob(probs, ks, ps, 20, impl="torch_native")
    assert (actual > 0).sum(dim=-1).tolist() == [10, 20, 1]
    # Ties may resolve to different token indices; compare the kept values.
    assert_accuracy(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
    assert_accuracy(actual.sum(dim=-1), torch.ones(3, device="npu"))


def test_rounding_does_not_change_selected_token():
    probs = torch.tensor(
        [0.375, 0.19, 0.19000001, 0.14, 0.10499999], device="npu"
    )
    ks = torch.tensor([2], device="npu", dtype=torch.int32)
    ps = torch.tensor([0.6], device="npu")
    for order in ([0, 1, 2, 3, 4], [0, 2, 1, 3, 4]):
        reordered = probs[order].unsqueeze(0).contiguous()
        actual = top_k_top_p_renorm_prob(reordered, ks, ps, 2, impl="triton")
        expected = top_k_top_p_renorm_prob(
            reordered, ks, ps, 2, impl="torch_native"
        )
        assert_accuracy(actual, expected)
        kept = (actual[0] > 0).nonzero().flatten().tolist()
        # 0.19000001 is the larger of the two near-ties and must be kept.
        assert order.index(2) in kept


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
    with pytest.raises(ValueError, match="max_keep"):
        top_k_top_p_renorm_prob(p[:0], 1, 0.9, 0, impl=impl)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_keep"):
            top_k_top_p_renorm_prob(p, 1, 0.9, bad, impl=impl)


@pytest.mark.extra_checks
def test_launches_on_input_device():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPUs")
    with torch.npu.device(0):
        probs = torch.softmax(torch.randn(4, 97, device="npu:1"), -1)
        ks = torch.full((4,), 8, device="npu:1", dtype=torch.int32)
        ps = torch.full((4,), 0.9, device="npu:1")
        actual = top_k_top_p_renorm_prob(probs, ks, ps, 8, impl="triton")
        assert actual.device == probs.device
        expected = top_k_top_p_renorm_prob(probs, ks, ps, 8, impl="torch_native")
        assert_accuracy(actual, expected)


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
