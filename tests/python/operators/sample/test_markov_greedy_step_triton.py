import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 129280, 256)]


def generate_test_data(rows, vocab, hidden, dtype):
    return {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "prev_embeds": torch.randn(rows, hidden, dtype=dtype),
        "w2_weight": torch.randn(vocab, hidden, dtype=dtype),
    }


@pytest.mark.parametrize(
    "rows,vocab,hidden,dtype",
    [
        pytest.param(1, 513, 33, torch.float32, id="single-odd-tile-and-rank"),
        pytest.param(8, 4097, 32, torch.float32, id="batched-cross-tile"),
        pytest.param(32, 4097, 32, torch.float32, id="large-batch-row-scan"),
        # Model vocabulary and Markov rank; batch is an example workload.
        pytest.param(1, 129280, 256, torch.bfloat16, id="deepseek-v4-flash-b1"),
    ],
)
def test_markov_greedy(rows, vocab, hidden, dtype):
    from sgl_kernel_npu.operators.sample import markov_greedy_step_triton

    kwargs = generate_test_data(rows, vocab, hidden, dtype)
    executor = check_accuracy(markov_greedy_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=(rows, vocab, hidden) in perf_shape)


@pytest.mark.parametrize("rows", [8, 32], ids=["partial-combine", "row-scan"])
def test_markov_greedy_graph_replay(rows):
    from sgl_kernel_npu.operators.sample import markov_greedy_step_triton

    kwargs = generate_test_data(rows, 4097, 32, torch.float32)
    logits = kwargs["base_logits"]
    kwargs["prev_embeds"].zero_()
    markov_greedy_step_triton(**kwargs, impl="triton")
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = markov_greedy_step_triton(**kwargs, impl="triton")

    for winner in (7, 4096):
        logits.zero_()
        logits[:, winner] = 1
        graph.replay()
        expected = markov_greedy_step_triton(**kwargs, impl="torch_native")
        assert torch.equal(actual.cpu(), expected.cpu())
        assert expected.tolist() == [winner] * rows
