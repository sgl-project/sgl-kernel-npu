import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(8, 6)]


def generate_test_data(rows, gamma):
    return {
        "draft_tokens": torch.arange(rows * gamma).reshape(rows, gamma),
        "correct_len": torch.arange(rows) % (gamma + 1),
        "bonus": torch.arange(rows) + 50,
        "verify_num_draft_tokens": gamma + 1,
        "gamma": gamma,
    }


@pytest.mark.parametrize(
    "rows,gamma",
    [
        pytest.param(1, 0, id="edge-zero-draft"),
        pytest.param(4, 3, id="all-accept-lengths"),
        pytest.param(18, 17, id="odd-width-all-accept-lengths"),
        # DSpark draft block size 6, verify length 7; batch is an example.
        pytest.param(8, 6, id="dspark-draft6-verify7-b8"),
    ],
)
def test_build_out_tokens(rows, gamma):
    from sgl_kernel_npu.operators.sample import build_out_tokens_triton

    kwargs = generate_test_data(rows, gamma)
    executor = check_accuracy(build_out_tokens_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=(rows, gamma) in perf_shape)
