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


@pytest.mark.parametrize("backend", ["triton", "torch_native"])
@pytest.mark.parametrize(
    "rows,gamma",
    [
        pytest.param(0, 7, id="empty-batch"),
        pytest.param(1, 0, id="zero-draft"),
        pytest.param(33, 7, id="ci-case"),
        pytest.param(18, 17, id="odd-width"),
    ],
)
@pytest.mark.parametrize(
    "token_dtype", [torch.int32, torch.int64], ids=["token-int32", "token-int64"]
)
@pytest.mark.parametrize(
    "length_dtype", [torch.int32, torch.int64], ids=["length-int32", "length-int64"]
)
@pytest.mark.parametrize("step", [1, 2], ids=["contiguous", "strided"])
def test_build_out_tokens_zeroes_rejected_suffix(
    backend, rows, gamma, token_dtype, length_dtype, step
):
    from sgl_kernel_npu.operators.sample import build_out_tokens_triton

    # Match SGLang's buffer contract independently of either backend. In
    # particular, nonzero rejected drafts must not survive past the bonus.
    offset = 2**40 if token_dtype == torch.int64 else 1000
    draft = (
        torch.arange(rows * gamma * step, dtype=token_dtype, device="cpu") + offset
    ).reshape(rows, gamma * step)
    lengths = (torch.arange(rows * step, device="cpu") // step % (gamma + 1)).to(
        length_dtype
    )
    bonus = torch.arange(rows * step, dtype=token_dtype, device="cpu") - offset
    device = torch.get_default_device()
    inputs = dict(
        draft_tokens=draft.to(device)[:, ::step],
        correct_len=lengths.to(device)[::step],
        bonus=bonus.to(device)[::step],
        verify_num_draft_tokens=gamma + 1,
        gamma=gamma,
    )
    expected = torch.zeros((rows, gamma + 1), dtype=torch.int64, device="cpu")
    for row in range(rows):
        length = int(lengths[row * step])
        expected[row, :length] = draft[row, : length * step : step]
        expected[row, length] = bonus[row * step]

    actual = build_out_tokens_triton(**inputs, impl=backend)
    assert actual.dtype == torch.int64
    assert actual.device == inputs["draft_tokens"].device
    assert torch.equal(actual.cpu(), expected)
