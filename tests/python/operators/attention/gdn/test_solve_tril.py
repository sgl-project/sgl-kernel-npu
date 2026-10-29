import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# SGLang Qwen3NextConfig defaults to 32 value heads; TP=4 gives H=8.
# gdn_triton.TritonGDNKernel.extend -> chunk_gated_delta_rule_npu ->
# chunk_scaled_dot_kkt_fwd_npu produces [1, total_tokens, H, 64] in FP32.
# Sources: sglang/srt/{configs,models}/qwen3_next.py and
# sglang/srt/layers/attention/linear/kernels/gdn_triton.py.
# Lengths below are representative prefill loads, not captured runtime inputs.
perf_shape = [(1, 2048, 32, 64), (1, 2048, 8, 64)]


def generate_test_data(shape, dtype, mode, lengths=None, index_dtype=torch.int64):
    """normal: stable Gram matrices; zero: inverse must be identity.

    Each packed sequence starts its own chunk, including unaligned boundaries.
    Inputs are allocated on the NPU selected by the shared fixture.
    """
    torch.npu.manual_seed(42)
    B, T, H, C = shape
    A = torch.zeros(shape, dtype=dtype)
    lengths = (T,) if lengths is None else lengths
    bos = 0
    for length in lengths:
        for start in range(0, length, C):
            size = min(C, length - start)
            if mode == "normal":
                keys = torch.nn.functional.normalize(
                    torch.randn(B, H, C, 32, dtype=torch.float32), dim=-1
                )
                block = (0.1 * (keys @ keys.transpose(-1, -2))).tril(-1)
                A[:, bos + start : bos + start + size] = block.permute(0, 2, 1, 3)[
                    :, :size
                ].to(dtype)
            elif mode != "zero":
                raise ValueError(f"Unknown mode: {mode}")
        bos += length
    cu = None
    if len(lengths) > 1:
        cu = torch.tensor((0,) + lengths, dtype=index_dtype).cumsum(0).to(index_dtype)
    return A, cu


@pytest.mark.parametrize(
    "shape,lengths",
    [
        pytest.param((1, 2048, 32, 64), None, id="qwen3-next-tp1-prefill"),
        pytest.param((1, 2048, 8, 64), None, id="qwen3-next-tp4-prefill"),
        pytest.param((1, 1, 1, 16), None, id="minimum-one-token"),
        pytest.param((2, 15, 3, 16), None, id="batched-odd-heads-tail16"),
        pytest.param((2, 33, 3, 32), None, id="tail32-plus-one"),
        pytest.param((1, 63, 1, 64), None, id="tail64-minus-one"),
        pytest.param((1, 64, 1, 64), None, id="exact64"),
        pytest.param((1, 65, 1, 64), None, id="tail64-plus-one"),
        # The diagonal kernel handles 1216 rows in two tasks of 608 rows.
        pytest.param((1, 607, 1, 64), None, id="task-boundary-minus-one"),
        pytest.param((1, 609, 1, 64), None, id="task-boundary-plus-one"),
        pytest.param((1, 1217, 1, 64), None, id="program-boundary-plus-one"),
        pytest.param((1, 82, 3, 16), (1, 16, 65), id="packed16"),
        pytest.param((1, 82, 3, 32), (1, 16, 65), id="packed32"),
        pytest.param((1, 1283, 8, 64), (1, 65, 1217), id="packed64-program-tail"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "zero"])
def test_solve_tril(
    shape,
    lengths,
    dtype,
    output_dtype,
    mode,
):
    from sgl_kernel_npu.operators.attention.gdn import solve_tril

    A, cu = generate_test_data(shape, dtype, mode, lengths)
    executor = check_accuracy(solve_tril, args=(A, cu, output_dtype))
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.float32
            and output_dtype == torch.bfloat16
            and mode == "normal"
        ),
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_solve_tril_packed_index_dtype(
    index_dtype,
):
    from sgl_kernel_npu.operators.attention.gdn import solve_tril

    A, cu = generate_test_data(
        (1, 82, 3, 64), torch.float32, "normal", (1, 16, 65), index_dtype
    )
    executor = check_accuracy(solve_tril, args=(A, cu))
    check_performance(executor, enable_perf=False)
