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
perf_shape = [(1, 2048, 32, 64), (1, 2048, 8, 64), (1, 2048, 48, 64), (1, 2048, 12, 64)]


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
                    torch.randn(B, H, C, 32, dtype=torch.float32),
                    dim=-1,
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


SHAPE_CASES = [
    pytest.param((1, 2048, 32, 64), None, id="qwen3-next-tp1-prefill"),
    pytest.param((1, 2048, 8, 64), None, id="qwen3-next-tp4-prefill"),
    # https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json
    # 16 key heads, 48 value heads,
    # K=V=128. solve_tril sees value heads only; T is a sample load.
    pytest.param((1, 2048, 48, 64), None, id="qwen3.5-27b-tp1-prefill"),
    pytest.param((1, 2048, 12, 64), None, id="qwen3.5-27b-tp4-prefill"),
    # Shapes from tests/python/sgl_kernel_npu/test_solve_tril.py.
    pytest.param((1, 63, 1, 16), None, id="legacy-b1-t63-h1-c16"),
    pytest.param((2, 500, 4, 32), None, id="legacy-b2-t500-h4-c32"),
    pytest.param((2, 1000, 5, 64), None, id="legacy-b2-t1000-h5-c64"),
    pytest.param((3, 1024, 6, 64), None, id="legacy-b3-t1024-h6-c64"),
    pytest.param((4, 2048, 8, 64), None, id="legacy-b4-t2048-h8-c64"),
    pytest.param((1, 1, 1, 16), None, id="minimum-one-token"),
    pytest.param((2, 33, 3, 32), None, id="tail32-plus-one"),
    pytest.param((1, 63, 1, 64), None, id="tail64-minus-one"),
    pytest.param((1, 65, 1, 64), None, id="tail64-plus-one"),
    # The diagonal kernel handles 1216 rows in two tasks of 608 rows.
    pytest.param((1, 607, 1, 64), None, id="task-boundary-minus-one"),
    pytest.param((1, 609, 1, 64), None, id="task-boundary-plus-one"),
    pytest.param((1, 1217, 1, 64), None, id="program-boundary-plus-one"),
    pytest.param((1, 82, 3, 16), (1, 16, 65), id="packed16"),
    pytest.param((1, 82, 3, 32), (1, 16, 65), id="packed32"),
    pytest.param((1, 1283, 8, 64), (1, 65, 1217), id="packed64-program-tail"),
]


def accuracy_cases():
    # Exercise all dtype conversions on representative chunk sizes and packed input.
    full_dtype_cases = {
        "legacy-b1-t63-h1-c16",
        "tail32-plus-one",
        "tail64-plus-one",
        "packed32",
    }
    # Zero matrices cover each chunk size in fixed-length and packed layouts.
    zero_cases = {
        "legacy-b1-t63-h1-c16",
        "tail32-plus-one",
        "tail64-plus-one",
        "packed16",
        "packed32",
        "packed64-program-tail",
    }
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    for case in SHAPE_CASES:
        pairs = (
            [(dtype, output_dtype) for dtype in dtypes for output_dtype in dtypes]
            if case.id in full_dtype_cases
            else [(torch.float32, torch.bfloat16)]
        )
        for dtype, output_dtype in pairs:
            yield pytest.param(
                *case.values,
                dtype,
                output_dtype,
                "normal",
                id=f"{case.id}-{dtype}-{output_dtype}-normal",
            )
        if case.id in zero_cases:
            yield pytest.param(
                *case.values,
                torch.float32,
                torch.bfloat16,
                "zero",
                id=f"{case.id}-zero",
            )


@pytest.mark.parametrize("shape,lengths,dtype,output_dtype,mode", accuracy_cases())
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


# Verify that cu_seqlens supports both int32 and int64 dtypes.
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
