import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# (B, T, Hg, H, K, V, C). Qwen3NextConfig has 16 key heads, 32 value
# heads and K=V=128; TP=4 gives Hg=4, H=8. The SGLang NPU GDN extend
# path calls fla/chunk.py -> recompute_w_u_fwd after solve_tril, using C=64.
# T=2048 is a representative prefill load, not a captured runtime input.
perf_shape = [
    (1, 2048, 16, 32, 128, 128, 64),
    (1, 2048, 4, 8, 128, 128, 64),
    (1, 2048, 16, 48, 128, 128, 64),
    (1, 2048, 4, 12, 128, 128, 64),
]


def generate_test_data(shape, dtype, mode, lengths=None, index_dtype=torch.int64):
    """normal: dense lower-triangular A and nonzero gates.

    zero_beta: both outputs must vanish; no_decay: exp(g) must equal one.
    A is unit lower triangular, as are the inverse matrices from solve_tril.
    """
    torch.npu.manual_seed(42)
    B, T, Hg, H, K, V, C = shape
    k = torch.nn.functional.normalize(torch.randn(B, T, Hg, K), dim=-1).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    beta = torch.rand(B, T, H).to(dtype)
    g = torch.empty(B, T, H, dtype=torch.float32)
    A = torch.zeros(B, T, H, C, dtype=dtype)
    lengths = (T,) if lengths is None else lengths
    bos = 0
    for length in lengths:
        for offset in range(0, length, C):
            start = bos + offset
            size = min(C, length - offset)
            block = torch.eye(C) + (0.02 * torch.randn(B, H, C, C)).tril(-1)
            A[:, start : start + size] = block.permute(0, 2, 1, 3)[:, :size].to(dtype)
            increments = -0.02 * torch.rand(B, size, H)
            g[:, start : start + size] = increments.cumsum(1)
        bos += length
    if mode == "zero_beta":
        beta.zero_()
    elif mode == "no_decay":
        g.zero_()
    elif mode != "normal":
        raise ValueError(f"Unknown mode: {mode}")
    cu = None
    if len(lengths) > 1:
        cu = torch.tensor((0,) + lengths, dtype=index_dtype).cumsum(0).to(index_dtype)
    return k, v, beta, g, A, cu


@pytest.mark.parametrize(
    "shape,lengths",
    [
        pytest.param(
            (1, 2048, 16, 32, 128, 128, 64), None, id="qwen3-next-tp1-prefill"
        ),
        pytest.param((1, 2048, 4, 8, 128, 128, 64), None, id="qwen3-next-tp4-prefill"),
        # https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json
        # Hg=16, H=48, K=V=128; TP=4 -> Hg=4, H=12. T is a sample load.
        pytest.param(
            (1, 2048, 16, 48, 128, 128, 64), None, id="qwen3.5-27b-tp1-prefill"
        ),
        pytest.param(
            (1, 2048, 4, 12, 128, 128, 64), None, id="qwen3.5-27b-tp4-prefill"
        ),
        pytest.param((1, 1, 1, 1, 1, 1, 16), None, id="minimum"),
        pytest.param((2, 15, 1, 3, 17, 33, 16), None, id="batch2-group3-tail16"),
        pytest.param((3, 33, 2, 4, 128, 128, 32), None, id="batch3-tail32"),
        pytest.param((1, 63, 1, 1, 64, 96, 64), None, id="chunk-minus-one"),
        pytest.param((1, 64, 2, 4, 128, 128, 64), None, id="exact-chunk"),
        pytest.param((2, 65, 1, 2, 129, 257, 64), None, id="batch2-feature-tile-tails"),
        pytest.param((1, 82, 1, 3, 17, 129, 16), (1, 16, 65), id="packed16"),
        pytest.param((1, 82, 2, 4, 129, 33, 32), (1, 16, 65), id="packed32"),
        pytest.param(
            (1, 2115, 4, 8, 128, 128, 64), (1, 65, 2049), id="packed64-prefill-tail"
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "zero_beta", "no_decay"])
def test_recompute_w_u_fwd(
    shape,
    lengths,
    dtype,
    mode,
):
    from sgl_kernel_npu.operators.attention.gdn import recompute_w_u_fwd

    args = generate_test_data(shape, dtype, mode, lengths)
    executor = check_accuracy(recompute_w_u_fwd, args=args)
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape and dtype == torch.bfloat16 and mode == "normal"
        ),
    )


# Verify that cu_seqlens supports both int32 and int64 dtypes.
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_recompute_w_u_fwd_packed_index_dtype(
    index_dtype,
):
    from sgl_kernel_npu.operators.attention.gdn import recompute_w_u_fwd

    args = generate_test_data(
        (1, 82, 2, 4, 128, 128, 64), torch.bfloat16, "normal", (1, 16, 65), index_dtype
    )
    executor = check_accuracy(recompute_w_u_fwd, args=args)
    check_performance(executor, enable_perf=False)
