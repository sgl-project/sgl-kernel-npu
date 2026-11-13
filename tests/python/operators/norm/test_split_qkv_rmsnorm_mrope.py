import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(65, 2, 1, 256), (17, 16, 4, 256)]
standard = {"torch.bfloat16": AccuracyStandard(5e-2, 5e-3)}


def generate_data(tokens, q_heads, kv_heads, head_size, rope_dim, gate, bias):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    qsize, kvsize = q_heads * head_size, kv_heads * head_size
    qkv = torch.randn(tokens, qsize * (2 if gate else 1) + 2 * kvsize, dtype=dtype)
    wq, wk = torch.randn(head_size, dtype=dtype), torch.randn(head_size, dtype=dtype)
    bq, bk = (
        (torch.randn(head_size, dtype=dtype), torch.randn(head_size, dtype=dtype))
        if bias
        else (None, None)
    )
    cs = torch.randn(3, tokens, rope_dim, dtype=dtype)
    return qkv, wq, wk, cs, bq, bk


@pytest.mark.parametrize(
    "tokens,q_heads,kv_heads,head_size,rope_dim,sections,interleaved,gate,bias",
    [
        pytest.param(
            7,
            2,
            1,
            256,
            256,
            [32, 48, 48],
            True,
            True,
            False,
            id="full-interleaved-gate",
        ),
        pytest.param(
            65,
            2,
            1,
            256,
            128,
            [48, 40, 40],
            False,
            False,
            True,
            id="partial-contiguous-bias",
        ),
        pytest.param(
            17, 16, 4, 256, 64, [11, 11, 10], True, True, False, id="qwen3-5-prefill"
        ),
    ],
)
def test_mrope(
    tokens, q_heads, kv_heads, head_size, rope_dim, sections, interleaved, gate, bias
):
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_mrope

    qkv, wq, wk, cs, bq, bk = generate_data(
        tokens, q_heads, kv_heads, head_size, rope_dim, gate, bias
    )
    args = (
        qkv,
        wq,
        wk,
        cs,
        q_heads,
        kv_heads,
        head_size,
        1e-6,
        sections,
        interleaved,
        rope_dim,
        bq,
        bk,
        gate,
    )
    executor = check_accuracy(split_qkv_rmsnorm_mrope, args=args, standard=standard)
    check_performance(
        executor, enable_perf=(tokens, q_heads, kv_heads, head_size) in perf_shape
    )
