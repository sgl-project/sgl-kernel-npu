"""Numerical checks for the three public TorchNPU RoPE variants."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_rope_data(shape, freq_shape):
    torch.npu.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float16)
    angles = torch.randn(freq_shape, dtype=torch.float32)
    return x, angles.cos().half(), angles.sin().half()


@pytest.mark.parametrize(
    "shape, freq_shape",
    [
        # DeepSeek MLA reshapes queries to [tokens, local_heads, 1, rope_dim].
        pytest.param((16, 16, 1, 64), (16, 1, 1, 64), id="deepseek-mla-decode-layout"),
    ],
)
def test_interleave_rope(shape, freq_shape):
    from sgl_kernel_npu.operators.rope import interleave_rope

    executor = check_accuracy(
        interleave_rope, args=generate_rope_data(shape, freq_shape)
    )
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "seq_len, rotary_mode", [pytest.param(17, "half", id="multimodal-bsnd")]
)
def test_rotary_mul(seq_len, rotary_mode):
    from sgl_kernel_npu.operators.rope import rotary_mul

    # SGLang multimodal embeddings use BSND with 1S1D cosine/sine.
    args = generate_rope_data((2, seq_len, 4, 64), (1, seq_len, 1, 64))
    executor = check_accuracy(rotary_mul, args=(*args, rotary_mode))
    check_performance(executor, enable_perf=True)


def generate_mrope_data(num_tokens):
    torch.npu.manual_seed(42)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    positions = torch.stack((positions, positions + 1, positions + 2))
    head_size = 128
    query = torch.randn(num_tokens, 4 * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, head_size, dtype=torch.float16)
    angles = torch.randn(num_tokens + 2, head_size // 2, dtype=torch.float32)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
    return positions, query, key, cache, head_size


@pytest.mark.parametrize(
    "num_tokens, section",
    [
        pytest.param(7, [16, 24, 24], id="multimodal-distinct-position-axes"),
    ],
)
def test_mrope(num_tokens, section):
    from sgl_kernel_npu.operators.rope import mrope

    require_test_impl(mrope, get_test_impls()[0])
    executor = check_accuracy(
        mrope,
        args=generate_mrope_data(num_tokens),
        kwargs={"mrope_section": section},
    )
    check_performance(executor, enable_perf=True)
