"""Numerical checks for the three public TorchNPU RoPE variants."""

import pytest

from tests.python.utils.accuracy import check_accuracy
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
        pytest.param((1, 1, 3, 64), (1, 1, 1, 64), id="sequence-broadcast"),
    ],
)
def test_interleave_rope(shape, freq_shape):
    from sgl_kernel_npu.operators.rope import interleave_rope

    executor = check_accuracy(
        interleave_rope, args=generate_rope_data(shape, freq_shape)
    )
    check_performance(executor, enable_perf=shape == (16, 16, 1, 64))


@pytest.mark.parametrize("rotary_mode", ["half", "interleave"])
@pytest.mark.parametrize("seq_len", [1, 17])
def test_rotary_mul(seq_len, rotary_mode):
    from sgl_kernel_npu.operators.rope import rotary_mul

    # SGLang multimodal embeddings use BSND with 1S1D cosine/sine.
    args = generate_rope_data((2, seq_len, 4, 64), (1, seq_len, 1, 64))
    executor = check_accuracy(rotary_mul, args=(*args, rotary_mode))
    check_performance(executor, enable_perf=seq_len == 17 and rotary_mode == "half")


def generate_mrope_data(num_tokens, multimodal):
    torch.npu.manual_seed(42)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    if multimodal:
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
        pytest.param(1, [0, 0, 0], id="text-one-token"),
        pytest.param(7, [16, 24, 24], id="multimodal-distinct-position-axes"),
    ],
)
def test_mrope(num_tokens, section):
    from sgl_kernel_npu.operators.rope import mrope

    multimodal = any(section)
    executor = check_accuracy(
        mrope,
        args=generate_mrope_data(num_tokens, multimodal),
        kwargs={"mrope_section": section},
    )
    check_performance(executor, enable_perf=multimodal)
