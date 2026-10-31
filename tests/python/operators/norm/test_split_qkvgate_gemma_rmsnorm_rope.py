"""Qwen3-Next/Qwen3.5 full-attention Q/gate projection layouts.

SGLang models/qwen3_next.py and models/qwen3_5.py::forward_prepare_npu
call this operator after QKVParallelLinear emits per-head [Q, gate] pairs.
Qwen3NextConfig and Qwen3.5-35B-A3B use 16 Q heads, 2 KV heads, D=256,
R=64. TP divides Q heads and partitions or replicates KV heads. The selected
token counts model decode/prefill loads; they are not runtime traces.
Inspected SGLang revision: 2456c9111a121a02e41e625e3930e1eb143fe9bf;
the Next defaults are in srt/configs/qwen3_next.py.
Qwen3.5 dimensions: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.norm import generate_qkv_data
from tests.python.utils.performance import check_performance

production_cases = []
perf_shape = []
for tp in (1, 2, 4, 8, 16):
    for tokens in (1, 16, 128, 2048):
        shape = (tokens, 16 // tp, max(1, 2 // tp), 256, 64)
        production_cases.append(
            pytest.param(shape, id=f"qwen3-next-and-35-tp{tp}-tokens{tokens}")
        )
        perf_shape.append(shape)


def generate_test_data(shape, dtype=torch.bfloat16, **kwargs):
    return generate_qkv_data(shape, dtype, gated=True, **kwargs)


# Required: gated projection widths after each supported attention TP partition.
@pytest.mark.parametrize("shape", production_cases)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_split_qkvgate_gemma_rmsnorm_rope_production(
    shape,
    cache_dtype,
):
    from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

    case = generate_test_data(shape, cache_dtype=cache_dtype)
    executor = check_accuracy(split_qkvgate_gemma_rmsnorm_rope, kwargs=case)
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape",
    [(0, 2, 1, 64, 32), (1, 1, 1, 16, 8), (7, 3, 1, 64, 16), (33, 4, 2, 128, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_split_qkvgate_gemma_rmsnorm_rope_generalization(
    shape,
    dtype,
    weight_dtype,
):
    # Cover other GQA ratios, partial rotary widths, mixed weights and empty rows.
    from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

    executor = check_accuracy(
        split_qkvgate_gemma_rmsnorm_rope,
        kwargs=generate_test_data(shape, dtype, weight_dtype=weight_dtype),
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
def test_split_qkvgate_gemma_rmsnorm_rope_output_contract():
    # Zero Gemma weights mean multiplier 1, not 0; gate is a per-head copy.
    if get_skip_acc_check() is True:
        pytest.skip("Output checks disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

    case = generate_test_data((3, 2, 1, 64, 32))
    source = case["input"]
    source[:, :256] = torch.cat(
        (
            torch.ones(3, 1, 64),
            torch.full((3, 1, 64), 2.0),
            torch.ones(3, 1, 64),
            torch.full((3, 1, 64), -3.0),
        ),
        dim=1,
    ).reshape(3, 256)
    source[:, 256:320] = 1
    case["q_weight"].zero_()
    case["k_weight"].zero_()
    case["sin"].zero_()
    case["cos"].fill_(1)
    snapshot = source.clone()
    impl, _ = get_test_impls()
    require_test_impl(split_qkvgate_gemma_rmsnorm_rope, impl)
    q, k, v, gate = split_qkvgate_gemma_rmsnorm_rope(**case, impl=impl)
    torch.testing.assert_close(q, torch.ones_like(q), atol=0, rtol=0)
    torch.testing.assert_close(k, torch.ones_like(k), atol=0, rtol=0)
    torch.testing.assert_close(v, source[:, -64:], atol=0, rtol=0)
    expected_gate = torch.cat(
        (torch.full((3, 64), 2.0), torch.full((3, 64), -3.0)), dim=-1
    ).to(gate.dtype)
    torch.testing.assert_close(gate, expected_gate, atol=0, rtol=0)
    for output in (q, k, v, gate):
        assert output.dtype == source.dtype and output.device == source.device
        assert output.is_contiguous()
        assert (
            output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(source, snapshot, atol=0, rtol=0)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input",
    [
        "width",
        "head_dim",
        "ratio",
        "rope_dim",
        "rope_odd",
        "weight_missing",
        "weight_short",
        "epsilon",
        "stride",
        "dtype",
    ],
)
def test_split_qkvgate_gemma_rmsnorm_rope_invalid_input(
    invalid_input,
):
    # Prevent malformed interleaving, missing weights and unsafe rotary widths.
    from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

    case = generate_test_data((3, 2, 1, 64, 32))
    if invalid_input == "width":
        case["input"] = torch.empty(3, 383)
    elif invalid_input == "head_dim":
        case["head_dim"] = 32.0
    elif invalid_input == "ratio":
        case["q_hidden_size"] = 96
    elif invalid_input == "rope_dim":
        case["rope_dim"] = 64
    elif invalid_input == "rope_odd":
        case["sin"] = torch.empty(3, 1, 1, 31)
    elif invalid_input == "weight_missing":
        case["k_weight"] = None
    elif invalid_input == "weight_short":
        case["q_weight"] = torch.ones(63)
    elif invalid_input == "epsilon":
        case["eps"] = -1e-6
    elif invalid_input == "stride":
        case["input"] = torch.empty(3, 768)[:, ::2]
    elif invalid_input == "dtype":
        case["input"] = case["input"].to(torch.int32)
    impl, _ = get_test_impls()
    require_test_impl(split_qkvgate_gemma_rmsnorm_rope, impl)
    with pytest.raises(AssertionError):
        split_qkvgate_gemma_rmsnorm_rope(**case, impl=impl)
