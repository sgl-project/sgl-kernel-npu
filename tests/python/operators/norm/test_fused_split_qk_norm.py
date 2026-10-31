"""MLA latent projection shapes from SGLang's DeepSeek NPU attention path.

Source: srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py,
both fused_split_qk_norm calls, and models/deepseek_v2.py's replicated
fused_qkv_a_proj_with_mqa output. DeepSeek-V3 config supplies ranks 1536/512
and rotary width 64; these are not divided by attention TP. Tokens below
model decode and prefill loads, not collected runtime traces.
Inspected SGLang revision: 2456c9111a121a02e41e625e3930e1eb143fe9bf.
Model dimensions: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.norm import generate_mla_data
from tests.python.utils.performance import check_performance

perf_shape = [(tokens, 1536, 512, 64) for tokens in (1, 16, 128, 2048)]


def generate_test_data(shape, dtype=torch.bfloat16, **kwargs):
    return generate_mla_data(shape, dtype, **kwargs)


# Required: every production-derived shape checks accuracy and measures performance.
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(perf_shape[0], id="deepseek-v3-single-decode"),
        pytest.param(perf_shape[1], id="deepseek-v3-batched-decode"),
        pytest.param(perf_shape[2], id="deepseek-v3-short-prefill"),
        pytest.param(perf_shape[3], id="deepseek-v3-chunked-prefill"),
    ],
)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_fused_split_qk_norm_production(
    shape,
    weight_dtype,
):
    from sgl_kernel_npu.operators.norm import fused_split_qk_norm

    case = generate_test_data(shape, weight_dtype=weight_dtype)
    executor = check_accuracy(fused_split_qk_norm, kwargs=case)
    check_performance(executor, enable_perf=shape in perf_shape)


# Supplementary checks are excluded by the repository's default marker expression.
@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape", [(0, 32, 16, 8), (1, 1, 1, 1), (7, 96, 48, 12), (33, 1536, 512, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bias_mode", ["none", "q", "k", "both"])
def test_fused_split_qk_norm_generalization(
    shape,
    dtype,
    bias_mode,
):
    # Exercise independent biases, empty rows and non-power-of-two rank tails.
    from sgl_kernel_npu.operators.norm import fused_split_qk_norm

    executor = check_accuracy(
        fused_split_qk_norm,
        kwargs=generate_test_data(shape, dtype, bias_mode=bias_mode),
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("zero_input", [False, True])
def test_fused_split_qk_norm_output_contract(
    zero_input,
):
    # Verify exact PE copy, singleton K dimensions, independent outputs and no mutation.
    if get_skip_acc_check() is True:
        pytest.skip("Output checks disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.norm import fused_split_qk_norm

    case = generate_test_data((7, 32, 16, 8), bias_mode="both")
    source = case["fused_qkv_a_proj_out"]
    if zero_input:
        source.zero_()
    snapshot = source.clone()
    impl, _ = get_test_impls()
    require_test_impl(fused_split_qk_norm, impl)
    q, k, pe = fused_split_qk_norm(**case, impl=impl)
    assert q.shape == (7, 32) and k.shape == (7, 1, 16) and pe.shape == (7, 1, 8)
    torch.testing.assert_close(pe[:, 0], source[:, -8:], atol=0, rtol=0)
    for output in (q, k, pe):
        assert output.dtype == source.dtype and output.device == source.device
        assert (
            output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
    if zero_input:
        assert_accuracy(q, case["q_a_layernorm"].bias.to(q.dtype).expand_as(q))
        assert_accuracy(
            k[:, 0], case["kv_a_layernorm"].bias.to(k.dtype).expand_as(k[:, 0])
        )
    torch.testing.assert_close(source, snapshot, atol=0, rtol=0)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input",
    [
        "q_rank",
        "kv_rank",
        "rope_dim",
        "width",
        "q_weight",
        "k_bias",
        "epsilon",
        "stride",
        "dtype",
    ],
)
def test_fused_split_qk_norm_invalid_input(
    invalid_input,
):
    # Metadata checks must reject malformed splits and affine buffers before launch.
    from sgl_kernel_npu.operators.norm import fused_split_qk_norm

    case = generate_test_data((3, 32, 16, 8), bias_mode="both")
    if invalid_input == "q_rank":
        case["q_lora_rank"] = 0
    elif invalid_input == "kv_rank":
        case["kv_lora_rank"] = -1
    elif invalid_input == "rope_dim":
        case["qk_rope_dim"] = True
    elif invalid_input == "width":
        case["fused_qkv_a_proj_out"] = torch.empty(3, 57)
    elif invalid_input == "q_weight":
        case["q_a_layernorm"].weight = torch.ones(31)
    elif invalid_input == "k_bias":
        case["kv_a_layernorm"].bias = torch.zeros(15)
    elif invalid_input == "epsilon":
        case["eps"] = 0
    elif invalid_input == "stride":
        case["fused_qkv_a_proj_out"] = torch.empty(3, 112)[:, ::2]
    elif invalid_input == "dtype":
        case["fused_qkv_a_proj_out"] = case["fused_qkv_a_proj_out"].to(torch.int32)
    impl, _ = get_test_impls()
    require_test_impl(fused_split_qk_norm, impl)
    with pytest.raises(AssertionError):
        fused_split_qk_norm(**case, impl=impl)
