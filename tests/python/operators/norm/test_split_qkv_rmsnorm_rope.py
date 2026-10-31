"""QKV layouts derived from SGLang Qwen3, Qwen3-MoE, GLM4-MoE and Llama.

The forward_prepare_npu calls use q_size=(global Q heads / attention TP)*D
and kv_size=max(1, global KV heads / attention TP)*D. RotaryEmbedding's
get_cos_sin_with_position builds contiguous [N,1,1,R] duplicated-half tables.
GLM's fused path is decode-only; its larger N cases mean batched decode.
Callers: srt/models/{qwen3,qwen3_moe,glm4_moe,llama,dflash}.py.
Inspected SGLang revision: 2456c9111a121a02e41e625e3930e1eb143fe9bf.
Model dimensions:
https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
https://huggingface.co/zai-org/GLM-4.5/blob/aa08a0da0888d6ba56175389d89b90eec6dc8ebe/config.json
The Llama no-norm cases use representative 32/8-head layouts, not a traced checkpoint.
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

# (model/scenario, global Q heads, global KV heads, D, R, eps, TP, token loads).
# These are possible call shapes derived from the source, not measured traces.
model_cases = [
    ("qwen3-8b-tp1", 32, 8, 128, 128, 1e-6, 1, (1, 16, 128, 2048)),
    ("qwen3-8b-tp8", 32, 8, 128, 128, 1e-6, 8, (1, 16, 128, 2048)),
    ("qwen3-8b-tp16-kv-replication", 32, 8, 128, 128, 1e-6, 16, (1, 16, 128, 2048)),
    ("qwen3-235b-tp1", 64, 4, 128, 128, 1e-6, 1, (1, 16, 128, 2048)),
    ("qwen3-235b-tp4", 64, 4, 128, 128, 1e-6, 4, (1, 16, 128, 2048)),
    ("qwen3-235b-tp8-kv-replication", 64, 4, 128, 128, 1e-6, 8, (1, 16, 128, 2048)),
    ("glm45-tp1-decode", 96, 8, 128, 64, 1e-5, 1, (1, 16, 128)),
    ("glm45-tp8-decode", 96, 8, 128, 64, 1e-5, 8, (1, 16, 128)),
    ("llama-layout-tp1-no-qk-norm", 32, 8, 128, 128, None, 1, (1, 16, 128, 2048)),
    ("llama-layout-tp8-no-qk-norm", 32, 8, 128, 128, None, 8, (1, 16, 128, 2048)),
]
production_cases = []
perf_shape = []
for model, qh, kh, dim, rotary_dim, norm_eps, tp, loads in model_cases:
    for tokens in loads:
        shape = (tokens, qh // tp, max(1, kh // tp), dim, rotary_dim)
        production_cases.append(
            pytest.param(shape, norm_eps, id=f"{model}-tokens{tokens}")
        )
        perf_shape.append(shape)


def generate_test_data(shape, dtype=torch.bfloat16, eps=1e-6, **kwargs):
    return generate_qkv_data(shape, dtype, norm=eps is not None, eps=eps, **kwargs)


# Required: TP-sharded GQA, replicated KV heads, partial RoPE and no-norm paths.
@pytest.mark.parametrize("shape,eps", production_cases)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_split_qkv_rmsnorm_rope_production(
    shape,
    eps,
    cache_dtype,
):
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

    case = generate_test_data(shape, eps=eps, cache_dtype=cache_dtype)
    executor = check_accuracy(split_qkv_rmsnorm_rope, kwargs=case)
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape",
    [(0, 2, 1, 64, 32), (1, 1, 1, 2, 2), (7, 3, 1, 96, 48), (33, 4, 2, 192, 96)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_split_qkv_rmsnorm_rope_generalization(
    shape,
    dtype,
    is_neox_style,
    bias,
):
    # Non-power-of-two heads, both rotary pairings, affine bias and empty inputs.
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

    case = generate_test_data(shape, dtype, bias=bias, weight_dtype=torch.float32)
    case["is_neox_style"] = is_neox_style
    executor = check_accuracy(split_qkv_rmsnorm_rope, kwargs=case)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("is_neox_style", [False, True])
def test_split_qkv_rmsnorm_rope_output_contract(
    is_neox_style,
):
    # An identity rotation must preserve unnormalized Q/K and copy V exactly.
    if get_skip_acc_check() is True:
        pytest.skip("Output checks disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

    case = generate_test_data((3, 2, 1, 64, 32), eps=None)
    case["is_neox_style"] = is_neox_style
    case["sin"].zero_()
    case["cos"].fill_(1)
    if not is_neox_style:
        # GPT-J consumes only the first half of the duplicated-half tables.
        case["sin"][..., 16:] = 7
        case["cos"][..., 16:] = -3
    source = case["input"]
    snapshot = source.clone()
    expected = source.split((128, 64, 64), dim=-1)
    impl, _ = get_test_impls()
    require_test_impl(split_qkv_rmsnorm_rope, impl)
    outputs = split_qkv_rmsnorm_rope(**case, impl=impl)
    for output, wanted in zip(outputs, expected):
        assert output.dtype == source.dtype and output.device == source.device
        assert output.is_contiguous()
        assert (
            output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(output, wanted, atol=0, rtol=0)
    torch.testing.assert_close(source, snapshot, atol=0, rtol=0)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input",
    [
        "width",
        "head_dim",
        "ratio",
        "sin_width",
        "cos_width",
        "sin_rows",
        "weight_missing",
        "weight_short",
        "bias_pair",
        "epsilon",
        "stride",
        "trig_stride",
        "dtype",
        "style",
    ],
)
def test_split_qkv_rmsnorm_rope_invalid_input(
    invalid_input,
):
    # Reject inconsistent widths, incomplete affine parameters and unsafe strides.
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

    case = generate_test_data((3, 2, 1, 64, 32))
    if invalid_input == "width":
        case["input"] = torch.empty(3, 255)
    elif invalid_input == "head_dim":
        case["head_dim"] = 63
    elif invalid_input == "ratio":
        case["q_hidden_size"] = 96
    elif invalid_input == "sin_width":
        case["sin"] = torch.empty(3, 1, 1, 31)
    elif invalid_input == "cos_width":
        case["cos"] = torch.empty(3, 1, 1, 16)
    elif invalid_input == "sin_rows":
        case["sin"] = torch.empty(2, 1, 1, 32)
    elif invalid_input == "weight_missing":
        case["k_weight"] = None
    elif invalid_input == "weight_short":
        case["q_weight"] = torch.ones(63)
    elif invalid_input == "bias_pair":
        case["q_bias"] = torch.zeros(64)
    elif invalid_input == "epsilon":
        case["eps"] = float("nan")
    elif invalid_input == "stride":
        case["input"] = torch.empty(3, 512)[:, ::2]
    elif invalid_input == "trig_stride":
        case["sin"] = torch.empty(3, 1, 1, 64)[..., ::2]
    elif invalid_input == "dtype":
        case["input"] = case["input"].to(torch.int32)
    elif invalid_input == "style":
        case["is_neox_style"] = 1
    impl, _ = get_test_impls()
    require_test_impl(split_qkv_rmsnorm_rope, impl)
    with pytest.raises(AssertionError):
        split_qkv_rmsnorm_rope(**case, impl=impl)
