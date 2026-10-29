"""Public-interface tests for Kimi-K3 attention-residual scoring and mixing.

SGLang's srt/layers/attn_residual.py supplies the calling contract: get_cw
produces FP32 weights, AttnResidual owns a [N, capacity, H] snapshot bank,
and _mix_fused returns a mixture before the separate output RMSNorm. Token
shards may select a subset of the bank's token rows. That source identifies
the K3 hidden size as 7168. The token counts, capacities and valid-row counts
below are synthetic scenarios, not shapes collected from runtime traces.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# (tokens, capacity, valid bank rows, hidden size). K3 H from the SGLang
# source; decode/batched token and bank dimensions are synthetic selections.
perf_shape = [
    (1, 8, 3, 7168),
    (17, 16, 8, 7168),
    (128, 8, 1, 7168),
    (512, 8, 8, 7168),
]


# Shared by accuracy/performance tests and supplementary output checks.
attn_res_mix_fused_cases = [
    pytest.param((1, 8, 3, 7168), "random", "contiguous", id="k3-decode-scenario"),
    pytest.param((17, 16, 8, 7168), "random", "contiguous", id="k3-batch-scenario"),
    pytest.param((1, 0, 0, 1), "random", "contiguous", id="minimum-empty-bank"),
    pytest.param((3, 5, 0, 16), "unused_nan", "contiguous", id="no-valid-rows"),
    pytest.param((31, 3, 1, 31), "random", "contiguous", id="tokens-31-h31-nb2"),
    pytest.param((32, 3, 2, 33), "random", "contiguous", id="tokens-32-h33-nb4"),
    pytest.param((33, 3, 3, 127), "random", "contiguous", id="tokens-33-h127-full-nb4"),
    pytest.param((65, 5, 4, 64), "random", "contiguous", id="tokens-65-nb8"),
    pytest.param((7, 8, 8, 128), "random", "contiguous", id="full-nb16"),
    pytest.param((5, 4, 3, 64), "zero_inputs", "contiguous", id="zero-variance"),
    pytest.param((9, 7, 5, 128), "zero_weight", "contiguous", id="equal-scores"),
    pytest.param((3, 4, 3, 32), "identical_rows", "contiguous", id="pre-norm-mixture"),
    pytest.param((3, 4, 3, 32), "large_scores", "contiguous", id="stable-softmax"),
    pytest.param((5, 6, 2, 64), "unused_nan", "contiguous", id="ignore-unused-bank"),
    pytest.param((7, 5, 3, 128), "random", "padded", id="token-row-strides"),
    pytest.param((3, 0, 0, 64), "random", "padded", id="padded-empty-bank"),
]


# Required K3 coverage: all eight bank depths, larger token batches and SP views.
# Keep this separate from the contract cases to avoid large input snapshots.
k3_shape_cases = [
    pytest.param(
        (1, 8, valid_rows, 7168),
        "unused_nan" if valid_rows < 8 else "random",
        "contiguous",
        id=f"k3-bank-depth-{valid_rows}",
    )
    for valid_rows in range(1, 9)
] + [
    pytest.param(
        (128, 8, 1, 7168), "unused_nan", "contiguous", id="k3-batch-shallow-bank"
    ),
    pytest.param((512, 8, 8, 7168), "random", "contiguous", id="k3-prefill-full-bank"),
    pytest.param((17, 8, 7, 7168), "unused_nan", "sp_slice", id="k3-sp-nb8"),
    pytest.param((17, 8, 8, 7168), "random", "sp_slice", id="k3-sp-nb16"),
]


# Optional stress and cross-configuration coverage, not traced model workloads.
generalization_cases = [
    pytest.param(
        (2048, 8, 1, 7168), "unused_nan", "contiguous", id="large-prefill-shallow"
    ),
    pytest.param(
        (2049, 8, 8, 7168), "random", "contiguous", id="large-prefill-full-tail"
    ),
    pytest.param((33, 16, 9, 7168), "unused_nan", "contiguous", id="bank-depth-9"),
    pytest.param((33, 16, 15, 7168), "unused_nan", "contiguous", id="bank-nb16"),
    pytest.param((33, 16, 16, 7168), "random", "contiguous", id="bank-nb32"),
    pytest.param((33, 17, 17, 7168), "random", "contiguous", id="bank-depth-17"),
    pytest.param((17, 8, 7, 4096), "unused_nan", "contiguous", id="hidden-4096"),
    pytest.param((17, 8, 7, 8191), "unused_nan", "contiguous", id="hidden-8191"),
    pytest.param((17, 8, 8, 8192), "random", "contiguous", id="hidden-8192"),
    pytest.param((17, 8, 8, 8193), "random", "contiguous", id="hidden-8193"),
    pytest.param((17, 8, 7, 7168), "unused_nan", "padded", id="k3-padded-strides"),
]


def generate_test_data(shape, dtype, mode="random", layout="contiguous", seed=42):
    """Create independent finite rows, except deliberately unused NaN rows.

    Scaled random projection weights keep scores away from saturation. Other
    modes provide independent oracles for zero variance, equal scores, an
    unchanged common row, and numerically stable softmax with large logits.
    Padded views retain unit hidden stride while varying token/row strides.
    sp_slice selects a contiguous token shard with a nonzero storage offset,
    matching bank[start:stop] in AttnResidual.forward.
    """
    torch.npu.manual_seed(seed)
    tokens, capacity, valid_rows, hidden_size = shape
    if layout == "padded":
        prefix = torch.randn(tokens * 2, hidden_size + 3, dtype=dtype)[
            1::2, :hidden_size
        ]
        bank = torch.randn(tokens * 2, capacity * 2 + 1, hidden_size + 3, dtype=dtype)[
            1::2, : capacity * 2 : 2, :hidden_size
        ]
    elif layout == "sp_slice":
        # Model rank 1's interval in a larger contiguous token buffer. The bank
        # retains its parent's row strides, as in the framework's SP fallback.
        start = tokens
        prefix = torch.randn(tokens * 3, hidden_size, dtype=dtype)[
            start : start + tokens
        ]
        bank = torch.randn(tokens * 3, capacity, hidden_size, dtype=dtype)[
            start : start + tokens
        ]
    else:
        assert layout == "contiguous"
        prefix = torch.randn(tokens, hidden_size, dtype=dtype)
        bank = torch.randn(tokens, capacity, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=torch.float32) / hidden_size**0.5
    if mode == "zero_inputs":
        prefix.zero_()
        bank.zero_()
    elif mode == "zero_weight":
        weight.zero_()
    elif mode == "identical_rows":
        # Nonunit magnitude also detects an accidental final output RMSNorm.
        prefix.fill_(2)
        bank.copy_(prefix[:, None, :].expand_as(bank))
    elif mode == "large_scores":
        assert valid_rows > 0
        prefix.fill_(-1)
        bank.fill_(-1)
        bank[:, 0].fill_(1)
        weight.fill_(128)
    elif mode == "unused_nan":
        assert valid_rows < capacity
        bank[:, valid_rows:].fill_(float("nan"))
    else:
        assert mode == "random"
    return prefix, bank, weight


def _check_output_contract(result, inputs, snapshots, valid_rows, mode):
    prefix, bank, _ = inputs
    assert isinstance(result, torch.Tensor)
    assert result.shape == prefix.shape
    assert result.dtype == prefix.dtype
    assert result.device == prefix.device
    assert result.is_contiguous()
    assert result.untyped_storage().data_ptr() not in [
        value.untyped_storage().data_ptr() for value in inputs
    ]
    assert torch.isfinite(result).all()
    for value, before in zip(inputs, snapshots):
        torch.testing.assert_close(value, before, atol=0, rtol=0, equal_nan=True)
    if valid_rows == 0 or mode in ("zero_inputs", "identical_rows"):
        torch.testing.assert_close(result, prefix, atol=0, rtol=0)
    elif mode == "zero_weight":
        expected = (bank[:, :valid_rows].float().sum(dim=1) + prefix.float()) / (
            valid_rows + 1
        )
        # The Triton path sums scaled rows sequentially; allow FP32 reduction
        # rounding before converting back to the output dtype.
        assert_accuracy(result, expected.to(prefix.dtype))
    elif mode == "large_scores":
        torch.testing.assert_close(result, bank[:, 0].to(prefix.dtype), atol=0, rtol=0)


# Required accuracy and selected performance checks.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,mode,layout", attn_res_mix_fused_cases)
def test_attn_res_mix_fused(
    shape,
    dtype,
    mode,
    layout,
):
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix, bank, weight = generate_test_data(shape, dtype, mode, layout)
    executor = check_accuracy(
        attn_res_mix_fused, args=(prefix, bank, shape[2], weight, 1e-6)
    )
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and dtype in (torch.float16, torch.bfloat16)),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,mode,layout", k3_shape_cases)
def test_attn_res_mix_fused_k3_shapes(shape, dtype, mode, layout):
    """Required synthetic decode, prefill and SP cases at K3's hidden size."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix, bank, weight = generate_test_data(shape, dtype, mode, layout)
    executor = check_accuracy(
        attn_res_mix_fused, args=(prefix, bank, shape[2], weight, 1e-6)
    )
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("valid_rows", [7, 8], ids=["nb8", "nb16"])
@pytest.mark.parametrize(
    "core_multiple,token_offset",
    [
        pytest.param(1, -1, id="cores-minus-one"),
        pytest.param(1, 0, id="cores"),
        pytest.param(1, 1, id="cores-plus-one"),
        pytest.param(2, 1, id="two-cores-plus-one"),
    ],
)
def test_attn_res_mix_fused_core_boundaries(
    dtype, valid_rows, core_multiple, token_offset
):
    """Check core scheduling boundaries together with B=7/8 softmax sizes."""
    # Query the same utility as the kernel at execution time, after the NPU
    # fixture selects the device. Collection must not initialize the runtime.
    from sgl_kernel_npu.implementation.triton.utils.triton_utils import (
        get_device_properties,
    )

    _, vector_cores = get_device_properties()
    tokens = max(1, core_multiple * vector_cores + token_offset)
    shape = (tokens, 8, valid_rows, 7168)
    mode = "unused_nan" if valid_rows < 8 else "random"
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix, bank, weight = generate_test_data(shape, dtype, mode, "contiguous")
    executor = check_accuracy(
        attn_res_mix_fused, args=(prefix, bank, shape[2], weight, 1e-6)
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "prefix_dtype,bank_dtype,weight_dtype",
    [
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float32, torch.float16, torch.bfloat16),
        (torch.float16, torch.float32, torch.float16),
    ],
)
def test_attn_res_mix_fused_mixed_dtypes(
    prefix_dtype,
    bank_dtype,
    weight_dtype,
):
    """Check accuracy when prefix, bank and scoring weight use different dtypes."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    shape = (5, 4, 3, 64)
    prefix, bank, weight = generate_test_data(shape, prefix_dtype, mode="unused_nan")
    executor = check_accuracy(
        attn_res_mix_fused,
        args=(prefix, bank.to(bank_dtype), shape[2], weight.to(weight_dtype), 1e-6),
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("epsilon", [1e-12, 1e-3, 1])
def test_attn_res_mix_fused_variance_epsilon(epsilon):
    """Check RMS scoring with tiny inputs and small or large epsilon values."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix, bank, weight = generate_test_data((3, 4, 3, 32), torch.float32)
    prefix.mul_(1e-4)
    bank.mul_(1e-4)
    executor = check_accuracy(
        attn_res_mix_fused, args=(prefix, bank, 3, weight, epsilon)
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_device", ["cpu", "npu"])
@pytest.mark.parametrize("capacity,valid_rows", [(0, 0), (5, 3)])
def test_attn_res_mix_fused_empty_tokens(
    dtype,
    input_device,
    capacity,
    valid_rows,
):
    """Check empty CPU/NPU inputs across dtypes and bank capacities."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    # Empty inputs bypass nonempty Triton stride/device launch restrictions.
    prefix = torch.empty(0, 6, dtype=dtype, device=input_device)[:, ::2]
    bank = torch.empty(0, capacity, 6, dtype=dtype, device=input_device)[..., ::2]
    weight = torch.empty(6, dtype=torch.float32, device=input_device)[::2]
    check_accuracy(attn_res_mix_fused, args=(prefix, bank, valid_rows, weight, 1e-6))
    # There is no device task to profile, even when --enable-perf=true.


# Supplementary checks, excluded by default.
@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,mode,layout", generalization_cases)
def test_attn_res_mix_fused_generalization(shape, dtype, mode, layout):
    """Check large batches, hidden-size boundaries, deep banks and strides."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix, bank, weight = generate_test_data(shape, dtype, mode, layout)
    executor = check_accuracy(
        attn_res_mix_fused, args=(prefix, bank, shape[2], weight, 1e-6)
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "prefix_dtype,bank_dtype,weight_dtype",
    [
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float32, torch.float16, torch.bfloat16),
        (torch.float16, torch.float32, torch.float16),
    ],
)
def test_attn_res_mix_fused_mixed_dtypes_k3(
    prefix_dtype,
    bank_dtype,
    weight_dtype,
):
    """Check mixed-dtype accuracy at K3's hidden size of 7168."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    shape = (17, 8, 7, 7168)
    prefix, bank, weight = generate_test_data(shape, prefix_dtype, mode="unused_nan")
    executor = check_accuracy(
        attn_res_mix_fused,
        args=(prefix, bank.to(bank_dtype), shape[2], weight.to(weight_dtype), 1e-6),
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,mode,layout", attn_res_mix_fused_cases)
def test_attn_res_mix_fused_output_contract(shape, dtype, mode, layout):
    """Check output metadata, separate storage, unchanged inputs and exact cases."""
    if get_skip_acc_check() is True:
        pytest.skip("Output-contract checks are disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    inputs = generate_test_data(shape, dtype, mode, layout)
    snapshots = tuple(value.clone() for value in inputs)
    prefix, bank, weight = inputs
    impl, _ = get_test_impls()
    require_test_impl(attn_res_mix_fused, impl)
    result = attn_res_mix_fused(prefix, bank, shape[2], weight, 1e-6, impl=impl)
    _check_output_contract(result, inputs, snapshots, shape[2], mode)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_device", ["cpu", "npu"])
@pytest.mark.parametrize("capacity,valid_rows", [(0, 0), (5, 3)])
def test_attn_res_mix_fused_empty_output_contract(
    dtype, input_device, capacity, valid_rows
):
    """Check that empty calls return the original prefix object."""
    if get_skip_acc_check() is True:
        pytest.skip("Output-contract checks are disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix = torch.empty(0, 6, dtype=dtype, device=input_device)[:, ::2]
    bank = torch.empty(0, capacity, 6, dtype=dtype, device=input_device)[..., ::2]
    weight = torch.empty(6, dtype=torch.float32, device=input_device)[::2]
    impl, _ = get_test_impls()
    require_test_impl(attn_res_mix_fused, impl)
    assert (
        attn_res_mix_fused(prefix, bank, valid_rows, weight, 1e-6, impl=impl) is prefix
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input,exception",
    [
        ("prefix_not_tensor", AssertionError),
        ("bank_not_tensor", AssertionError),
        ("weight_not_tensor", AssertionError),
        ("prefix_rank", AssertionError),
        ("bank_rank", AssertionError),
        ("weight_rank", AssertionError),
        ("token_mismatch", AssertionError),
        ("hidden_mismatch", AssertionError),
        ("weight_length", AssertionError),
        ("zero_hidden", AssertionError),
        ("prefix_dtype", AssertionError),
        ("bank_dtype", AssertionError),
        ("weight_dtype", AssertionError),
        ("device_mismatch", AssertionError),
        ("blocks_negative", ValueError),
        ("blocks_exceed_capacity", ValueError),
        ("blocks_float", AssertionError),
        ("blocks_bool", AssertionError),
        ("epsilon_zero", AssertionError),
        ("epsilon_negative", AssertionError),
        ("epsilon_nan", AssertionError),
        ("epsilon_inf", AssertionError),
        ("epsilon_bool", AssertionError),
        ("epsilon_tensor", AssertionError),
        ("prefix_hidden_stride", AssertionError),
        ("bank_hidden_stride", AssertionError),
        ("weight_stride", AssertionError),
        ("nonempty_cpu", AssertionError),
        ("empty_invalid_blocks", ValueError),
        ("empty_invalid_epsilon", AssertionError),
        ("empty_device_mismatch", AssertionError),
    ],
)
def test_attn_res_mix_fused_invalid_input(
    invalid_input,
    exception,
):
    """Check Triton rejects invalid types, shapes, devices, strides and options."""
    from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

    prefix, bank, weight = generate_test_data((3, 4, 3, 32), torch.float16)
    valid_rows, epsilon = 3, 1e-6
    if invalid_input == "prefix_not_tensor":
        prefix = None
    elif invalid_input == "bank_not_tensor":
        bank = None
    elif invalid_input == "weight_not_tensor":
        weight = None
    elif invalid_input == "prefix_rank":
        prefix = prefix.unsqueeze(0)
    elif invalid_input == "bank_rank":
        bank = bank.flatten(1)
    elif invalid_input == "weight_rank":
        weight = weight.unsqueeze(0)
    elif invalid_input == "token_mismatch":
        bank = bank[:2]
    elif invalid_input == "hidden_mismatch":
        bank = bank[..., :16]
    elif invalid_input == "weight_length":
        weight = weight[:16]
    elif invalid_input == "zero_hidden":
        prefix, bank, weight = prefix[:, :0], bank[..., :0], weight[:0]
    elif invalid_input == "prefix_dtype":
        prefix = prefix.to(torch.int32)
    elif invalid_input == "bank_dtype":
        bank = bank.to(torch.int32)
    elif invalid_input == "weight_dtype":
        weight = weight.to(torch.int32)
    elif invalid_input == "device_mismatch":
        weight = weight.cpu()
    elif invalid_input == "blocks_negative":
        valid_rows = -1
    elif invalid_input == "blocks_exceed_capacity":
        valid_rows = 5
    elif invalid_input == "blocks_float":
        valid_rows = 3.0
    elif invalid_input == "blocks_bool":
        valid_rows = True
    elif invalid_input == "epsilon_zero":
        epsilon = 0
    elif invalid_input == "epsilon_negative":
        epsilon = -1e-6
    elif invalid_input == "epsilon_nan":
        epsilon = float("nan")
    elif invalid_input == "epsilon_inf":
        epsilon = float("inf")
    elif invalid_input == "epsilon_bool":
        epsilon = True
    elif invalid_input == "epsilon_tensor":
        epsilon = torch.tensor(1e-6)
    elif invalid_input == "prefix_hidden_stride":
        prefix = torch.empty(3, 64, dtype=prefix.dtype)[:, ::2]
    elif invalid_input == "bank_hidden_stride":
        bank = torch.empty(3, 4, 64, dtype=bank.dtype)[..., ::2]
    elif invalid_input == "weight_stride":
        weight = torch.empty(64, dtype=weight.dtype)[::2]
    elif invalid_input == "nonempty_cpu":
        prefix, bank, weight = prefix.cpu(), bank.cpu(), weight.cpu()
    elif invalid_input == "empty_invalid_blocks":
        prefix, bank, valid_rows = prefix[:0], bank[:0], 5
    elif invalid_input == "empty_invalid_epsilon":
        prefix, bank, epsilon = prefix[:0], bank[:0], 0
    elif invalid_input == "empty_device_mismatch":
        prefix, bank, weight = prefix[:0], bank[:0], weight.cpu()
    else:
        raise AssertionError(f"Unhandled invalid input: {invalid_input}")
    with pytest.raises(exception):
        attn_res_mix_fused(prefix, bank, valid_rows, weight, epsilon, impl="triton")
