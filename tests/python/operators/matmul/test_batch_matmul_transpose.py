"""Dispatch, accuracy and performance through the public matmul API."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# SGLang MLA/DSA post-attention: shape=(tokens, local_heads, kv_lora_rank,
# v_head_dim), local_heads=num_attention_heads/effective_attention_tp.
# On A2/A3, MLA uses AscendC when M, B*K or N reaches 65536. DSA uses
# TorchNPU and pre-transposes activations when B*K reaches 65536.
# Keep the latest model in each family with identical shapes.
# (Model ID, total attention heads, K, N, representative attention TP sizes.)
MODEL_PROFILES = [
    ("deepseek-v3.2", 128, 512, 128, (1, 2, 4, 8)),
    ("glm-5.3-flash", 64, 512, 256, (1, 2, 4, 8)),
]
# Representative loads, not captured traces or fixed per-model batch sizes.
# M counts operator input tokens, including flattened verification tokens.
MODEL_TOKEN_COUNTS = (1, 8, 36, 128, 1024)
MODEL_SHAPES = [
    (
        (tokens, heads // tp, k, n),
        f"{model}-attn-tp{tp}-tokens{tokens}",
    )
    for model, heads, k, n, attention_tp_sizes in MODEL_PROFILES
    for tp in attention_tp_sizes
    for tokens in MODEL_TOKEN_COUNTS
]
# Exercise both real output widths around the M=16 tile boundary.
MODEL_SHAPES += [
    (
        (tokens, heads // 8, k, n),
        f"{model}-attn-tp8-tokens{tokens}",
    )
    for model, heads, k, n, _ in MODEL_PROFILES
    for tokens in (15, 16, 17)
]
GENERIC_SHAPES = [
    ((1, 1, 1, 1), "minimum"),
    ((2, 3, 4, 5), "small"),
    ((7, 3, 31, 33), "unaligned-k-n-tail"),
    ((15, 2, 16, 16), "below-m-tile"),
    ((17, 3, 33, 17), "above-m-k-n-tiles"),
    ((1024, 1, 16, 16), "maximum-m"),
]
ND_CASES = [
    pytest.param(shape, mode, id=f"{shape_id}-{mode}")
    for mode, shapes in (
        ("random", MODEL_SHAPES + GENERIC_SHAPES),
        ("zero", GENERIC_SHAPES),
    )
    for shape, shape_id in shapes
]
# Generic shapes exercise zero products in the default suite.
ND_CASES += [
    pytest.param(shape, "zero", id=f"{shape_id}-zero", marks=pytest.mark.extra_checks)
    for shape, shape_id in MODEL_SHAPES
]
PERF_SHAPES = [(1, 128, 512, 128), (8, 16, 512, 128), (36, 128, 512, 128)]


def generate_test_data(shape, dtype, mode):
    """Generate inputs for random and exact-zero products."""
    torch.npu.manual_seed(42)
    m, batch, k, n = shape
    a = torch.randn((m, batch, k), dtype=dtype) * 0.1
    b = torch.randn((batch, k, n), dtype=dtype) * 0.1
    if mode == "zero":
        a.zero_()
    return a, b


def run_case(shape, dtype, mode):
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

    a, b = generate_test_data(shape, dtype, mode)
    a_before, b_before = a.clone(), b.clone()
    impl, impl_ref = get_test_impls()
    # Cover AscendC tiling boundaries, including shapes auto sends to TorchNPU.
    if impl == "auto":
        impl = "ascendc"
    if impl_ref == "auto":
        impl_ref = "ascendc"
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(batch_matmul_transpose, impl)
    else:
        require_distinct_impls(batch_matmul_transpose, impl, impl_ref)
        reference = batch_matmul_transpose(a.clone(), b.clone(), impl=impl_ref)
        output = batch_matmul_transpose(a, b, impl=impl)
        m, batch, _, n = shape
        assert output.shape == (m, batch, n)
        assert output.dtype == a.dtype
        assert output.device == a.device
        assert_accuracy(output, reference)
        assert torch.equal(a, a_before)
        assert torch.equal(b, b_before)
        if mode == "zero":
            assert torch.count_nonzero(output) == 0
    executor = make_executor(
        batch_matmul_transpose,
        args=(a, b),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(
        executor,
        enable_perf=(
            shape in PERF_SHAPES and dtype == torch.bfloat16 and mode == "random"
        ),
    )


@pytest.mark.parametrize("shape, mode", ND_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_matmul_transpose_nd(
    shape,
    dtype,
    mode,
):
    run_case(shape, dtype, mode)


@pytest.mark.parametrize("generation, arch_id", [("ARCH22", 2201), ("ARCH35", 3510)])
@pytest.mark.parametrize(
    "shape, is_dsa, impl, arch22_route",
    [
        pytest.param((2, 16, 512, 128), False, "auto", "vendor", id="mla-small"),
        pytest.param((2, 128, 512, 128), False, "auto", "ascendc", id="mla-bk-limit"),
        pytest.param((65536, 1, 1, 1), False, "auto", "ascendc", id="mla-m-limit"),
        pytest.param((1, 1, 1, 65536), False, "auto", "ascendc", id="mla-n-limit"),
        pytest.param((2, 16, 512, 128), True, "auto", "vendor", id="dsa-small"),
        pytest.param((2, 128, 512, 128), True, "auto", "transpose", id="dsa-bk-limit"),
        pytest.param(
            (2, 128, 512, 128), False, "torch_npu", "vendor", id="dsa-query-projection"
        ),
    ],
)
def test_batch_matmul_transpose_dispatch(
    monkeypatch,
    generation,
    arch_id,
    shape,
    is_dsa,
    impl,
    arch22_route,
):
    """Check backend dispatch and input layout without launching NPU kernels."""
    import torch_npu
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose
    from sgl_kernel_npu.operators.utils.generation import Generation, Hardware

    module = import_module("sgl_kernel_npu.operators.matmul.batch_matmul_transpose")
    hardware = Hardware(arch_id, Generation[generation])
    monkeypatch.setattr(module, "get_hardware", lambda: hardware)

    m, batch, k, n = shape
    a = torch.randn((m, batch, k * 2), dtype=torch.bfloat16, device="cpu")[..., ::2]
    if is_dsa:
        b = torch.zeros((batch, n, k), dtype=a.dtype, device="cpu").transpose(1, 2)
    else:
        b = torch.zeros((batch, k, n), dtype=a.dtype, device="cpu")
    vendor_output = a.new_ones((m, batch, n))
    vendor = Mock(return_value=vendor_output)
    ascendc = Mock(side_effect=lambda a, b, out: out.fill_(2))
    monkeypatch.setattr(torch_npu, "npu_transpose_batchmatmul", vendor)
    monkeypatch.setattr(
        torch.ops, "npu", SimpleNamespace(batch_matmul_transpose=ascendc)
    )

    output = batch_matmul_transpose(a, b, is_dsa=is_dsa, impl=impl)
    route = arch22_route if generation == "ARCH22" else "vendor"
    if route == "ascendc":
        vendor.assert_not_called()
        ascendc.assert_called_once()
        assert ascendc.call_args.args[0].is_contiguous()
        assert ascendc.call_args.args[2] is output
        torch.testing.assert_close(output, torch.full_like(vendor_output, 2))
    else:
        ascendc.assert_not_called()
        vendor.assert_called_once()
        activation, weight = vendor.call_args.args
        assert weight is b
        perm_x1 = (1, 0, 2)
        if route == "transpose":
            assert activation.is_contiguous()
            torch.testing.assert_close(activation, a.transpose(0, 1))
            perm_x1 = (0, 1, 2)
        else:
            assert activation is a
        assert vendor.call_args.kwargs == {
            "perm_x1": perm_x1,
            "perm_x2": (0, 1, 2),
            "perm_y": (1, 0, 2),
        }
        assert output is vendor_output


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "dtype",
        "batch",
        "k",
        "noncontiguous-b",
        "empty",
        "m-limit",
    ],
)
def test_batch_matmul_transpose_rejects_invalid_input(
    invalid,
):
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

    impl, _ = get_test_impls()
    if impl == "auto":
        impl = "ascendc"
    elif impl == "torch_npu":
        pytest.skip("These cases exercise AscendC and torch_native input limits")
    require_test_impl(batch_matmul_transpose, impl)
    a, b = generate_test_data((3, 2, 32, 32), torch.float16, "random")
    if invalid == "rank":
        a = a.flatten()
    elif invalid == "dtype":
        b = b.float()
    elif invalid == "batch":
        b = b[:1]
    elif invalid == "k":
        a = a[..., :16].contiguous()
    elif invalid == "noncontiguous-b":
        b = b.transpose(1, 2)
    elif invalid == "empty":
        a = a[:0]
    elif invalid == "m-limit":
        a = torch.empty((1025, 2, 32), dtype=a.dtype)
    with pytest.raises((ValueError, RuntimeError)):
        batch_matmul_transpose(a, b, impl=impl)
