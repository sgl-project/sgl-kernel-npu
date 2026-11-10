"""Top-K set comparison against independent dequantized PyTorch math."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, make_executor
from tests.python.utils.performance import check_performance

CASES = [
    ((513,), (1,), (3,)),
    ((513,), (128,), (0,)),
    pytest.param(((513, 257), (2, 3), (0, 1)), marks=pytest.mark.extra_checks),
    pytest.param(((33,), (4,), (0,)), marks=pytest.mark.extra_checks),
    pytest.param(((1,), (4,), (0,)), marks=pytest.mark.extra_checks),
]
CASE_IDS = [
    "deepseek-v4-flash-c4-decode-q1-n64-d128-topk512-page32",
    "deepseek-v4-flash-c4-prefill-q128-n64-d128-topk512-page32",
    "synthetic-ragged-prefill",
    "synthetic-page-boundary",
    "synthetic-empty-causal-prefix",
]


def generate_data(case, mode):
    lengths, queries, residuals = case
    dtype = torch.float8_e4m3fn if mode == 1 else torch.int8
    scale_dtype = torch.float32 if mode == 1 else torch.float16
    # A5 QuantLightningIndexer requires FP32 weights; A3 uses FP16.
    weight_dtype = torch.float32 if mode == 1 else torch.float16
    pages_per_batch = (max(lengths) + 31) // 32
    pages = len(lengths) * pages_per_batch
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(
        len(lengths), pages_per_batch
    )
    q = torch.zeros(sum(queries), 64, 128, device="cpu")
    q[:, :32, :4] = 1
    q[:, 32:, :4] = -1
    k = torch.zeros(pages, 32, 1, 128, device="cpu")
    k[..., :4] = 1
    scales = torch.ones(pages, 32, 1, dtype=scale_dtype)
    for b, length in enumerate(lengths):
        for j in range(length):
            scales[int(table[b, j // 32]), j % 32, 0] = 2 ** (j / 64)
    args = (
        q.to(dtype).view(torch.uint8).to(table.device).view(dtype),
        k.to(dtype).view(torch.uint8).to(table.device).view(dtype),
        torch.ones(sum(queries), 64, dtype=weight_dtype),
        torch.full((sum(queries), 64), 0.5, dtype=scale_dtype),
        scales,
        512,
        mode,
    )
    cu = torch.tensor([0, *queries], dtype=torch.int32).cumsum(0).to(torch.int32)
    common = dict(
        cu_seqlens_q=cu,
        seqused_k=torch.tensor(lengths, dtype=torch.int32),
        cmp_residual_k=torch.tensor(residuals, dtype=torch.int32),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )
    return args, common, table


def hardware_mode():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    return 1 if get_hardware().generation == Generation.ARCH35 else 2


def _cpu_tensor(value):
    if not isinstance(value, torch.Tensor):
        return value
    if str(value.dtype).startswith("torch.float8_"):
        return value.view(torch.uint8).cpu().view(value.dtype)
    return value.cpu()


def reference_inputs(args, kwargs, kwargs_ref, impl_ref):
    """Keep eager Top-K off NPU and copy bytes without an FP8 device cast."""
    kwargs = {**kwargs, **kwargs_ref}
    if impl_ref != "torch_native":
        return args, kwargs
    return (
        tuple(_cpu_tensor(value) for value in args),
        {key: _cpu_tensor(value) for key, value in kwargs.items()},
    )


def paired_inputs(data):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    from tests.python.utils.common import (
        get_skip_acc_check,
        get_test_impls,
        require_distinct_impls,
        require_test_impl,
    )

    args, common, table = data
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is True:
        require_test_impl(quant_lightning_indexer, impl)
    else:
        require_distinct_impls(quant_lightning_indexer, impl, impl_ref)
    metadata_args = (64, 1, 128, 512, args[-1])
    metadata = quant_lightning_indexer_metadata(*metadata_args, impl=impl, **common)
    kwargs = dict(common, block_table=table, metadata=metadata, return_value=0)
    kwargs_ref = None
    if get_skip_acc_check() is not True:
        ref_args, ref_common = reference_inputs(metadata_args, common, {}, impl_ref)
        kwargs_ref = dict(
            metadata=quant_lightning_indexer_metadata(
                *ref_args,
                impl=impl_ref,
                **ref_common,
            )
        )
    return args, kwargs, kwargs_ref, metadata_args, common


perf_shape = [CASES[1]]


def assert_indexer_accuracy(actual, expected):
    # Compare the selected set; accelerator kernels can return a different order.
    actual_indices, actual_values = actual
    expected_indices, expected_values = expected
    assert_accuracy(
        (actual_indices.cpu().sort(dim=-1).values, actual_values.cpu()),
        (expected_indices.cpu().sort(dim=-1).values, expected_values.cpu()),
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_quant_lightning_indexer(case):
    from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

    args, kwargs, kwargs_ref, _, _ = paired_inputs(generate_data(case, hardware_mode()))
    executor = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    if get_skip_acc_check() is not True:
        reference_args, reference_kwargs = reference_inputs(
            args, kwargs, kwargs_ref, executor.impl_ref
        )
        expected = quant_lightning_indexer(
            *reference_args, **reference_kwargs, impl=executor.impl_ref
        )
        assert_indexer_accuracy(executor(), expected)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=case in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize("case", CASES[:2], ids=CASE_IDS[:2])
def test_quant_lightning_indexer_queued_lifetimes(case):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    if get_skip_acc_check() is True:
        pytest.skip("Queued lifetime regression requires accuracy comparison")
    args, kwargs, kwargs_ref, metadata_args, common = paired_inputs(
        generate_data(case, hardware_mode())
    )
    executor = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    reference_args, reference_kwargs = reference_inputs(
        args, kwargs, kwargs_ref, executor.impl_ref
    )
    expected = quant_lightning_indexer(
        *reference_args, **reference_kwargs, impl=executor.impl_ref
    )
    results = []
    for _ in range(4):
        # Each iteration replaces the previous metadata Tensor. The queued
        # bridge must own its inputs and workspace until the ACL call runs.
        metadata = quant_lightning_indexer_metadata(
            *metadata_args, impl=executor.impl, **common
        )
        results.append(
            quant_lightning_indexer(
                *args, **{**kwargs, "metadata": metadata}, impl=executor.impl
            )
        )
        # Encourage allocator reuse without a host read or synchronization.
        torch.empty(1 << 20, dtype=torch.uint8, device=args[0].device).fill_(0xA5)
    for result in results:
        assert_indexer_accuracy(result, expected)
