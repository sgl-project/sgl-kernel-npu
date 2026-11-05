"""Deterministic paged inputs with strictly ordered, non-tied Top-K scores."""

import pytest
import torch

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


def paired_inputs(data):
    from sgl_kernel_npu.operators.attention import (
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
        kwargs_ref = dict(
            metadata=quant_lightning_indexer_metadata(
                *metadata_args, impl=impl_ref, **common
            )
        )
    return args, kwargs, kwargs_ref, metadata_args, common
