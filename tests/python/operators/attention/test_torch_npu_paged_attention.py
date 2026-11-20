"""Validate the legacy ATB attention APIs through their registered references."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_PAGE_SIZE = 128


def generate_data(
    q_heads, kv_heads, head_dim, context_lengths, query_lengths, mla_vheadsize
):
    generator = torch.Generator(device="cpu").manual_seed(42)
    prefill = query_lengths is not None
    dtype = torch.bfloat16 if prefill else torch.float16
    q_tokens = sum(query_lengths) if prefill else len(context_lengths)
    pages_per_request = (max(context_lengths) + _PAGE_SIZE - 1) // _PAGE_SIZE
    pages = len(context_lengths) * pages_per_request
    query = (
        torch.randn((q_tokens, q_heads, head_dim), generator=generator, device="cpu")
        .to(dtype)
        .npu()
    )
    key = (
        torch.randn(
            (pages, _PAGE_SIZE, kv_heads, head_dim), generator=generator, device="cpu"
        )
        .to(dtype)
        .npu()
    )
    page_rows = (
        torch.arange(pages, dtype=torch.int32, device="cpu")
        .reshape(-1, pages_per_request)
        .flip(-1)
    )
    kwargs = dict(
        query=query,
        key_cache=key,
        block_table=page_rows.to(query.device) if prefill else page_rows,
        context_lens=torch.tensor(context_lengths, dtype=torch.int32, device="cpu"),
        scale_value=(head_dim if mla_vheadsize is None else 192) ** -0.5,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
    )
    if mla_vheadsize is None:
        kwargs["value_cache"] = (
            torch.randn(key.shape, generator=generator, device="cpu").to(dtype).npu()
        )
    else:
        kwargs["mla_vheadsize"] = mla_vheadsize
    if prefill:
        kwargs["query"] = query.flatten(1)
        kwargs["mask"] = (
            torch.ones((128, 128), dtype=dtype, device="cpu").triu_(1).to(query.device)
        )
        kwargs["seq_len"] = torch.tensor(query_lengths, dtype=torch.int32, device="cpu")
    out_shape = (
        (*query.shape[:-1], mla_vheadsize) if mla_vheadsize else kwargs["query"].shape
    )
    kwargs["out"] = torch.full(
        out_shape, float("nan"), dtype=query.dtype, device=query.device
    )
    return kwargs


@pytest.mark.parametrize(
    "name, q_heads, kv_heads, head_dim, context_lengths, query_lengths, mla_vheadsize",
    [
        pytest.param(
            "flash_attention_qlens",
            16,
            4,
            128,
            (128, 1024),
            (128, 256),
            None,
            id="qwen3-8b-prefix-cache-prefill",
        ),
        pytest.param(
            "paged_attention",
            16,
            4,
            128,
            (255, 129),
            None,
            None,
            id="qwen3-8b-paged-decode",
        ),
        pytest.param(
            "paged_attention_mla",
            16,
            1,
            576,
            (255,),
            None,
            512,
            id="deepseek-v3-paged-mla-decode",
        ),
    ],
)
def test_legacy_atb_attention(
    name, q_heads, kv_heads, head_dim, context_lengths, query_lengths, mla_vheadsize
):
    import torch_npu
    from sgl_kernel_npu.operators import attention

    operator = getattr(attention, name)
    require_test_impl(operator, get_test_impls()[0])
    if not callable(getattr(torch_npu, f"_npu_{name}", None)):
        pytest.skip(f"Installed TorchNPU lacks _npu_{name}")
    kwargs = generate_data(
        q_heads, kv_heads, head_dim, context_lengths, query_lengths, mla_vheadsize
    )
    standard = (
        AccuracyStandard(atol=5e-3, rtol=6e-3)
        if mla_vheadsize
        else AccuracyStandard(atol=4e-3, rtol=5e-3)
    )
    executor = check_accuracy(
        operator,
        kwargs=kwargs,
        mutable_kwargs=("out",),
        standard={
            "torch.float16": standard,
            "torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=2e-2),
        },
    )
    check_performance(executor)
