"""SGLang Qwen3 GQA and DeepSeek-V2 MLA cases for legacy ATB attention."""

import math

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
    dtype = torch.bfloat16
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
            (pages + 1, _PAGE_SIZE, kv_heads, head_dim),
            generator=generator,
            device="cpu",
        )
        .to(dtype)
        .npu()
    )
    page_rows = (
        torch.arange(1, pages + 1, dtype=torch.int32, device="cpu")
        .reshape(-1, pages_per_request)
        .flip(-1)
    )
    kwargs = dict(
        query=query,
        key_cache=key,
        block_table=page_rows.to(query.device),
        context_lens=torch.tensor(context_lengths, dtype=torch.int32, device="cpu"),
        scale_value=head_dim**-0.5,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
    )
    if mla_vheadsize is None:
        kwargs["value_cache"] = (
            torch.randn(key.shape, generator=generator, device="cpu").to(dtype).npu()
        )
    else:
        kwargs["mla_vheadsize"] = mla_vheadsize
        # DeepSeek-V2 computes scale from the pre-absorption head width and YaRN.
        kwargs["scale_value"] = 192**-0.5 * (1 + 0.1 * 0.707 * math.log(40)) ** 2
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
            id="qwen3-8b-tp2-prefix-cache-prefill",
        ),
        pytest.param(
            "paged_attention",
            16,
            4,
            128,
            (255, 129),
            None,
            None,
            id="qwen3-8b-tp2-paged-decode",
        ),
        pytest.param(
            "paged_attention",
            16,
            4,
            128,
            (1, 128, 257),
            None,
            None,
            id="qwen3-8b-tp2-mixed-page-boundaries",
        ),
        pytest.param(
            "paged_attention_mla",
            16,
            1,
            576,
            (255,),
            None,
            512,
            id="deepseek-v2-tp8-paged-mla-decode",
        ),
        pytest.param(
            "paged_attention_mla",
            4,
            1,
            576,
            (255, 129),
            None,
            512,
            id="deepseek-v2-tp32-4q1kv-paged-mla-decode",
        ),
    ],
)
def test_legacy_atb_attention(
    name, q_heads, kv_heads, head_dim, context_lengths, query_lengths, mla_vheadsize
):
    import torch_npu
    from sgl_kernel_npu.operators import attention
    from sgl_kernel_npu.operators.utils.registry import get_impl

    operator = getattr(attention, name)
    if name == "flash_attention_qlens":
        require_test_impl(operator, get_test_impls()[0])
        if not callable(getattr(torch_npu, f"_npu_{name}", None)):
            pytest.skip(f"Installed TorchNPU lacks _npu_{name}")
    else:
        assert get_impl(operator, "auto") is get_impl(
            operator, "ascend-transformer-boost"
        )
    kwargs = generate_data(
        q_heads, kv_heads, head_dim, context_lengths, query_lengths, mla_vheadsize
    )
    executor = check_accuracy(
        operator,
        kwargs=kwargs,
        mutable_kwargs=("out",),
        standard={"torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=2e-2)},
    )
    check_performance(executor)


@pytest.mark.parametrize("blocked_value", [1.0, -torch.inf], ids=["binary", "additive"])
def test_torch_native_fp16_short_prefill_mask(blocked_value):
    from sgl_kernel_npu.operators.attention import flash_attention_qlens

    positions = torch.arange(_PAGE_SIZE, dtype=torch.float16, device="cpu")
    key = positions.new_zeros((1, _PAGE_SIZE, 1, 16))
    values = positions.view(1, -1, 1, 1).expand_as(key)
    lengths = torch.tensor([_PAGE_SIZE], dtype=torch.int32, device="cpu")
    out = torch.full_like(key[0], torch.nan)
    flash_attention_qlens(
        query=key[0],
        key_cache=key,
        value_cache=values,
        mask=positions.new_full((_PAGE_SIZE, _PAGE_SIZE), blocked_value).triu_(1),
        block_table=lengths.new_zeros((1, 1)),
        seq_len=lengths,
        context_lens=lengths,
        scale_value=16**-0.5,
        num_heads=1,
        num_kv_heads=1,
        out=out,
        impl="torch_native",
    )
    # Zero logits give row i the mean of the visible values 0 through i.
    expected = (positions / 2).view(-1, 1, 1).expand_as(out)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
