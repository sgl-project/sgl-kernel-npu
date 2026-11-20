"""Validate paged KV cache writes against PyTorch through the public API."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")


def generate_paged_data(slot_ids, page_size, heads, width):
    torch.npu.manual_seed(42)
    key = torch.randn(len(slot_ids), heads, width, dtype=torch.float16)
    cache_shape = (2, page_size, heads, width)
    return dict(
        key=key,
        value=torch.randn_like(key),
        key_cache=torch.zeros(cache_shape, dtype=key.dtype),
        value_cache=torch.full(cache_shape, -1, dtype=key.dtype),
    )


@pytest.mark.parametrize(
    "name, slot_ids, page_size, heads, width",
    [
        pytest.param(
            "reshape_and_cache", (31, 15, 16), 16, 2, 128, id="gqa-kv-pool-reshape"
        ),
        pytest.param(
            "scatter_pa_kv_cache", (31, 15, 16), 16, 2, 128, id="gqa-kv-pool-scatter"
        ),
    ],
)
def test_paged_cache_write(name, slot_ids, page_size, heads, width):
    from sgl_kernel_npu.operators import mem_cache

    kwargs = generate_paged_data(slot_ids, page_size, heads, width)
    slots = torch.tensor(slot_ids, dtype=torch.int32)
    if name == "scatter_pa_kv_cache":
        kwargs.update(slot_mapping=slots, cache_mode="Norm")
    else:
        kwargs["slot_indices"] = slots
    executor = check_accuracy(
        getattr(mem_cache, name),
        kwargs=kwargs,
        mutable_kwargs=("key_cache", "value_cache"),
        standard={"torch.float16": AccuracyStandard(atol=0, rtol=0)},
    )
    check_performance(executor)


def generate_rmsnorm_cache_data(tokens, page_size):
    torch.npu.manual_seed(42)
    dtype = torch.float16
    latent = torch.randn(tokens, 1, 1, 512, dtype=dtype)
    rope = torch.full((tokens, 1, 1, 64), 0.75, dtype=dtype)
    cos = torch.ones(tokens, 1, 1, 64, dtype=dtype)
    args = (
        torch.cat((latent, rope), dim=-1),
        torch.rand(512, dtype=dtype),
        cos,
        torch.zeros_like(cos),
        torch.tensor([page_size + 3, 0], dtype=torch.int64),
    )
    return args, dict(
        k_cache=torch.zeros(2, page_size, 1, 64, dtype=dtype),
        ckv_cache=torch.zeros(2, page_size, 1, 512, dtype=dtype),
        cache_mode="PA_BNSD",
        is_output_kv=True,
    )


@pytest.mark.parametrize(
    "tokens, page_size", [pytest.param(2, 16, id="deepseek-mla-decode")]
)
def test_kv_rmsnorm_rope_cache_paged(tokens, page_size):
    from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache

    args, kwargs = generate_rmsnorm_cache_data(tokens, page_size)
    executor = check_accuracy(
        kv_rmsnorm_rope_cache,
        args,
        kwargs=kwargs,
        mutable_kwargs=("k_cache", "ckv_cache"),
        standard={"torch.float16": AccuracyStandard(atol=0.003, rtol=0.004)},
    )
    check_performance(executor)
