"""Validate the paged KV writes used by Qwen3 GQA and DeepSeek MLA."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("reshape_and_cache", id="qwen3-8b-tp4-decode-legacy-writer"),
        pytest.param("scatter_pa_kv_cache", id="qwen3-8b-tp4-decode-scatter-pa"),
    ],
)
def test_paged_cache_write(name):
    from sgl_kernel_npu.operators import mem_cache

    torch.npu.manual_seed(42)
    # Three requests, eight KV heads sharded over four ranks, 128 slots per page.
    slots = torch.tensor([261, 255, 384], dtype=torch.int32)
    key = torch.randn(3, 2, 128, dtype=torch.bfloat16)
    cache_shape = (4, 128, 2, 128)
    kwargs = dict(
        key=key,
        value=torch.randn_like(key),
        key_cache=torch.zeros(cache_shape, dtype=key.dtype),
        value_cache=torch.full(cache_shape, -1, dtype=key.dtype),
    )
    if name == "scatter_pa_kv_cache":
        kwargs.update(slot_mapping=slots, cache_mode="Norm")
    else:
        kwargs["slot_indices"] = slots
    executor = check_accuracy(
        getattr(mem_cache, name),
        kwargs=kwargs,
        mutable_kwargs=("key_cache", "value_cache"),
        standard={"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)},
    )
    check_performance(executor)


@pytest.mark.parametrize(
    "positions", [pytest.param((127, 257), id="deepseek-v2-mla-decode-pa-bnsd")]
)
def test_kv_rmsnorm_rope_cache_paged(positions):
    from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache

    torch.npu.manual_seed(42)
    dtype = torch.bfloat16
    # DeepSeek-V2 YaRN uses theta=10000, factor=40, original context=4096
    # and beta_fast/slow=32/1. The resulting correction interval is [10, 23].
    lane = torch.arange(32, dtype=torch.float32)
    inv_freq = 1.0 / 10000 ** (lane / 32)
    ramp = ((lane - 10) / 13).clamp(0, 1)
    inv_freq = inv_freq * ((1 - ramp) + ramp / 40)
    phase = torch.outer(torch.tensor(positions, dtype=torch.float32), inv_freq)
    phase = torch.cat((phase, phase), dim=-1).view(-1, 1, 1, 64)
    args = (
        torch.randn(len(positions), 1, 1, 512 + 64, dtype=dtype),
        torch.rand(512, dtype=dtype),
        phase.cos().to(dtype),
        phase.sin().to(dtype),
        torch.tensor([259, 129], dtype=torch.int64),
    )
    executor = check_accuracy(
        kv_rmsnorm_rope_cache,
        args,
        kwargs=dict(
            k_cache=torch.zeros(3, 128, 1, 64, dtype=dtype),
            ckv_cache=torch.zeros(3, 128, 1, 512, dtype=dtype),
            epsilon=1e-6,
            cache_mode="PA_BNSD",
            is_output_kv=True,
        ),
        mutable_kwargs=("k_cache", "ckv_cache"),
        standard={"torch.bfloat16": AccuracyStandard(atol=0.015625, rtol=0.015625)},
    )
    check_performance(executor)
