"""mixed_quant_sparse_flash_mla accuracy and isolated operator timing.

Metadata correctness uses paired compute with an independent eager reference.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [1]


def _pack_kv(x, width):
    """Build reader inputs locally; no cache-writer operator participates."""
    values = x.float().cpu()
    groups = values[:, :448].reshape(-1, 7, 64)
    amax = groups.abs().amax(-1, keepdim=True)
    if width == 640:
        raw_scale = amax.clamp_min(1e-4) * torch.tensor(
            1.0 / 448.0, dtype=torch.float32, device="cpu"
        )
        _, exponent = torch.frexp(raw_scale)
        scales = torch.ldexp(torch.ones_like(raw_scale), exponent)
    elif width == 608:
        raw_scale = amax / 448.0
        raw_scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
        scales = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    else:
        raise ValueError("attention fixture supports 608B and 640B KV rows")
    payload = (groups / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    packed = torch.zeros(values.shape[0], width, dtype=torch.uint8, device="cpu")
    packed[:, :128] = values[:, 448:].to(torch.bfloat16).contiguous().view(torch.uint8)
    packed[:, 128:576] = payload.contiguous().view(torch.uint8).reshape(-1, 448)
    if width == 640:
        packed[:, 576:583] = (torch.log2(scales).reshape(-1, 7) + 127).to(torch.uint8)
    else:
        packed[:, 576:590] = (
            scales.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(-1, 14)
        )
    return (
        packed.to(x.device).reshape(1, x.shape[0], 1, width).view(torch.float8_e4m3fn)
    )


def generate_data(query_len):
    """DeepSeek-V4-Flash: Q [T,64,512], NoPE=448, RoPE=64, page=128."""
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("mixed-quant reader/writer requires A5")
    torch.manual_seed(13)
    x = torch.randn(128, 512, dtype=torch.bfloat16)
    q = torch.randn(query_len, 64, 512, dtype=torch.bfloat16) * 0.1
    common = dict(
        cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32),
        seqused_q=torch.tensor([query_len], dtype=torch.int32),
        seqused_ori_kv=torch.tensor([128], dtype=torch.int32),
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        quant_mode=1,
        rope_head_dim=64,
    )
    width = 640 if detect_cann_version().startswith("9.1.") else 608
    cache = _pack_kv(x, width)
    return q, common, cache


def check_mixed_quant_sparse_flash_mla_accuracy(data, *, target):
    from sgl_kernel_npu.operators.attention import (
        mixed_quant_sparse_flash_mla,
        mixed_quant_sparse_flash_mla_metadata,
    )

    q, common, cache = data
    impl, impl_ref = get_test_impls()
    skip_accuracy = get_skip_acc_check() is True
    if skip_accuracy:
        require_test_impl(mixed_quant_sparse_flash_mla, impl)
    else:
        require_distinct_impls(mixed_quant_sparse_flash_mla, impl, impl_ref)
    cache_ref = cache.view(torch.uint8).clone().view(torch.float8_e4m3fn)
    # Both readers consume identical locally packed bytes. Cache-writer
    # accuracy is covered by its own mem_cache tests.
    metadata_kwargs = dict(
        batch_size=1,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        has_ori_kv=True,
        has_cmp_kv=False,
        **common,
    )
    kwargs = dict(
        ori_kv=cache,
        ori_block_table=torch.zeros(1, 1, dtype=torch.int32),
        sinks=torch.zeros(64, dtype=torch.float32),
        metadata=mixed_quant_sparse_flash_mla_metadata(impl=impl, **metadata_kwargs),
        softmax_scale=512**-0.5,
        **common,
    )
    executor = make_executor(mixed_quant_sparse_flash_mla, args=(q,), kwargs=kwargs)
    if not skip_accuracy:
        reference_kwargs = {
            **kwargs,
            "ori_kv": cache_ref,
            "metadata": mixed_quant_sparse_flash_mla_metadata(
                impl=impl_ref, **metadata_kwargs
            ),
        }
        expected = mixed_quant_sparse_flash_mla(q, impl=impl_ref, **reference_kwargs)
        actual = executor()
        assert_accuracy(
            actual,
            expected,
            {"torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03)},
        )
        executor.accuracy_checked = True
    if target == "compute":
        return executor
    metadata = kwargs["metadata"]
    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    from sgl_kernel_npu.operators.attention import (
        get_mixed_quant_sparse_flash_mla_metadata_size,
    )

    assert metadata.shape == (get_mixed_quant_sparse_flash_mla_metadata_size(),)
    assert metadata.is_contiguous()
    # Opaque metadata is checked through paired compute, not byte comparison.
    return make_executor(
        mixed_quant_sparse_flash_mla_metadata,
        kwargs=metadata_kwargs,
        accuracy_checked=executor.accuracy_checked,
    )


@pytest.mark.parametrize(
    "query_len",
    [1, 128],
    ids=[
        "deepseek-v4-flash-decode-q1-n64-d512-page128",
        "deepseek-v4-flash-prefill-q128-n64-d512-page128",
    ],
)
def test_mixed_quant_sparse_flash_mla(query_len):
    data = generate_data(query_len)
    executor = check_mixed_quant_sparse_flash_mla_accuracy(data, target="compute")
    check_performance(executor, enable_perf=query_len in perf_shape)
