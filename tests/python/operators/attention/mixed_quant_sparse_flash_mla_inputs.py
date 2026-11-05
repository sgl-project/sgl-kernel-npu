"""A5 mixed-quant attention accuracy with paired KV encodings and metadata."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.standards import AccuracyStandard


def generate_data(query_len):
    """DeepSeek-V4-Flash: Q [T,64,512], NoPE=448, RoPE=64, page=128."""
    from sgl_kernel_npu.operators.mem_cache import get_npu_kv_quant_path
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("mixed-quant reader/writer requires A5")
    torch.manual_seed(13)
    x = torch.randn(128, 512, dtype=torch.bfloat16)
    q = torch.randn(query_len, 64, 512, dtype=torch.bfloat16) * 0.1
    slots = torch.arange(128, dtype=torch.int32)
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
    path = get_npu_kv_quant_path()
    cache = torch.zeros(1, 128, 1, path.packed_dim(448, 64), dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    return x, q, slots, common, path, cache


def check_mixed_quant_sparse_flash_mla_accuracy(data, *, target):
    from sgl_kernel_npu.operators.attention import (
        mixed_quant_sparse_flash_mla,
        mixed_quant_sparse_flash_mla_metadata,
    )
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

    x, q, slots, common, path, cache = data
    impl, impl_ref = get_test_impls()
    skip_accuracy = get_skip_acc_check() is True
    if skip_accuracy:
        require_test_impl(mixed_quant_sparse_flash_mla, impl)
    else:
        require_distinct_impls(mixed_quant_sparse_flash_mla, impl, impl_ref)
    cache_ref = cache.view(torch.uint8).clone().view(torch.float8_e4m3fn)
    writer_kwargs = dict(
        quant_group_size=64,
        quant_mode=path.writer_quant_mode,
        round_scale=True,
        x_scale=1.0,
    )
    # Setup is outside the executor. The writer has its own byte-accuracy test.
    kv_compress_epilog(cache, x, slots, impl=impl, **writer_kwargs)
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
        kv_compress_epilog(cache_ref, x, slots, impl=impl_ref, **writer_kwargs)
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
    assert metadata.ndim == 1 and metadata.numel() > 0
    assert metadata.is_contiguous()
    # Opaque metadata is checked through paired compute, not byte comparison.
    return make_executor(
        mixed_quant_sparse_flash_mla_metadata,
        kwargs=metadata_kwargs,
        accuracy_checked=executor.accuracy_checked,
    )
