"""A5 mixed-quant attention accuracy with paired KV encodings and metadata."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_skip_acc_check, get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [1]


@pytest.mark.parametrize(
    "query_len", [1, 2], ids=["synthetic-decode", "synthetic-extend"]
)
def test_mixed_attention(query_len):
    from sgl_kernel_npu.operators.attention import (
        mixed_quant_sparse_flash_mla,
        mixed_quant_sparse_flash_mla_metadata,
    )
    from sgl_kernel_npu.operators.kvcacheio import (
        get_npu_kv_quant_path,
        kv_compress_epilog,
    )
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("mixed-quant reader/writer requires A5")
    torch.manual_seed(13)
    x = torch.randn(128, 512, dtype=torch.bfloat16)
    q = torch.randn(query_len, 64, 512, dtype=torch.bfloat16) * 0.1
    slots = torch.arange(128, dtype=torch.int32)
    slots[-1] = -1
    common = dict(
        cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32),
        seqused_q=torch.tensor([query_len], dtype=torch.int32),
        seqused_ori_kv=torch.tensor([127], dtype=torch.int32),
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
    impl, impl_ref = get_test_impls()
    cache = torch.zeros(1, 128, 1, path.packed_dim(448, 64), dtype=torch.float8_e4m3fn)
    cache_ref = cache.clone()
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
    kwargs_ref = None
    if get_skip_acc_check() is not True:
        kv_compress_epilog(cache_ref, x, slots, impl=impl_ref, **writer_kwargs)
        kwargs_ref = dict(
            ori_kv=cache_ref,
            metadata=mixed_quant_sparse_flash_mla_metadata(
                impl=impl_ref, **metadata_kwargs
            ),
        )
    executor = check_accuracy(
        mixed_quant_sparse_flash_mla,
        args=(q,),
        kwargs=kwargs,
        kwargs_ref=kwargs_ref,
        standard={"torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03)},
    )
    check_performance(executor, enable_perf=query_len in perf_shape)
