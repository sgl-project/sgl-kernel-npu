"""BF16 sparse-attention accuracy with paired metadata and NPU profiling."""

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


def run_sparse_flash_mla_case(batch, query_len, *, target):
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    torch.manual_seed(12)
    q = torch.randn(batch * query_len, 64, 512, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(batch, 128, 1, 512, dtype=torch.bfloat16)
    cu = torch.arange(batch + 1, dtype=torch.int32) * query_len
    lengths = torch.tensor([64, 128][:batch], dtype=torch.int32)
    common = dict(
        cu_seqlens_q=cu,
        seqused_q=cu[1:] - cu[:-1],
        seqused_ori_kv=lengths,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    impl, impl_ref = get_test_impls()
    skip_accuracy = get_skip_acc_check() is True
    if skip_accuracy:
        require_test_impl(sparse_flash_mla, impl)
    else:
        require_distinct_impls(sparse_flash_mla, impl, impl_ref)
    metadata_kwargs = dict(
        batch_size=batch,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        has_ori_kv=True,
        has_cmp_kv=False,
        **common,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
        sinks=torch.zeros(64, dtype=torch.float32),
        metadata=sparse_flash_mla_metadata(impl=impl, **metadata_kwargs),
        softmax_scale=512**-0.5,
        **common,
    )
    executor = make_executor(sparse_flash_mla, args=(q,), kwargs=kwargs)
    if not skip_accuracy:
        # Opaque CANN tiling and eager descriptors must be built independently.
        reference_kwargs = {
            **kwargs,
            "metadata": sparse_flash_mla_metadata(impl=impl_ref, **metadata_kwargs),
        }
        expected = sparse_flash_mla(q, impl=impl_ref, **reference_kwargs)
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
        sparse_flash_mla_metadata,
        kwargs=metadata_kwargs,
        accuracy_checked=executor.accuracy_checked,
    )
