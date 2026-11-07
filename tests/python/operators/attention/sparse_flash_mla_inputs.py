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


def generate_data(batch, query_len):
    """DeepSeek-V4-Flash Q [T,64,512], shared KV and 128-token pages."""
    torch.manual_seed(12)
    q = torch.randn(batch * query_len, 64, 512, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(batch * 2, 128, 1, 512, dtype=torch.bfloat16)
    cu = torch.arange(batch + 1, dtype=torch.int32) * query_len
    lengths = torch.tensor(([256, 128] * batch)[:batch], dtype=torch.int32)
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
    return q, kv, common, batch


def check_sparse_flash_mla_accuracy(data, *, target):
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    q, kv, common, batch = data
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
        ori_block_table=torch.arange(batch * 2, dtype=torch.int32).reshape(batch, 2),
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
    from sgl_kernel_npu.operators.attention import get_sparse_flash_mla_metadata_size

    assert metadata.shape == (get_sparse_flash_mla_metadata_size(),)
    assert metadata.is_contiguous()
    # Opaque metadata is checked through paired compute, not byte comparison.
    return make_executor(
        sparse_flash_mla_metadata,
        kwargs=metadata_kwargs,
        accuracy_checked=executor.accuracy_checked,
    )


def generate_dspark_data(batch, query_len, kv_lengths=None):
    """Explicit logical sparse rows for the DSpark multi-token draft ABI."""
    q, kv, common, _ = generate_data(batch, query_len)
    lengths = common["seqused_ori_kv"]
    if kv_lengths is not None:
        lengths.copy_(torch.tensor(kv_lengths, dtype=torch.int32, device=q.device))
    lengths_cpu = lengths.cpu()
    width = ((128 + query_len + 127) // 128) * 128
    indices_cpu = torch.full(
        (batch * query_len, 1, width), -1, dtype=torch.int32, device="cpu"
    )
    counts_cpu = torch.zeros(batch * query_len, 1, dtype=torch.int32, device="cpu")
    for b, kv_len in enumerate(lengths_cpu.tolist()):
        for offset in range(query_len):
            end = kv_len - query_len + offset + 1
            start = max(0, end - 128)
            row = b * query_len + offset
            count = end - start
            indices_cpu[row, 0, :count] = torch.arange(
                start, end, dtype=torch.int32, device="cpu"
            )
            counts_cpu[row, 0] = count
    common.update(
        ori_mask_mode=0,
        ori_win_left=127 + query_len,
        ori_topk_length=counts_cpu.to(q.device),
        cu_seqlens_q_cpu=common["cu_seqlens_q"].cpu(),
        seqused_ori_kv_cpu=lengths_cpu,
    )
    metadata_kwargs = dict(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        batch_size=batch,
        has_ori_kv=True,
        has_cmp_kv=False,
        ori_topk=width,
        **common,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.arange(batch * 2, dtype=torch.int32).reshape(batch, 2),
        ori_sparse_indices=indices_cpu.to(q.device),
        sinks=torch.zeros(64, dtype=torch.float32),
        softmax_scale=512**-0.5,
        **common,
    )
    return q, metadata_kwargs, kwargs
