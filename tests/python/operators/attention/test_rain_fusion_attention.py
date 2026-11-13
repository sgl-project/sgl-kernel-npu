"""RainFusionAttention accuracy through the public attention interface.

Only attention_out is compared: the device Kernel may leave softmax_lse
untouched, so its content stays undefined. The FP32 torch_native reference
casts the result back to the query dtype; FP16 uses a wider tolerance because
the Kernel defaults to FP16 softmax (innerPrecise=1).
"""

import pytest

torch = pytest.importorskip("torch")

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

# One representative TND call; only this tuple is profiled.
perf_shape = [((2048,), 2, 128, 128)]

RAIN_FUSION_ATTENTION_STANDARD = {
    "torch.float16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}


def generate_test_data(seqs, heads, dim, block, dtype, selection="dense"):
    """Build packed TND inputs; selections are ascending KV block ids."""
    seqs = list(seqs)
    total = sum(seqs)
    generator = torch.Generator(device="cpu").manual_seed(0)
    shape = (total, heads, dim)
    query = (
        torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        .to(dtype)
        .npu()
    )
    key = (
        torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        .to(dtype)
        .npu()
    )
    value = (
        torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        .to(dtype)
        .npu()
    )

    q_block_counts = [(seq + block - 1) // block for seq in seqs]
    max_kv_blocks = max((seq + block - 1) // block for seq in seqs)
    # Build the int64 control tensors on CPU: the fixture sets the default device
    # to NPU, and filling them through in-place index assignment on NPU can read
    # stale allocator memory on some torch_npu/CANN combinations.
    select_idx = torch.full(
        (sum(q_block_counts), heads, max_kv_blocks),
        -1,
        dtype=torch.int64,
        device="cpu",
    )
    select_num_idx = torch.zeros(
        (sum(q_block_counts), heads), dtype=torch.int64, device="cpu"
    )
    row = 0
    for count, seq in zip(q_block_counts, seqs):
        kv_blocks = (seq + block - 1) // block
        selected = (
            torch.arange(kv_blocks, dtype=torch.int64, device="cpu")
            if selection == "dense"
            else torch.arange(0, kv_blocks, 2, dtype=torch.int64, device="cpu")
        )
        for _ in range(count):
            select_idx[row, :, : selected.numel()] = selected
            select_num_idx[row, :] = selected.numel()
            row += 1
    return dict(
        query=query,
        key=key,
        value=value,
        select_idx=select_idx.npu(),
        select_num_idx=select_num_idx.npu(),
        blockshape=[block, block],
        actual_seq_qlen=seqs,
        actual_seq_kvlen=seqs,
        head_num=heads,
        scale=dim**-0.5,
        # Match the sglang call site: float softmax, required for BF16.
        inner_precise=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqs,heads,dim,block,selection",
    [
        pytest.param((256,), 1, 128, 128, "dense", id="single-head"),
        pytest.param((256,), 2, 128, 128, "sparse", id="sparse-selection"),
        # 200 is not a multiple of the block tile: both tails are partial.
        pytest.param((200,), 2, 128, 128, "dense", id="tail-seq"),
        # Packed batches: select rows follow the flattened [block, head] order.
        pytest.param((128, 72), 2, 128, 128, "dense", id="two-batches"),
        pytest.param((2048,), 2, 128, 128, "dense", id="perf-tnd-2k"),
    ],
)
def test_rain_fusion_attention(seqs, heads, dim, block, selection, dtype):
    from sgl_kernel_npu.operators.attention import rain_fusion_attention

    impl, impl_ref = get_test_impls()
    data = generate_test_data(seqs, heads, dim, block, dtype, selection)
    query, key, value = data["query"], data["key"], data["value"]
    query_before, key_before, value_before = query.clone(), key.clone(), value.clone()

    if get_skip_acc_check():
        require_test_impl(rain_fusion_attention, impl)
        attention_out, softmax_lse = rain_fusion_attention(**data, impl=impl)
        accuracy_checked = False
    else:
        require_distinct_impls(rain_fusion_attention, impl, impl_ref)
        result_ref = rain_fusion_attention(**data, impl=impl_ref)
        attention_out, softmax_lse = rain_fusion_attention(**data, impl=impl)
        assert attention_out.shape == result_ref[0].shape
        assert attention_out.dtype == result_ref[0].dtype
        assert attention_out.device == result_ref[0].device
        assert softmax_lse.shape == result_ref[1].shape
        assert softmax_lse.dtype == result_ref[1].dtype
        assert_accuracy(
            attention_out,
            result_ref[0],
            standard=RAIN_FUSION_ATTENTION_STANDARD,
            path="attention_out",
        )
        accuracy_checked = True

    assert attention_out.shape == query.shape
    assert attention_out.dtype == query.dtype
    assert attention_out.device == query.device
    # Legacy allocation: the first three query dimensions, query dtype/format.
    assert softmax_lse.shape == query.shape
    assert softmax_lse.dtype == query.dtype
    assert torch.equal(query, query_before)
    assert torch.equal(key, key_before)
    assert torch.equal(value, value_before)

    repeated, _ = rain_fusion_attention(**data, impl=impl)
    assert torch.equal(attention_out, repeated)

    executor = make_executor(
        rain_fusion_attention,
        kwargs=data,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    check_performance(
        executor,
        enable_perf=(tuple(seqs), heads, dim, block) in perf_shape
        and selection == "dense"
        and dtype == torch.float16,
    )
