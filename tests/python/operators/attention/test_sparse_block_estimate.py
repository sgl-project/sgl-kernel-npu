import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 4, 4, 4096, 64)]  # The only shape measured for performance.


def generate_test_data(
    shape, dtype, layout, mode, seed=42, *, stride=8, sparse_size=128
):
    """Separate block scores; partial lengths have a separate test.

    A positive Q signal and shuffled, block-wise K offsets separate logits.
    Bound their range to avoid an almost one-hot softmax on long sequences.
    Small random perturbations retain variation across tokens, heads and
    batches. Ties and zero scale remain explicit cases.
    """
    batch, heads, kv_heads, seq, dim = shape
    torch.npu.manual_seed(seed)
    q = torch.randn((batch, heads, seq, dim), dtype=torch.float32) / 64 + 0.25
    k = torch.randn((batch, kv_heads, seq, dim), dtype=torch.float32) / 256
    span = seq * (kv_heads if layout != "BNSD" else 1)
    blocks = (span + sparse_size - 1) // sparse_size
    levels = torch.randperm(blocks, device=k.device).to(torch.float32)
    gap = min(0.5, 8.0 / max(blocks - 1, 1))
    levels = (levels - (blocks - 1) / 2) * gap
    scale = dim**-0.5 if mode == "scaled" else 1.0
    offsets = levels / (stride * dim * 0.25 * scale)
    if layout == "BNSD":
        block_ids = torch.arange(seq, device=k.device) // sparse_size
        k += offsets[block_ids].view(1, 1, seq, 1)
    else:
        # The legacy BSND/BSH Matmul consumes raw contiguous K spans across
        # heads. Give each such span its own offset rather than repeating
        # one offset across multiple spans of the same logical token block.
        block_ids = torch.arange(span, device=k.device).view(seq, kv_heads).T
        k += offsets[block_ids // sparse_size].view(1, kv_heads, seq, 1)
    q, k = q.to(dtype), k.to(dtype)
    if layout != "BNSD":
        q, k = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
        if layout == "BSH":
            q, k = q.flatten(2), k.flatten(2)
    if mode == "offset":
        # A contiguous view whose storage starts before its data pointer.
        def with_offset(t):
            storage = torch.empty(t.numel() + 16, dtype=t.dtype)
            view = storage[16:].view_as(t)
            view.copy_(t)
            return view

        q, k = with_offset(q), with_offset(k)
    if mode == "ties":
        q.zero_()
        k.zero_()
    kwargs = dict(
        input_layout=layout,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        stride=stride,
        sparse_size=sparse_size,
    )
    if mode == "none":
        kwargs.update(actual_seq_lengths=None, actual_seq_lengths_kv=None)
    elif mode == "empty":
        kwargs.update(actual_seq_lengths=[], actual_seq_lengths_kv=[])
    elif mode == "broadcast":
        kwargs.update(actual_seq_lengths=[seq], actual_seq_lengths_kv=[seq])
    elif mode == "per_batch":
        kwargs.update(
            actual_seq_lengths=[seq] * batch, actual_seq_lengths_kv=[seq] * batch
        )
    elif mode == "zero_scale":
        kwargs["scale_value"] = 0.0
    elif mode == "scaled":
        kwargs["scale_value"] = dim**-0.5
    elif mode == "row_sparse":
        kwargs["row_sparse"] = 0.4
    return q, k, kwargs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "BSH"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 64), id="edge-causal-shortcut"),
        pytest.param((2, 4, 2, 2048, 64), id="gqa-row-sparse-branch"),
        # Same tuple as perf_shape; the only case that is timed.
        pytest.param((1, 4, 4, 4096, 64), id="perf-4k-mha"),
        pytest.param((1, 1, 1, 8200, 64), id="edge-sparse-tail"),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
def test_sparse_block_estimate(shape, dtype, layout, causal):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    q, k, kwargs = generate_test_data(shape, dtype, layout, "normal")
    kwargs["causal"] = causal
    before_q, before_k = q.clone(), k.clone()
    executor = check_accuracy(sparse_block_estimate, args=(q, k), kwargs=kwargs)
    if executor.accuracy_checked:
        assert torch.equal(q, before_q)
        assert torch.equal(k, before_k)
    check_performance(
        executor,
        enable_perf=shape in perf_shape
        and dtype == torch.float16
        and layout == "BNSD"
        and causal,
    )


# Extended parameter combinations are excluded by the repository's default
# marker expression; the accuracy and performance regression cases above stay.
@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "causal,mode",
    [
        pytest.param(causal, mode, id=f"{causal}-{mode}")
        for causal in (True, False)
        for mode in (
            "normal",
            "none",
            "empty",
            "broadcast",
            "per_batch",
            "zero_scale",
            "scaled",
            "row_sparse",
            "offset",
            "ties",
        )
        if causal or mode != "ties"
    ],
)
@pytest.mark.parametrize("threshold", [1.0])
def test_sparse_block_estimate_modes(dtype, mode, causal, threshold):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    q, k, kwargs = generate_test_data((2, 4, 2, 2048, 64), dtype, "BNSD", mode)
    kwargs.update(causal=causal, threshold=threshold)
    executor = check_accuracy(sparse_block_estimate, args=(q, k), kwargs=kwargs)
    check_performance(executor, enable_perf=False)
