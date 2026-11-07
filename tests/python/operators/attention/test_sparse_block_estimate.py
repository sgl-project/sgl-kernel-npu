"""Sparse estimation accuracy/performance through the public operator API."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 4, 4, 4096, 64)]


def generate_test_data(shape, dtype, layout, mode, seed=42):
    """Full written rows for check_accuracy; partial lengths have a separate test."""
    batch, heads, kv_heads, seq, dim = shape
    torch.npu.manual_seed(seed)
    q = torch.randn((batch, heads, seq, dim), dtype=dtype) * 0.1
    k = torch.randn((batch, kv_heads, seq, dim), dtype=dtype) * 0.1
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
    kwargs = dict(input_layout=layout, num_heads=heads, num_key_value_heads=kv_heads)
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
        (1, 1, 1, 256, 64),  # Causal <=2 blocks shortcut.
        (2, 4, 2, 2048, 64),  # GQA and row_sparse's >=10 block branch.
        (1, 4, 4, 4096, 64),
        (1, 1, 1, 8200, 64),  # Sparse tail, >32 columns, multiple inner tiles.
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mode",
    [
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
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("threshold", [1.0, 0.8])
def test_sparse_block_estimate_modes(dtype, mode, causal, threshold):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    q, k, kwargs = generate_test_data((2, 4, 2, 2048, 64), dtype, "BNSD", mode)
    kwargs.update(causal=causal, threshold=threshold)
    executor = check_accuracy(sparse_block_estimate, args=(q, k), kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("keep_sink", [True, False])
@pytest.mark.parametrize("keep_recent", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.99999, 1.0])
def test_sparse_block_estimate_selection(keep_sink, keep_recent, causal, threshold):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    q, k, kwargs = generate_test_data(
        (1, 1, 1, 2048, 64), torch.float16, "BNSD", "normal"
    )
    kwargs.update(
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        causal=causal,
        threshold=threshold,
        row_sparse=0.4,
    )
    executor = check_accuracy(sparse_block_estimate, args=(q, k), kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("stride,sparse_size", [(4, 64), (8, 128), (16, 256)])
def test_sparse_block_estimate_block_sizes(stride, sparse_size):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    q, k, kwargs = generate_test_data(
        (1, 1, 1, 2048, 64), torch.float16, "BNSD", "normal"
    )
    kwargs.update(stride=stride, sparse_size=sparse_size, threshold=0.8)
    executor = check_accuracy(sparse_block_estimate, args=(q, k), kwargs=kwargs)
    check_performance(executor, enable_perf=False)


def test_sparse_block_estimate_defaults_and_direct_entry():
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    q, k, _ = generate_test_data((1, 1, 1, 256, 64), torch.float16, "BNSD", "normal")
    # Actually omit all optional parameters rather than populating default kwargs.
    executor = check_accuracy(sparse_block_estimate, args=(q, k))
    check_performance(executor, enable_perf=False)
    result = sparse_block_estimate(q, k)
    assert_accuracy(result, sparse_block_estimate(q, k, impl="ascendc"))
    assert_accuracy(result, torch.ops.attentions.sparse_block_estimate(q, k))
    explicit = sparse_block_estimate(
        q, k, None, None, "BNSD", 8, 128, 1, 1, 1, 1, True, True, True, 1
    )
    assert_accuracy(result, explicit)


@pytest.mark.parametrize(
    "actual_q,actual_k", [([1024], None), (None, [1024]), ([1024, 1536], [1024, 1536])]
)
def test_sparse_block_estimate_actual_lengths(actual_q, actual_k):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    from tests.python.utils.common import (
        get_skip_acc_check,
        get_test_impls,
        require_distinct_impls,
        require_test_impl,
    )

    q, k, kwargs = generate_test_data(
        (2, 1, 1, 2048, 64), torch.float16, "BNSD", "normal"
    )
    kwargs.update(
        actual_seq_lengths=actual_q, actual_seq_lengths_kv=actual_k, causal=False
    )
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check():
        require_test_impl(sparse_block_estimate, impl)
        return
    require_distinct_impls(sparse_block_estimate, impl, impl_ref)
    result_ref = sparse_block_estimate(q, k, impl=impl_ref, **kwargs)
    result = sparse_block_estimate(q, k, impl=impl, **kwargs)
    assert result[0].shape == (2, 1, 16, 32)
    assert result[1].shape == (2, 1, 16)
    # Rows outside actual Q lengths were allocated with empty and never written.
    for b in range(2):
        length = 2048 if actual_q is None else actual_q[b if len(actual_q) > 1 else 0]
        rows = (length + 127) // 128
        assert_accuracy(
            tuple(t[b, :, :rows] for t in result),
            tuple(t[b, :, :rows] for t in result_ref),
        )


@pytest.mark.parametrize(
    "name,message",
    [
        ("num_heads", "num_heads must be nonzero"),
        ("sparse_size", "sparse_size must be nonzero"),
    ],
)
@pytest.mark.parametrize("impl", ["ascendc", "torch_native"])
def test_sparse_block_estimate_explicit_checks(name, message, impl):
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    from tests.python.utils.common import require_test_impl

    require_test_impl(sparse_block_estimate, impl)
    q, k, _ = generate_test_data((1, 1, 1, 256, 64), torch.float16, "BNSD", "normal")
    with pytest.raises(RuntimeError, match=message):
        sparse_block_estimate(q, k, **{name: 0}, impl=impl)


def test_sparse_block_estimate_legacy_descriptor_dtype_check():
    from sgl_kernel_npu.operators.attention import sparse_block_estimate

    from tests.python.utils.common import require_test_impl

    require_test_impl(sparse_block_estimate, "ascendc")
    q, _, _ = generate_test_data((1, 1, 1, 256, 64), torch.float16, "BNSD", "normal")
    # A CPU key still selects PrivateUse1 because query is on NPU. The legacy
    # ConvertType rejects this dtype before the ACLNN workspace query.
    key = torch.empty(q.shape, dtype=torch.complex32, device="cpu")
    with pytest.raises(RuntimeError, match="ComplexHalf has not been supported"):
        sparse_block_estimate(q, key, impl="ascendc")
