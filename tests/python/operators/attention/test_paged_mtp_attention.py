# SPDX-License-Identifier: Apache-2.0
"""Public paged DSpark GQA accuracy, FIA parity and mutable graph regressions."""

import math
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# (gamma, GQA group, head dimension, page size, KV lengths, KV heads).
# Kimi-K3's DSparkDraftModel has 64 Q heads, 16 KV heads and head_dim=64.
# TP4 gives rank-local Q=[B*7,16,64], 4 KV heads, draft block size 7 and
# page size 128. Batch sizes and KV lengths below sweep short/long contexts.
perf_shape = [
    (7, 4, 64, 128, (129,), 4),
    (7, 4, 64, 128, (2049,), 4),
    (7, 4, 64, 128, (4096,) * 4, 4),
    (7, 4, 64, 128, (1024,) * 16, 4),
    (7, 4, 64, 128, (16007,) * 16, 4),
]

edge_shape = [
    pytest.param((1, 1, 16, 16, (0, 1, 17), 2), id="minimum-and-page-tail"),
    pytest.param((7, 3, 48, 16, (7, 15, 17), 2), id="irregular-group-and-head"),
    pytest.param((7, 4, 64, 128, (0, 127, 134), 2), id="empty-and-page-tail"),
    pytest.param((8, 2, 32, 32, (31, 32, 33), 2), id="page-boundaries"),
    pytest.param((7, 8, 128, 128, (0, 128, 262), 2), id="large-group"),
    pytest.param((1, 1, 64, 128, (1, 129, 1025), 2), id="single-query-mha"),
    pytest.param((7, 1, 128, 32, (0, 7, 65), 2), id="multi-query-mha"),
    pytest.param((8, 4, 256, 128, (8, 127, 256), 2), id="wide-head-memory-bound"),
    pytest.param((7, 4, 64, 128, (2049,), 2), id="split-kv"),
    pytest.param((7, 4, 64, 128, (1, 3, 7), 2), id="fully-masked-causal-rows"),
    pytest.param((7, 4, 64, 128, (0, 0), 2), id="all-padding"),
    pytest.param((1, 1, 16, 16, (), 2), id="empty-batch"),
]

standard = {
    "torch.float16": AccuracyStandard(atol=0.004, rtol=0.03),
    "torch.bfloat16": AccuracyStandard(atol=0.004, rtol=0.03),
}


def generate_data(shape, dtype, mode="random", page_dtype=torch.int32):
    """Seed CPU data explicitly; conftest supplies the destination NPU device.

    Page order is shuffled. Unused cache tails contain NaNs and empty request
    rows contain -1 page indices, exposing accidental reads of padding.
    """
    gamma, group, dim, page, lengths, hkv = shape
    generator = torch.Generator(device="cpu").manual_seed(26)
    device = torch.get_default_device()
    batch = len(lengths)
    pages = max(1, math.ceil(max(lengths, default=0) / page))
    table = torch.randperm(batch * pages, generator=generator, device="cpu")
    table = table.reshape(batch, pages).to(page_dtype)
    query = (
        torch.randn(batch * gamma, hkv * group, dim, generator=generator, device="cpu")
        * 0.3
    ).to(dtype)
    keys = torch.full(
        (batch * pages, page, hkv * dim), float("nan"), dtype=dtype, device="cpu"
    )
    values = keys.clone()
    for b, length in enumerate(lengths):
        if length == 0:
            table[b].fill_(-1)
            continue
        positions = torch.arange(length, device="cpu")
        physical = table[b, positions // page].long()
        keys[physical, positions % page] = (
            torch.randn(length, hkv * dim, generator=generator, device="cpu") * 0.3
        ).to(dtype)
        values[physical, positions % page] = (
            torch.randn(length, hkv * dim, generator=generator, device="cpu") * 0.3
        ).to(dtype)
    if mode == "zero_query":
        query.zero_()
    elif mode not in ("random", "strided_query", "strided_cache", "strided_table"):
        raise ValueError(f"Unknown data mode: {mode}")
    args = dict(
        query=query.to(device),
        k_cache=keys.to(device),
        v_cache=values.to(device),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        cu_q_lens=torch.arange(
            0, (batch + 1) * gamma, gamma, dtype=torch.int32, device=device
        ),
        block_tables=table.to(device),
        page_size=page,
        num_q_heads=hkv * group,
        num_kv_heads=hkv,
        head_dim=dim,
        gamma=gamma,
        softmax_scale=dim**-0.5,
        max_kv_len=pages * page,
    )
    if mode == "strided_query":
        backing = torch.empty(
            (batch * gamma, hkv * group, dim * 2), dtype=dtype, device=device
        )
        args["query"] = backing[..., ::2].copy_(args["query"])
    elif mode == "strided_cache":
        # Both outer strides differ from contiguous ND storage, while the
        # innermost packed head dimension remains contiguous.
        for name in ("k_cache", "v_cache"):
            backing = torch.empty(
                (batch * pages * 2, page * 2, hkv * dim), dtype=dtype, device=device
            )
            args[name] = backing[::2, ::2].copy_(args[name])
    elif mode == "strided_table":
        backing = torch.empty((batch * 2, pages * 2), dtype=page_dtype, device=device)
        args["block_tables"] = backing[::2, ::2].copy_(args["block_tables"])
    return args


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(shape, id=f"dspark-draft-b{len(shape[4])}-kv{shape[4][0]}")
        for shape in perf_shape
    ]
    + edge_shape,
)
def test_paged_mtp_attention(shape, dtype, causal):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data(shape, dtype)
    executor = check_accuracy(
        paged_mtp_attention_decode,
        kwargs=dict(args, causal=causal),
        standard=standard,
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and dtype == torch.bfloat16
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "mode", ["zero_query", "strided_query", "strided_cache", "strided_table"]
)
@pytest.mark.parametrize("page_dtype", [torch.int32, torch.int64])
def test_layouts_and_zero_query(dtype, causal, mode, page_dtype):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data((7, 4, 64, 128, (0, 129), 4), dtype, mode, page_dtype)
    executor = check_accuracy(
        paged_mtp_attention_decode,
        kwargs=dict(args, causal=causal),
        standard=standard,
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_gqa_matches_fia(dtype, causal):
    import torch_npu
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data((7, 4, 64, 128, (129, 2049), 4), dtype)
    executor = check_accuracy(
        paged_mtp_attention_decode,
        kwargs=dict(args, causal=causal),
        standard=standard,
    )
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu_(1) if causal else None
    fia, _ = torch_npu.npu_fused_infer_attention_score(
        args["query"],
        args["k_cache"],
        args["v_cache"],
        block_table=args["block_tables"],
        block_size=128,
        num_heads=16,
        num_key_value_heads=4,
        input_layout="TND",
        atten_mask=mask,
        scale=64**-0.5,
        actual_seq_lengths=[7, 14],
        actual_seq_lengths_kv=[129, 2049],
        sparse_mode=3 if causal else 0,
    )
    assert_accuracy(executor(), fia, standard)


@pytest.mark.parametrize("lengths", [(129, 0), (2049,), (16007,) * 16])
def test_graph_replay_reads_changed_inputs(lengths):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    impl, impl_ref = get_test_impls()
    require_test_impl(paged_mtp_attention_decode, impl)
    if impl == "torch_native":
        pytest.skip("The host-reading reference does not support NPU graph capture")
    capacity = max(256, max(lengths))
    args = generate_data((7, 4, 64, 128, (capacity,) * len(lengths), 4), torch.bfloat16)
    original_table = args["block_tables"].clone()
    executor = check_accuracy(
        paged_mtp_attention_decode,
        kwargs=dict(args, causal=True),
        standard=standard,
    )
    for _ in range(3):
        executor()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = executor()
    for current in (
        lengths,
        (17,) * len(lengths),
        (0,) * len(lengths),
        (129,) * len(lengths),
    ):
        args["seq_lens"].copy_(torch.tensor(current, dtype=torch.int32))
        table = original_table.clone()
        for b, length in enumerate(current):
            if length == 0:
                table[b].fill_(-1)
        args["block_tables"].copy_(table)
        graph.replay()
        expected = paged_mtp_attention_decode(**args, causal=True, impl=impl_ref)
        assert_accuracy(output, expected, standard)
        for b, length in enumerate(current):
            if length == 0:
                assert torch.count_nonzero(output[b * 7 : (b + 1) * 7]).item() == 0
    args["query"].mul_(1.5)
    args["k_cache"].mul_(0.75)
    args["v_cache"].mul_(-0.5)
    table = original_table.clone()
    table[:, :2] = table[:, :2].flip(1)
    args["block_tables"].copy_(table)
    graph.replay()
    expected = paged_mtp_attention_decode(**args, causal=True, impl=impl_ref)
    assert_accuracy(output, expected, standard)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [48, 64])
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param((7, 15, 17), id="multiple-requests"),
        pytest.param((2049,), id="split-kv"),
    ],
)
def test_irregular_group_output_stays_in_bounds(dtype, causal, head_dim, lengths):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    impl, impl_ref = get_test_impls()
    require_test_impl(paged_mtp_attention_decode, impl)
    if impl == "torch_native":
        pytest.skip("This regression checks Triton output stores")
    args = generate_data((7, 3, head_dim, 16, lengths, 2), dtype)
    expected = paged_mtp_attention_decode(**args, causal=causal, impl=impl_ref)
    original_empty_like = torch.empty_like
    guarded_outputs = []
    guard_tokens = 32
    sentinel = 123.0

    def allocate_output(tensor, *positional, **kwargs):
        if (
            tensor.ndim != 3
            or tensor.shape[:2] != args["query"].shape[:2]
            or tensor.dtype != dtype
            or tensor.device != args["query"].device
        ):
            return original_empty_like(tensor, *positional, **kwargs)
        # Guard the output allocation, then validate the public result below.
        tokens, heads, dim = tensor.shape
        storage = torch.full(
            (tokens + 2 * guard_tokens, heads, dim),
            sentinel,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        guarded_outputs.append((storage, tokens))
        return storage[guard_tokens : guard_tokens + tokens]

    with patch.object(torch, "empty_like", side_effect=allocate_output):
        output = paged_mtp_attention_decode(**args, causal=causal, impl=impl)
    torch.npu.synchronize()
    assert guarded_outputs, "Triton output allocation was not intercepted"
    for storage, tokens in guarded_outputs:
        assert torch.all(
            storage[:guard_tokens] == sentinel
        ).item(), "Triton wrote before the output tensor"
        assert torch.all(
            storage[guard_tokens + tokens :] == sentinel
        ).item(), "Triton wrote past the output tensor"
    assert_accuracy(output, expected, standard)


@pytest.mark.parametrize("shape", [perf_shape[0], (7, 3, 48, 16, (7, 15, 17), 2)])
def test_triton_does_not_read_lengths_on_host(shape):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data(shape, torch.bfloat16)
    expected = paged_mtp_attention_decode(**args, causal=True, impl="torch_native")
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
        patch.object(
            torch.Tensor, "item", side_effect=AssertionError("unexpected item")
        ),
        patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("unexpected tolist")
        ),
    ):
        output = paged_mtp_attention_decode(**args, causal=True, impl="triton")
    assert_accuracy(output, expected, standard)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softmax_scale", [0.0, 0.25, -0.125])
@pytest.mark.parametrize("max_kv_len", [0, 1, 8192])
def test_scale_and_scheduling_bound(causal, softmax_scale, max_kv_len):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data(perf_shape[0], torch.bfloat16)
    # Even a zero scheduling hint must not truncate device lengths. An oversized
    # hint creates empty split-KV partitions, which must reduce without NaNs.
    args.update(softmax_scale=softmax_scale, max_kv_len=max_kv_len, causal=causal)
    executor = check_accuracy(
        paged_mtp_attention_decode, kwargs=args, standard=standard
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["auto", "torch_native"])
@pytest.mark.parametrize(
    "argument,value,message",
    [
        ("gamma", 0, "positive query width"),
        ("gamma", True, "must be integers"),
        ("num_q_heads", 7, "integral GQA groups"),
        ("num_kv_heads", 0, "integral GQA groups"),
        ("page_size", 15, "power-of-two page size"),
        ("page_size", 48, "power-of-two page size"),
        ("head_dim", 15, "head dimension"),
        ("head_dim", 257, "head dimension"),
        ("max_kv_len", -1, "max_kv_len >= 0"),
        ("max_kv_len", 1.5, "must be integers"),
        ("softmax_scale", float("nan"), "finite softmax_scale"),
        ("softmax_scale", float("inf"), "finite softmax_scale"),
        ("causal", 1, "boolean causal flag"),
        ("query", None, "inputs must be tensors"),
    ],
)
def test_rejects_invalid_arguments(impl, argument, value, message):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data((7, 4, 64, 128, (129, 17), 4), torch.bfloat16)
    args[argument] = value
    with pytest.raises((TypeError, ValueError), match=message):
        paged_mtp_attention_decode(**args, impl=impl)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["auto", "torch_native"])
@pytest.mark.parametrize(
    "invalid",
    [
        "length_rank",
        "length_dtype",
        "length_stride",
        "offset_shape",
        "offset_stride",
        "table_rank",
        "table_batch",
        "table_dtype",
        "query_shape",
        "query_dtype",
        "cache_rank",
        "cache_shape",
        "cache_stride",
        "cache_dtype",
        "different_device",
    ],
)
def test_rejects_invalid_tensor_metadata(impl, invalid):
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    args = generate_data((7, 4, 64, 128, (129, 17), 4), torch.bfloat16)
    if invalid == "length_rank":
        args["seq_lens"] = args["seq_lens"][None, :]
    elif invalid == "length_dtype":
        args["seq_lens"] = args["seq_lens"].long()
    elif invalid == "length_stride":
        args["seq_lens"] = torch.stack([args["seq_lens"]] * 2, dim=-1)[:, 0]
    elif invalid == "offset_shape":
        args["cu_q_lens"] = args["cu_q_lens"][:-1]
    elif invalid == "offset_stride":
        args["cu_q_lens"] = torch.stack([args["cu_q_lens"]] * 2, dim=-1)[:, 0]
    elif invalid == "table_rank":
        args["block_tables"] = args["block_tables"].flatten()
    elif invalid == "table_batch":
        args["block_tables"] = args["block_tables"][:1]
    elif invalid == "table_dtype":
        args["block_tables"] = args["block_tables"].float()
    elif invalid == "query_shape":
        args["query"] = args["query"][:-1]
    elif invalid == "query_dtype":
        args["query"] = args["query"].float()
    elif invalid == "cache_rank":
        args["k_cache"] = args["k_cache"][None, :]
    elif invalid == "cache_shape":
        args["v_cache"] = args["v_cache"][:-1]
    elif invalid == "cache_stride":
        cache = args["k_cache"]
        backing = torch.empty(
            (*cache.shape[:-1], cache.shape[-1] * 2), dtype=cache.dtype
        )
        args["k_cache"] = backing[..., ::2].copy_(cache)
    elif invalid == "cache_dtype":
        args["v_cache"] = args["v_cache"].half()
    elif invalid == "different_device":
        args["block_tables"] = args["block_tables"].cpu()
    with pytest.raises(ValueError, match="Paged MTP attention"):
        paged_mtp_attention_decode(**args, impl=impl)


@pytest.mark.extra_checks
def test_public_registration():
    from sgl_kernel_npu.operators import get_impl, list_impls
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

    assert list_impls(paged_mtp_attention_decode) == ["auto", "torch_native", "triton"]
    assert get_impl(paged_mtp_attention_decode, "auto") is get_impl(
        paged_mtp_attention_decode, "triton"
    )
    assert get_impl(paged_mtp_attention_decode, "triton") is not get_impl(
        paged_mtp_attention_decode, "torch_native"
    )
    args = generate_data(perf_shape[0], torch.bfloat16)
    with pytest.raises(ValueError, match="Unknown impl"):
        paged_mtp_attention_decode(**args, impl="unknown")


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "soc_family,packed_query,small_tile,wide_kv",
    [
        pytest.param(220, True, 32, 64, id="a2-arch22"),
        pytest.param(250, True, 32, 64, id="a3-arch22"),
        pytest.param(260, False, 16, 128, id="a5-arch35"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((7, 3, 16, 16, (7, 15, 17), 2), id="small-head-and-page"),
        pytest.param((8, 4, 256, 128, (8, 127, 256), 2), id="wide-head"),
    ],
)
def test_device_family_launch_config(
    soc_family, packed_query, small_tile, wide_kv, shape
):
    """Simulate device detection and capture launches; no foreign kernel runs."""
    import importlib

    import torch_npu
    from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode
    from sgl_kernel_npu.operators.utils import generation

    backend = importlib.import_module(
        "sgl_kernel_npu.implementation.triton.attention.paged_mtp_attention"
    )
    api = importlib.import_module(
        "sgl_kernel_npu.operators.attention.paged_mtp_attention"
    )
    args = generate_data(shape, torch.bfloat16)
    with patch.object(torch_npu._C, "_npu_get_soc_version", return_value=soc_family):
        arch_id = generation._arch_from_soc()
    hardware = generation.Hardware(arch_id, generation.generation_from_arch(arch_id))
    with (
        patch.object(api, "get_hardware", return_value=hardware),
        patch.object(backend, "get_device_properties", return_value=(24, 48)),
        patch.object(backend, "_paged_mtp_kernel") as kernel,
    ):
        output = paged_mtp_attention_decode(**args, impl="triton")
    kernel.__getitem__.assert_called_once_with((24,))
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    inputs, config = launch.call_args
    assert config["PACKED_QUERY"] is packed_query
    assert config["SPLITS"] == 1
    assert config["Q_TILE"] == 32
    dim, page = shape[2:4]
    assert config["BLOCK_D"] == (small_tile if dim == 16 else 256)
    assert config["BLOCK_N"] == (small_tile if page == 16 else wide_kv)
    assert config["KV_BLOCK"] == (16 if page == 16 else wide_kv)
    if packed_query:
        query = inputs[0]
        assert query.shape == (3, 2, shape[0], shape[1], dim)
        assert query.is_contiguous()
        # A packed KV-head matrix must retain each query's original GQA heads.
        for batch in range(3):
            for kv_head in range(2):
                torch.testing.assert_close(
                    query[batch, kv_head, -1],
                    args["query"][
                        (batch + 1) * shape[0] - 1,
                        kv_head * shape[1] : (kv_head + 1) * shape[1],
                    ],
                    atol=0,
                    rtol=0,
                )
    else:
        assert inputs[0] is args["query"]
    assert output.shape == args["query"].shape
    assert output.dtype == args["query"].dtype
    assert output.device == args["query"].device
