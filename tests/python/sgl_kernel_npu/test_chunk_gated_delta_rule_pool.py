import pytest
import sgl_kernel_npu  # noqa: F401  registers npu ops before pytestmark
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.ops.npu namespace available
from utils import require_npu_op


pytestmark = require_npu_op("chunk_gated_delta_rule_pool")


def _make_strided_pool(
    num_slots=32, num_heads=4, head_dim=128, seed=0, head_pad=0, storage_offset=0
):
    device = "npu:0"
    torch.manual_seed(seed)
    values = 0.05 * torch.randn(
        num_slots,
        num_heads,
        head_dim,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    slot_stride = num_heads * head_dim * head_dim + 32
    head_stride = head_dim * head_dim + head_pad
    storage = torch.empty(
        num_slots * slot_stride + num_heads * head_stride,
        dtype=torch.bfloat16,
        device=device,
    )
    pool = storage.as_strided(
        values.shape,
        (slot_stride, head_stride, head_dim, 1),
        storage_offset=storage_offset,
    )
    pool.copy_(values)
    return pool


def _make_inputs(lengths, seed=1):
    device = "npu:0"
    torch.manual_seed(seed)
    total = sum(lengths)
    num_heads = 4
    head_dim = 128
    q = F.normalize(
        torch.randn(total, num_heads, head_dim, device=device), p=2, dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(total, num_heads, head_dim, device=device), p=2, dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    g = F.logsigmoid(torch.randn(total, num_heads, device=device)).float()
    beta = torch.rand(total, num_heads, device=device).to(torch.bfloat16)
    actual = torch.tensor(lengths, dtype=torch.int32, device=device)
    return q, k, v, g, beta, actual


def _legacy_step(pool, inputs, src_slots, dst_slots, has_initial_state=None):
    q, k, v, g, beta, actual = inputs
    initial_state = pool[src_slots.long()].contiguous()
    if has_initial_state is not None:
        initial_state = initial_state.clone()
        initial_state[~has_initial_state] = 0
    chunk_state = torch.empty(
        sum((int(length) + 63) // 64 for length in actual.cpu()),
        v.shape[1],
        v.shape[2],
        k.shape[2],
        dtype=v.dtype,
        device=v.device,
    )
    out, final_state = torch.ops.npu.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual,
        scale=k.shape[-1] ** -0.5,
        chunk_state=chunk_state,
        g=g,
    )
    pool[dst_slots.long()] = final_state
    return out, final_state, chunk_state


def _pool_call(pool, inputs, src_slots, dst_slots, has_initial_state, chunk_state):
    q, k, v, g, beta, actual = inputs
    return torch.ops.npu.chunk_gated_delta_rule_pool(
        q,
        k,
        v,
        pool,
        src_slots,
        dst_slots,
        beta=beta,
        actual_seq_lengths=actual,
        scale=k.shape[-1] ** -0.5,
        chunk_state=chunk_state,
        g=g,
        has_initial_state=has_initial_state,
    )


def test_shared_source_and_independent_destinations():
    torch.npu.set_device("npu:0")
    pool = _make_strided_pool()
    reference = pool.contiguous()
    inputs = _make_inputs([65, 7])
    src_slots = torch.tensor([5, 5], dtype=torch.int32, device="npu:0")
    dst_slots = torch.tensor([17, 29], dtype=torch.int32, device="npu:0")
    before = pool.clone()

    expected_out, expected_final, expected_h = _legacy_step(
        reference, inputs, src_slots, dst_slots
    )
    actual_h = torch.empty_like(expected_h)
    actual_out = _pool_call(pool, inputs, src_slots, dst_slots, None, actual_h)
    torch.npu.synchronize()

    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(pool[dst_slots.long()], expected_final, rtol=0, atol=0)
    torch.testing.assert_close(pool[5], before[5], rtol=0, atol=0)
    untouched = [slot for slot in range(pool.shape[0]) if slot not in (17, 29)]
    torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)


def test_fresh_padding_and_unselected_slots():
    torch.npu.set_device("npu:0")
    pool = _make_strided_pool(seed=2)
    reference = pool.contiguous()
    inputs = _make_inputs([5, 0, 7, 65], seed=3)
    src_slots = torch.tensor([5, -1, 5, 5], dtype=torch.int32, device="npu:0")
    dst_slots = torch.tensor([17, -1, 23, 29], dtype=torch.int32, device="npu:0")
    has_initial = torch.tensor(
        [False, False, True, True], dtype=torch.bool, device="npu:0"
    )
    before = pool.clone()

    expected_out, expected_final, expected_h = _legacy_step(
        reference, inputs, src_slots, dst_slots, has_initial
    )
    actual_h = torch.empty_like(expected_h)
    actual_out = _pool_call(pool, inputs, src_slots, dst_slots, has_initial, actual_h)
    torch.npu.synchronize()

    live_tokens = list(range(5)) + list(range(12, sum([5, 0, 7, 65])))
    torch.testing.assert_close(
        actual_out[live_tokens], expected_out[live_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_h[[0, 2, 3]], expected_h[[0, 2, 3]], rtol=0, atol=0
    )
    selected = torch.tensor([17, 23, 29], device="npu:0")
    expected = expected_final[[0, 2, 3]]
    torch.testing.assert_close(pool[selected], expected, rtol=0, atol=0)
    untouched = [slot for slot in range(pool.shape[0]) if slot not in (17, 23, 29)]
    torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)


def test_rejects_noncontiguous_destination_heads():
    pool = _make_strided_pool(head_pad=16)
    inputs = _make_inputs([4, 4])
    src_slots = torch.tensor([5, 5], dtype=torch.int32, device="npu:0")
    dst_slots = torch.tensor([17, 29], dtype=torch.int32, device="npu:0")
    chunk_state = torch.empty(2, 4, 128, 128, dtype=torch.bfloat16, device="npu:0")
    with pytest.raises(RuntimeError, match="destination pool slots must be contiguous"):
        _pool_call(pool, inputs, src_slots, dst_slots, None, chunk_state)

    unaligned = _make_strided_pool(storage_offset=16)
    with pytest.raises(RuntimeError, match="destination pool base and slot stride"):
        _pool_call(unaligned, inputs, src_slots, dst_slots, None, chunk_state)


def test_distinct_source_and_destination_pools():
    torch.npu.set_device("npu:0")
    src_pool = _make_strided_pool(seed=4, head_pad=16)
    dst_pool = _make_strided_pool(seed=5)
    reference = dst_pool.contiguous()
    inputs = _make_inputs([3, 65], seed=6)
    src_slots = torch.tensor([5, 5], dtype=torch.int32, device="npu:0")
    dst_slots = torch.tensor([17, 29], dtype=torch.int32, device="npu:0")
    src_before = src_pool.clone()
    dst_before = dst_pool.clone()

    initial_state = src_pool[src_slots.long()].contiguous()
    q, k, v, g, beta, actual = inputs
    expected_h = torch.empty(
        3, v.shape[1], v.shape[2], k.shape[2], dtype=v.dtype, device=v.device
    )
    expected_out, expected_final = torch.ops.npu.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual,
        scale=k.shape[-1] ** -0.5,
        chunk_state=expected_h,
        g=g,
    )
    actual_h = torch.empty_like(expected_h)
    actual_out = torch.ops.npu.chunk_gated_delta_rule_pool(
        q,
        k,
        v,
        dst_pool,
        src_slots,
        dst_slots,
        src_pool=src_pool,
        beta=beta,
        actual_seq_lengths=actual,
        scale=k.shape[-1] ** -0.5,
        chunk_state=actual_h,
        g=g,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(
        dst_pool[dst_slots.long()], expected_final, rtol=0, atol=0
    )
    torch.testing.assert_close(src_pool, src_before, rtol=0, atol=0)
    untouched = [slot for slot in range(dst_pool.shape[0]) if slot not in (17, 29)]
    torch.testing.assert_close(
        dst_pool[untouched], dst_before[untouched], rtol=0, atol=0
    )
    torch.testing.assert_close(reference, dst_before.contiguous(), rtol=0, atol=0)


def test_graph_replay_index_changes_shared_sources_and_slot_recycling():
    torch.npu.set_device("npu:0")
    initial_pool = _make_strided_pool(seed=7)
    pool = _make_strided_pool(seed=7)
    reference = initial_pool.contiguous()
    inputs = _make_inputs([4, 4], seed=8)
    src_slots = torch.tensor([5, 5], dtype=torch.int32, device="npu:0")
    dst_slots = torch.tensor([17, 29], dtype=torch.int32, device="npu:0")
    has_initial = torch.ones(2, dtype=torch.bool, device="npu:0")
    chunk_state = torch.empty(2, 4, 128, 128, dtype=torch.bfloat16, device="npu:0")

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        _pool_call(pool, inputs, src_slots, dst_slots, has_initial, chunk_state)
    torch.npu.synchronize()
    pool.copy_(initial_pool)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output = _pool_call(
            pool, inputs, src_slots, dst_slots, has_initial, chunk_state
        )
    torch.npu.synchronize()
    pool.copy_(initial_pool)

    steps = [
        ([5, 5], [17, 29]),
        ([17, 17], [23, 24]),
        ([23, 24], [17, 29]),
    ]
    for src, dst in steps:
        src_slots.copy_(torch.tensor(src, dtype=torch.int32, device="npu:0"))
        dst_slots.copy_(torch.tensor(dst, dtype=torch.int32, device="npu:0"))
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        expected_out, _, expected_h = _legacy_step(
            reference, inputs, src_slots, dst_slots, has_initial
        )
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(chunk_state, expected_h, rtol=0, atol=0)
        torch.testing.assert_close(pool, reference, rtol=0, atol=0)
