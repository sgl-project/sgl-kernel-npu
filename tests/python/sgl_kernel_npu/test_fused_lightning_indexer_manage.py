"""LIM metadata correctness and replay using only the normal kernel wheel."""

import pytest
import torch

pytest.importorskip("torch_npu")
import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import fused_lightning_indexer_manage
from utils import require_npu_op

pytestmark = require_npu_op("fused_lightning_indexer_manage")


def test_native_schema_has_no_scale_inputs():
    schema = torch.ops.npu.fused_lightning_indexer_manage.default._schema
    names = [argument.name for argument in schema.arguments]
    assert len(names) == 18  # 17 tensors and full_visible
    assert not any("scale" in name for name in names)


def make_case(batch=1, heads=32, dtype=torch.bfloat16, source_capacity=8320):
    torch.npu.set_device(0)
    device = "npu:0"
    capacity, prefix = 4096, 8192
    query = torch.zeros(batch, heads, 128, dtype=dtype, device=device)
    query[..., 0] = 1
    weights = torch.full((batch, heads), 1 / heads, dtype=dtype, device=device)
    key = torch.zeros(source_capacity // 128, 128, 1, 128, dtype=dtype, device=device)
    table = torch.arange(
        source_capacity // 128, dtype=torch.int32, device=device
    ).repeat(batch, 1)

    vector = lambda value: torch.full((batch,), value, dtype=torch.int32, device=device)

    pool = torch.full((batch, source_capacity), -1, dtype=torch.int32, device=device)
    ids = torch.empty(batch, 2048, dtype=torch.int32, device=device)
    misses = torch.empty(batch, 32768, dtype=torch.int32, device=device)
    args = [
        weights,
        query,
        key,
        table,
        torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        vector(prefix + 1),
        vector(prefix),
        vector(capacity),
        vector(-2),
        torch.arange(batch, dtype=torch.int32, device=device),
        pool,
        ids,
        torch.empty_like(ids),
        vector(0),
        misses,
        torch.empty_like(misses),
        vector(0),
    ]
    return args, torch.full((batch, capacity), -1, dtype=torch.int32)


def set_hotset(args, offset):
    # Exactly K scores are positive, with a wide gap at the TopK boundary.
    # Their internal order is immaterial to the metadata/payload contract.
    keys = args[2].view(-1, 128)
    keys[:, 0] = -8
    keys[offset : offset + 2048, 0] = 8


def consume_plan(args, resident, offset):
    torch.npu.synchronize()
    ids, slots = args[11].cpu(), args[12].cpu()
    src, dst, counts = args[14].cpu(), args[15].cpu(), args[16].cpu()
    expected = torch.arange(offset, offset + 2048, dtype=torch.int32)
    for row in range(ids.shape[0]):
        count = int(counts[row])
        assert 0 <= count <= resident.shape[1]
        assert torch.all(
            (dst[row, :count] >= 0) & (dst[row, :count] < resident.shape[1])
        )
        assert dst[row, :count].unique().numel() == count
        resident[row, dst[row, :count].long()] = src[row, :count]
        torch.testing.assert_close(ids[row].sort().values, expected, rtol=0, atol=0)
        assert torch.all((slots[row] >= 0) & (slots[row] < resident.shape[1]))
        # The actual miss plan must reconstruct every returned source/slot pair.
        torch.testing.assert_close(
            resident[row, slots[row].long()], ids[row], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "length", [2048, 2049, 2175, 2176, 4097, 32767, 32768, 32769, 36864]
)
def test_full_visible_q1_includes_partial_page_and_warm_new_token(length):
    source_capacity = max(8320, (length + 2 + 127) // 128 * 128)
    args, _ = make_case(batch=4, source_capacity=source_capacity)
    args[7].fill_(2048)
    resident = torch.full((4, 2048), -1, dtype=torch.int32)
    for step in range(3):
        visible = length + step
        args[5].fill_(visible)
        args[6].fill_(visible)
        set_hotset(args, visible - 2048)
        fused_lightning_indexer_manage(*args)
        consume_plan(args, resident, visible - 2048)
        args[8].fill_(-1)


@pytest.mark.parametrize("query_width", [1, 2, 3, 4, 5, 8, 14])
def test_full_visible_mtp_causal_and_rewritten_residents(query_width):
    capacity = max(4096, query_width * 2048)
    length = max(16384, capacity + 128) + query_width
    source_capacity = (length + 127) // 128 * 128
    args, _ = make_case(source_capacity=source_capacity)
    resident = torch.full((capacity,), -1, dtype=torch.int32)
    device = args[1].device
    heads = args[1].shape[1]
    args[0] = args[0].expand(query_width, heads).contiguous()
    args[1] = args[1].expand(query_width, heads, 128).contiguous()
    args[4].fill_(query_width)
    args[5].fill_(length)
    args[6].fill_(length)
    args[7].fill_(capacity)
    args[11] = torch.empty(query_width, 2048, dtype=torch.int32, device=device)
    args[12] = torch.empty_like(args[11])
    args[13] = torch.zeros(query_width, dtype=torch.int32, device=device)
    keys = args[2].view(-1, 128)
    keys[:, 0] = -8
    keys[:2048, 0] = 1
    keys[length - query_width : length, 0] = 8

    for step in range(3):
        # Reuse the same logical positions after a rejected speculative draft.
        keys[length - query_width : length, 1] = step + 1
        fused_lightning_indexer_manage(*args, full_visible=True)
        torch.npu.synchronize()
        ids = args[11].cpu()
        slots = args[12].cpu()
        count = args[16].item()
        sources = args[14].cpu()[0, :count]
        destinations = args[15].cpu()[0, :count]
        assert sources.unique().numel() == count
        assert destinations.unique().numel() == count
        resident[destinations.long()] = sources
        for route in range(query_width):
            visible = length - query_width + route + 1
            assert ids[route].unique().numel() == 2048
            assert torch.all(ids[route] < visible)
            assert set(range(length - query_width, visible)).issubset(
                set(ids[route].tolist())
            )
            assert torch.all((slots[route] >= 0) & (slots[route] < capacity))
            torch.testing.assert_close(
                resident[slots[route].long()], ids[route], rtol=0, atol=0
            )
        if step:
            # Every rewritten resident must be present in the copy plan.
            pool = args[10].cpu()[0]
            plan = dict(zip(sources.tolist(), destinations.tolist()))
            for source in range(length - query_width, length):
                slot = int(pool[source])
                if slot >= 0:
                    assert plan[source] == slot
        args[8].fill_(-1)


def test_full_visible_mtp_large_steady_union_and_refresh():
    query_width = 8
    capacity = query_width * 2048
    length = 2 * capacity + 128 + query_width
    args, _ = make_case(source_capacity=(length + 127) // 128 * 128)
    device = args[1].device
    heads = args[1].shape[1]
    args[0] = args[0].expand(query_width, heads).contiguous()
    args[1] = torch.zeros(query_width, heads, 128, dtype=args[1].dtype, device=device)
    for route in range(query_width):
        args[1][route, :, route] = 1
    args[4].fill_(query_width)
    args[5].fill_(length)
    args[6].fill_(length)
    args[7].fill_(capacity)
    args[11] = torch.empty(query_width, 2048, dtype=torch.int32, device=device)
    args[12] = torch.empty_like(args[11])
    args[13] = torch.zeros(query_width, dtype=torch.int32, device=device)
    keys = args[2].view(-1, 128)
    keys[:, :query_width] = -8
    resident = torch.full((capacity,), -1, dtype=torch.int32)
    for step in range(2):
        keys[:, :query_width] = -8
        for route in range(query_width):
            start = step * capacity + route * 2048
            keys[start : start + 2048, route] = 8
        # Ensure each current speculative source is selected and stays resident.
        keys[length - query_width : length, :query_width] = 16
        fused_lightning_indexer_manage(*args, full_visible=True)
        torch.npu.synchronize()
        count = args[16].item()
        sources = args[14].cpu()[0, :count]
        destinations = args[15].cpu()[0, :count]
        assert 0 <= count <= capacity
        assert sources.unique().numel() == count
        assert destinations.unique().numel() == count
        assert torch.all((destinations >= 0) & (destinations < capacity))
        resident[destinations.long()] = sources
        ids, slots = args[11].cpu(), args[12].cpu()
        for route in range(query_width):
            assert torch.all(ids[route] < length - query_width + route + 1)
            torch.testing.assert_close(
                resident[slots[route].long()], ids[route], rtol=0, atol=0
            )
        if step:
            assert count > 8192
            pool = args[10].cpu()[0]
            plan = dict(zip(sources.tolist(), destinations.tolist()))
            for source in range(length - query_width, length):
                slot = int(pool[source])
                if slot >= 0:
                    assert plan[source] == slot
        args[8].fill_(-1)


def test_full_visible_query_width_capability():
    assert torch.ops.npu.lim_full_visible_max_query_width() == 14


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("batch", [1, 4])
def test_cold_warm_miss_and_request_reuse(dtype, heads, batch):
    args, resident = make_case(batch, heads, dtype)
    set_hotset(args, 0)
    fused_lightning_indexer_manage(*args)
    consume_plan(args, resident, 0)
    args[8].fill_(-1)
    set_hotset(args, 4096)
    fused_lightning_indexer_manage(*args)
    consume_plan(args, resident, 4096)
    # Reuse the request row while retaining its previous encoded pool contents.
    args[8].fill_(-2)
    resident.fill_(-1)
    set_hotset(args, 2048)
    fused_lightning_indexer_manage(*args)
    consume_plan(args, resident, 2048)


def test_graph_replay_reads_current_state_and_keys():
    args, resident = make_case()
    set_hotset(args, 0)
    fused_lightning_indexer_manage(*args)  # warm the persistent tiling cache
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused_lightning_indexer_manage(*args)
    args[10].fill_(-1)
    args[8].fill_(-2)
    graph.replay()
    consume_plan(args, resident, 0)
    args[8].fill_(-1)
    set_hotset(args, 4096)
    graph.replay()
    consume_plan(args, resident, 4096)


@pytest.mark.parametrize("capacity", [2048, 4096])
def test_graph_non_offload_to_offload_reset(capacity):
    """Cold reset must not mistake identity metadata for populated resident KV."""
    args, _ = make_case()
    args[7].fill_(capacity)
    args[5].fill_(384)
    args[6].fill_(256)
    args[8].fill_(-3)
    set_hotset(args, 4096)
    fused_lightning_indexer_manage(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused_lightning_indexer_manage(*args)

    resident = torch.full((1, capacity), -1, dtype=torch.int32)
    args[5].fill_(8193)
    args[6].fill_(8192)
    # SGLang records last_prefix=-1 for non-offload rows, so entry into the
    # sparse route uses -2. Native -1 instead assumes an existing resident map.
    args[8].fill_(-2)
    graph.replay()
    consume_plan(args, resident, 4096)

    # Repeat the lifecycle on the same captured addresses with a new hot set.
    args[5].fill_(384)
    args[6].fill_(256)
    args[8].fill_(-3)
    graph.replay()
    args[5].fill_(8193)
    args[6].fill_(8192)
    args[8].fill_(-2)
    set_hotset(args, 0)
    resident.fill_(-1)
    graph.replay()
    consume_plan(args, resident, 0)


def test_non_offload_short_row_and_invalid_dtype():
    args, _ = make_case()
    args[5].fill_(384)
    args[8].fill_(-3)
    fused_lightning_indexer_manage(*args)
    torch.npu.synchronize()
    ids, slots = args[11].cpu().view(-1), args[12].cpu().view(-1)
    valid = ids >= 0
    torch.testing.assert_close(
        ids[valid].sort().values, torch.arange(384, dtype=torch.int32)
    )
    torch.testing.assert_close(slots, ids)
    assert args[13].item() == args[16].item() == 0
    args[1] = args[1].float()
    with pytest.raises(RuntimeError, match="fp16 or bf16"):
        fused_lightning_indexer_manage(*args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
